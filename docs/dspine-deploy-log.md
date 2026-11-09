# DSpine 部署日志——170hx sglang 0.5.20→0.5.22 升级 + DSpine 插件 + 推测解码服务冒烟

> 单号 task_9eb15b032d06 · 2026-11-09 · 依据 dspine-170hx-plan.md §7/§8/§9
> **最终状态:A(DSpine 服务健康+冒烟通过+spec_accept_length=3.5)**——8 枪迭代,过五关:ninja PATH→fa3 硬锁→8 并发帽→草稿 KV 必须 bf16→FP8 粗检索无 SM80 内核(打补丁)→主后端必须 fa3→Ampere FA 不吃 fp8 KV(全 bf16)

## 1. 前置验证(全部通过)

| 项 | 结果 |
|---|---|
| 回滚三件套 | `~/sglang-llm-venv.bak-0520` 9.1G ✓ · `~/deploy/launch_user.sh.bak` 1692B ✓ · `~/deploy/sglang-llm.service.bak` 369B ✓ |
| 草稿模型 | `/mnt/storage/models/llm/qwen3.8-27b-dspine-b16` model.safetensors 4,051,831,880B(4.05G)✓ |
| 升级前服务 | sglang-llm active,显存 57,907 MiB(与 58G 基线吻合)|
| 红线核对 | 内核 `6.17.0-35-generic` 全程未动;NVIDIA 驱动 610.43.03 未动 |

停服:`systemctl --user stop sglang-llm` → 显存归零(systemd 记 exit 1/SIGINT,属停服正常表现)。

## 2. 升级轮子清单(pip --dry-run --report 实际解析,23 个轮子)

```
flashinfer-python==0.7.1 (py3-none-any 纯Python轮)
nccl-extensions==0.1.0
nvidia-cudnn-cu13==9.24.0.43
nvidia-cudnn-frontend==1.32.2
nvidia-cutlass-dsl==4.8.0 (+libs-base/core/cu12/cu13 各4.8.0)
nvidia-nccl-cu13==2.30.7
quack-kernels==0.6.5
sentencepiece==0.2.1
sgl-deep-ep==0.1.2.post1
sgl-deep-gemm==0.2.2
sglang==0.5.22
sglang-kernel==0.4.9
tokenizers==0.23.2
torch==2.14.1 (cp312, 实测 +cu130 构建)
torchcodec==0.17.0
torchvision==0.29.2
transformers==5.19.0
triton==3.8.0
xgrammar==0.2.7
```

**flashinfer CUDA 构建版本核对(装前)**:0.7.1 是 `py3-none-any` 纯 Python 轮,CUDA 件靠运行时 JIT + `cu13` extra 依赖(nvidia-cutlass-dsl[cu13] 4.8.0、nccl-cu13 2.30.7 等),**无固定 CUDA 构建版本,与 /usr/local/cuda-13.0 无冲突**。torch 装完实测 `2.14.1+cu130`,与 cuda_env.sh 指的 13.0 精确对齐。

## 3. 安装与三连验证(全部通过)

- `pip install "sglang==0.5.22" "torch==2.14.1" "flashinfer-python==0.7.1" ninja` → 成功(ninja 1.13.2 原已满足)
- `git clone https://github.com/Tencent/DSpine.git ~/DSpine`(HEAD 0cc92d8, 2026-11-08)→ `pip install -e ~/DSpine` → sglang-dspine 0.2.0 editable 装好
- ① `import torch; torch.cuda.init()` → `torch 2.14.1+cu130 cuda_ok True NVIDIA CMP 170HX` ✓
- ② `python -m sglang.check_env` → 正常跑完 ✓
- ③ `import sglang_dspine` → `sglang_dspine 0.2.0 OK` ✓

## 4. 启动脚本 ~/deploy/launch_dspine.sh(参数全文)

```bash
#!/bin/bash
# DSpine 投测服务:sglang 0.5.22 + DSPINE 草稿 tencent/Qwen3.8-27B-DSpine-b16
# 参数按 dspine-170hx-plan.md §8 官方示例 × SM80/FP8 适配:
#   fa3→flashinfer(两处)、mamba 64→16 槽 fp32、mem-frac 0.80、kv fp8_e4m3、context 显式 131072
source "$HOME/deploy/cuda_env.sh"
export PATH="$HOME/sglang-llm-venv/bin:$PATH"
exec "$HOME/sglang-llm-venv/bin/python" -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 30020 \
  --trust-remote-code \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path /mnt/storage/models/llm/qwen3.8-27b-dspine-b16 \
  --speculative-num-draft-tokens 16 \
  --attention-backend flashinfer \
  --speculative-draft-attention-backend flashinfer \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 16 \
  --mem-fraction-static 0.80 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 131072 \
  --chunked-prefill-size 2048 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report --allow-auto-truncate --log-level info
```

## 5. 冒烟矩阵过程

### 第①枪(首次启动)——失败,环境问题已修复

日志 `/tmp/dspine-launch-1.log`:`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`。
调用链:load_model → FP8 process_weights_after_loading → gptq_marlin_repack → sglang JIT → ninja 子进程。原因:脚本首版漏了 `export PATH="$HOME/sglang-llm-venv/bin:$PATH"`(cuda_env.sh 不含 venv bin,ninja 装在 venv 里),已补一行重启。注意:此时模型加载已走到 FP8 marlin 重打包,栈本身兼容。

### 第②枪(补 PATH 后)——失败,插件硬锁 fa3,已定位绕法

日志 `/tmp/dspine-launch-2.log`:`ValueError: Native DSPINE SWA configuration requires explicit --speculative-draft-attention-backend fa3`。
源码定位 `~/DSpine/sglang_dspine/layers/attention.py:6`:`BACKENDS = frozenset(("fa3",))`——草稿滑窗注意力后端**只认字面 "fa3"**,flashinfer/triton 都过不了校验。
**关键转机**:sglang 0.5.22 的 fa3 后端实际 import 的是 `sglang.kernels.ops.attention.flash_attention`(sglang-kernel 0.4.9 自带包装,底层 flash-attn-4 cute 接口,**有 Ampere 支持**),不是裸 flash-attn 包的 Hopper-only FA3。SM80 实测滑窗 varlen 前向(8 头×128 维、window(-8192,0)、causal)跑通 ✓。
处置:`--speculative-draft-attention-backend flashinfer → fa3`(主 `--attention-backend flashinfer` 不动)。
插曲:一次误启动后用 spkill 清残留,把新进程一并清掉(日志尾 `_pickle.UnpoplingError: pickle data was truncated`),清干净后重启,无碍。

### 第③枪(草稿后端 fa3)——失败,插件限 8 并发,已补参数

底模 66 分片加载全部完成、显存冲到 29.5G 后,草稿 worker 初始化时报:
`ValueError: Qwen3.8-27B DSpine supports at most eight concurrent requests`(源码 `~/DSpine/sglang_dspine/runtime/worker.py:555`,optimized 采样路径硬校验 `max_bs > 8` 即拒)。
原因:脚本没带官方示例的 `--max-running-requests 8`,sglang 默认 48。
处置:按官方 Qwen3.8-27B 示例补三件套 `--max-running-requests 8 --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1`。

### 第④枪(8 并发帽)——失败,草稿 KV 池要求 BF16,已单独指定

草稿 worker 走到 `alloc_memory_pool` 时报:
`ValueError: Native DSPINE SWA candidate requires the standard dense BF16 MHATokenToKVPool; bounded/hybrid/quantized pools are unvalidated`(源码 `~/DSpine/sglang_dspine/runtime/worker.py:711`,校验对象是 **草稿池** `draft_model_runner.token_to_kv_pool`)。
原因:`--kv-cache-dtype fp8_e4m3` 全局生效,草稿池被连坐成量化池;`--speculative-draft-kv-cache-dtype` 未设(None)时草稿继承全局值。
处置:加 `--speculative-draft-kv-cache-dtype bfloat16`——草稿池单独 BF16 过校验,底模 KV 维持 fp8_e4m3 省显存(草稿仅 5 层,BF16 池增量小)。这正是方案预判的风险 R2 的变体:不是 FP8 权重不兼容(底模 66 分片 FP8 加载已通过),而是草稿 KV 池不许量化。

### 第⑤枪(草稿 KV bf16)——失败,FP8 粗检索要 SM89+,已打 SM80 补丁

底模/草稿/内存池全过,CUDA 图捕获预热时报:
`RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`
调用链:CUDA 图捕获 → 插件 hook `around_capture` → `DraftHead.retrieve` → `sglang_dspine/layers/fp8/coarse.py:44` 的 `torch.ops.aten._scaled_mm`。这是**草稿 LM head 的 FP8 粗检索**(全词表 248320×5120 先 FP8 粗筛 top-128,再 BF16 精排),SM80 无此内核。注:底模 FP8 权重本身没问题(0.5.22 在 SM80 走 marlin W8A16 回退,首枪 traceback 可证)。
**处置(插件最小补丁,~/DSpine editable 安装直接生效)**:`coarse.py` 的 `retrieve_and_rescore` 按算力门控——`cap < 8.9` 时粗检索改普通 BF16 matmul(`hidden @ original_weight.t()`),≥8.9 走原 FP8 路径不变。数值安全依据:最终分数永远出自 `rescore()` 的原始 BF16 head,粗检索只影响候选集;bf16 粗筛比 fp8 更准。代价:粗筛行数仅 max_bs×15=120 行,GA100 bf16 算力 ~1-2ms/步。原文件留档 `coarse.py.orig`,pycache 已清。

### 第⑥枪(SM80 补丁)——失败,SPS 验证要底模后端支持 ragged 图,主后端换 fa3

补丁生效(FP8 粗检索关过了),显存冲到 55.5G(权重+池全就位),`init_cuda_graphs` → `RaggedSpsCaps` 初始化时报:
`ValueError: The configured attention backend does not support ragged graphs`(源码 `~/DSpine/sglang_dspine/verification/ragged.py:98`,校验 `runner._get_attention_backend(...).supports_ragged_verify_graph`)。
原因:DSpine 的 SPS 验证路径要**底模**注意力后端支持 ragged 验证图;sglang 0.5.22 里带此旗标的是 fa3(`flashattention_backend.py:151`)/flashinfer-mla/deepseek-v4,普通 flashinfer 后端没有。Qwen3.8 混合线性架构下 hybrid 包装器会把旗标透传给 full-attention 部分。官方示例底模本来就是 fa3。
处置:主 `--attention-backend flashinfer → fa3`(草稿已是 fa3)。SM80 可行性前面已实测。

### 第⑦枪(主后端 fa3)——失败,Ampere FA 不吃 fp8 KV,底模 KV 退 bf16

`RuntimeError: FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type`
原因:fa3 后端(sglang-kernel flash attention,Ampere 路径)读 KV 池只支持 fp16/bf16,底模 `--kv-cache-dtype fp8_e4m3` 直接被拒。fa3(DSpine 必需,ragged 图)× fp8 KV(Ampere FA 不支持)二者互斥,**这是硬约束不是 OOM**,不属于方案预列的降级链,是被迫的 dtype 改动。
处置:去掉 `--kv-cache-dtype fp8_e4m3`(target KV 随模型走 bf16);`--speculative-draft-kv-cache-dtype bfloat16` 保留。代价:KV 容量减半(Qwen3.8 混合架构全注意力层占比小,0.80 池下预计仍够 131K 上下文,以日志 max_total_num_tokens 实测为准)。

### 第⑧枪(全 bf16 KV)——**成功,服务就绪** ✅

```
[15:31:31] Load weight end. target: Qwen3_5ForConditionalGeneration fp8-e4m3, 28.97 GB
[15:31:37] Load weight end. draft: DSpineDraftModel, 3.86 GB
[15:31:38] max_running_requests is capped to 3 by the mamba state cache (16 槽, 每请求 5 状态槽)
[15:31:38] Mamba Cache allocated: ssm_state 2.39GB (fp32, 16槽)
[15:31:40] KV Cache allocated: bf16, #tokens=174870, K 5.34GB + V 5.34GB
[15:34:12] max_total_num_tokens=174870, context_len=131072, available_gpu_mem=6.20 GB
[15:34:23] The server is fired up and ready to roll!
```
注:bf16 KV 下 token 预算 174,870(> 131,072,单请求全上下文无忧;8 并发全 131K 同时在线则超预算,靠排队/ retract 兜底)。mamba 16 槽被 ReplaySSM 验证多吃(每请求 5 槽)压到并发 3——这是 fp32 16 槽的实质代价,方案 §8 显存预算表已预判"槽=并发上限"。

## 6. 最终服务状态与显存实测(交付状态 A)

| 项 | 实测 |
|---|---|
| 服务 | `~/deploy/launch_dspine.sh` setsid 手动进程(pid 42258),port 30020,**保持运行中**;未挂 systemd(按单子要求) |
| 健康 | `/health` → 200 |
| 冒烟生成(英) | "The capital of France is" → " Paris.\nThe capital of Germany is Berlin.…" 连贯,48 tok / 0.54s |
| 冒烟生成(中) | "简单介绍量子计算的基本原理:" → 结构化科普文本,质量正常,200 tok 无乱码 |
| **spec_accept_length** | **3.5**(中文 200 tok 生成后;首次 48 tok 短测后读数为 0,指标按批次刷新,属正常) |
| spec_accept_rate | 0.0→有值(gauge 随批次刷新) |
| 显存 | **58,951 MiB / 65,536 MiB**(权重 28.97 + 草稿 3.86 + mamba 2.5 + KV 10.7 + 图/池/杂项) |
| 系统盘 | venv 升级后 + 约 3G(pip 缓存 10G 未清,可 `pip cache purge` 回收) |
| 内核/驱动 | 6.17.0-35-generic / 610.43.03 **零改动** ✓ |

**最终参数全文**(即 `~/deploy/launch_dspine.sh`,头部注释已同步真实差异):
```bash
--model-path /mnt/models/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
--host 0.0.0.0 --port 30020 --trust-remote-code \
--speculative-algorithm DSPINE \
--speculative-draft-model-path /mnt/storage/models/llm/qwen3.8-27b-dspine-b16 \
--speculative-num-draft-tokens 16 \
--attention-backend fa3 --speculative-draft-attention-backend fa3 \
--linear-attn-backend triton --enable-linear-replayssm-spec \
--mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
--max-mamba-cache-size 16 --mem-fraction-static 0.80 \
--speculative-draft-kv-cache-dtype bfloat16 \
--context-length 131072 --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
--min-free-slots-delay 1 --chunked-prefill-size 2048 \
--sampling-backend flashinfer \
--reasoning-parser qwen3 --tool-call-parser qwen3_coder \
--enable-metrics --enable-cache-report --allow-auto-truncate --log-level info
```

## 7. 遗留问题

1. **插件补丁依赖**:`~/DSpine/sglang_dspine/layers/fp8/coarse.py` 打了 SM80 bf16 粗检索回退(原文件在 `coarse.py.orig`)。`git pull` 升级插件会冲掉补丁——上游未修 SM80 支持前,升级需重打或先提 issue/PR 给 Tencent/DSpine(补丁 15 行,含算力门控,可直接贡献)。
2. **并发实际 3 不是 8**:mamba fp32×16 槽被 ReplaySSM 验证按每请求 5 槽吃掉。要恢复并发 8 需 `--max-mamba-cache-size 40+`(显存 +~3.6G,KV token 预算会再降)或 `--mamba-ssm-dtype bfloat16`(非官方,回滚数值风险)。未动,留控制台拍板。
3. **KV 丢了 fp8**:与现役 DFlash2(fp8 KV,256K 上下文)相比,KV 容量减半,上下文显式压到 131072,token 总预算 174,870。长上下文(>131K)请求会被拒/截断(有 --allow-auto-truncate 兜底)。
4. **spec_accept_length 3.5 的口径**:温度 0、单条中文 200 tok 的粗测,非官方基准口径(官方 27B bf16:GSM8K T0 7.20)。FP8 底模是否拉低接受率,需按方案 §9 bench_serving 正式对比(DFlash2 基线已不可再测,现役服务本单已停)。
5. systemd 单元 `sglang-llm` 仍指向 launch_user.sh(DFlash2),现为 failed/stopped 状态,未动。切回 DFlash2:`systemctl --user start sglang-llm`(先停本 DSpine 进程);DSpine 转正式:新写 unit 或改 ExecStart 指向 launch_dspine.sh,由控制台定。
6. 日志:/tmp/dspine-launch-{1..8}.log 在 170hx 上(1-7 为失败枪,8 为成功枪),/tmp 重启即失,需要存档趁开机拷走。
