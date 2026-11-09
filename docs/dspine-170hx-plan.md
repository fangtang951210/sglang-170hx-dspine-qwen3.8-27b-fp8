# DSpine 接入 170hx 推理服务——调研+摸底+实施方案

> 单号 task_211fb390e976 · 2026-11-09 · 本单只读不写,170hx 所有数据来自实际 ssh 查询(命令随文附),执行由控制台做
>
> **一句话结论:能做,但要闯三关——①严格版本升级(sglang 0.5.20→0.5.22 + torch 2.13.0→2.14.1 + flashinfer 0.6.18→0.7.1 + DSpine 插件);②SM80 显卡必须把官方示例的 fa3 后端换成 flashinfer/triton(官方未测过);③FP8 底模+DSPINE 官方没背书,需实测。显存按 58G/64G 基线算,草稿模型 4.05G,塞得下但没余量,靠 mamba 槽位/上下文长度调参。**

---

## 1. DSpine 是什么(联网调研)

**DSpine = 腾讯开源的并行起草(parallel drafter)推测解码草稿模型框架**,论文《Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding》(arXiv 2609.36173,2026-10 放出)。

- **架构**:传统 EAGLE/MTP 是串行起草(一个 token 一个 token 猜);DFlash 类并行方法各位置独立预测、彼此不通信。DSpine 的创新是"**门控相邻注入**":每层把每个位置预测的特征传给相邻后继位置,让因果依赖在网络**深度方向**展开,而所有位置**同时并行**更新——兼得并行速度和条件依赖。仓库原文:"Each draft layer passes the predicted feature of each position to the next" / "token dependencies to be modeled during parallel drafting"。
- **Qwen3.8-27B 版草稿模型实测 config**(来源 HF `config.json`):
  - `architectures: ["DFlashDraftModel"]`,`model_type: qwen3`,5 层,hidden 5120(与底模同宽),32 头/8 KV 头,头维 128,MLP 17408
  - `block_size: 16`(一步起草 16 token,即型号名 b16);`sliding_window: 8192`;`max_position_embeddings: 262144`
  - `target_layer_ids: [1, 16, 31, 46, 61]`(挂载底模 64 层中的这 5 层取特征)
  - 词表 248320,含一个 ~1.27B 参数的 LM head(248320×5120),全模型 **2B 参数,bf16**
- **体积/显存**:HF 仓库单文件 `model.safetensors` **4.05 GB**(API 实查 4,051,831,880 字节)。bf16 加载≈4.05G 显存权重 + 少量激活/CUDA graph。对比现有 DFlash2 草稿(本地实测 safetensors 3.85G),**净增仅 ~0.2G**。
- **"xhigh/low 推理模式"辨析**:这不是 DSpine 的模式,是**底模 Qwen3.8 的 reasoning effort 档位**。官方 benchmark 按 `low` / `xhigh` 推理档分开测——xhigh 思考链长、decode 占比高,推测解码收益更大(见 §3)。DSpine 自身唯一可调项是 **SPS 动态验证表**(按草稿置信度动态分配验证预算,`docs/sps_tuning.md`,不调表则默认整 16-token 块验证)。
- **支持底模列表**(README):`Qwen3-8B`(草稿 tencent/Qwen3-8B-DSpine-b16,2026-10 发布)+ `Qwen3.8-27B`(草稿 tencent/Qwen3.8-27B-DSpine-b16,2026-11 支持,**权重已实际发布**——HF 仓库 2026-11-07 创建、公开可下、652 次下载;魔搭同步有镜像)。**Qwen3.8-27B 已覆盖 ✓**。
- **FP8 量化底模是否支持:官方无表态**。README 和 HF 卡的示例全用 `--dtype bfloat16` 底模,无一处提 FP8;issues(仅 2 个开放)也无相关讨论。代码里 `sglang_dspine/layers/fp8/` 只是草稿自己 LM head 的"FP8 粗检索+BF16 精排"加速(源码 docstring:"Row-scaled FP8 retrieval; final scores use the original BF16 head"),**与底模量化无关**。→ FP8 底模列为待实测项(§8 风险 R2)。
- **许可证**:**Apache 2.0**(GitHub License.txt + HF 卡 + pyproject `license = "Apache-2.0"`)✓。

来源:
- https://github.com/Tencent/DSpine (README)
- https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 (模型卡+config)
- https://arxiv.org/abs/2609.36173 (论文)

## 2. SGLang 集成要求

DSpine 不进 sglang 主干,是 **pip 插件**(`sglang-dspine` 0.2.0,entry point `sglang.srt.plugins` 组注册 `dspine = sglang_dspine.plugin:register`)。装完插件后 `--speculative-algorithm DSPINE` 即可用。

**硬性版本要求**(pyproject.toml 原文,注释自述"Built and tested against this exact release: the plugin subclasses SGLang internals"):
```toml
dependencies = ["sglang==0.5.22", "torch==2.14.1", "flashinfer-python==0.7.1", "ninja"]
requires-python = ">=3.11"
```
全是 **`==` 严格锁死**,不能部分升级。另 sglang 0.5.22 自身 breaking change:要求 Python ≥3.11、发布说明提到 transformers 5.18/5.19 适配(本地 5.12.1 大概率要连带升)。

**官方 Qwen3.8-27B 启动参数全文**(README/HF 卡一字不差):
```bash
sglang serve \
  --model-path /path/to/Qwen3.8-27B \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path tencent/Qwen3.8-27B-DSpine-b16 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 --dtype bfloat16 \
  --attention-backend fa3 --speculative-draft-attention-backend fa3 \
  --linear-attn-backend triton --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 64 --mem-fraction-static 0.80 \
  --context-length 131072 --max-running-requests 8 \
  --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --trust-remote-code
```

**与既有推测方案的参数差异**(对照 170hx 现役 DFlash2):
| 参数 | 现役 DFLASH | DSPINE(官方) | 说明 |
|---|---|---|---|
| `--speculative-algorithm` | DFLASH | DSPINE | 值变了,参数名不变 |
| `--speculative-num-draft-tokens` | 8 | **16** | 块起草 16 token,验证批翻倍 |
| `--speculative-draft-attention-backend` | (无) | **fa3**(新增) | 草稿滑窗注意力专用后端 |
| `--attention-backend` | flashinfer | **fa3** | ⚠️ FA3 要 SM90+,170HX 是 SM80,必须替换 |
| `--linear-attn-backend` | (默认) | **triton**(新增) | Qwen3.8 混合线性注意力,triton 在 SM80 可用 |
| `--enable-linear-replayssm-spec` | (无) | **新增** | 拒绝 token 的 SSM 状态回滚(ReplaySSM),混合架构推测解码必需 |
| `--mamba-ssm-dtype` | bfloat16 | **float32** | 官方要求 fp32(回滚数值安全),显存翻倍 |
| `--max-mamba-cache-size` | 12 | **64** | mamba 状态槽数,直接吃显存 |
| `--mem-fraction-static` | 0.85 | 0.80 | 官方给 27B 留了更大余量 |
| `--sampling-backend` | (默认) | **flashinfer**(显式) | DSPINE 稀疏验证路径依赖 |
| (可选)`--speculative-dspark-sps-table-path` | — | sps.json | SPS 动态验证预算表,调优产物 |
| `--kv-cache-dtype fp8_e4m3` | 有 | 官方示例未设 | 170HX 现役省显存手段,建议保留待实测 |

与 EAGLE/MTP 系的差异:EAGLE 系是串行自回归起草、草稿单层循环;MTP/NEXTN 用底模自带 MTP 头;DSpine 是**独立多位置并行草稿+深度方向条件注入**,且为混合线性架构(Qwen3.8)带了 ReplaySSM 状态回滚——这三家在 sglang 里互斥共用 `--speculative-algorithm` 开关。

来源:https://github.com/Tencent/DSpine/blob/main/pyproject.toml · https://github.com/sgl-project/sglang/releases/tag/v0.5.22(0.5.22 变更:DSpark 原生支持 #41623、DFlash 块验证 #43012、"requires Python 3.11 or newer"、混合线性模型推测解码+结构化输出的 mamba 状态错误修复 #43029)

## 3. 官方 benchmark 与复现方法

**"3.3×" 的出处**(HF 模型卡吞吐表,非论文):WorkBuddy Web、xhigh 推理档下 **DSpine 840.14 tok/s(3.33×),同场 DFlash2 734.04(2.91×),AR 基线 252.26**。全表(AR | DFlash2 | DSpine,tok/s,括号为对 AR 加速比):

| 场景 | low | xhigh |
|---|---|---|
| WorkBuddy Code | 686.23 (2.71×) | 613.55 (3.18×) |
| WorkBuddy Web | — | **840.14 (3.33×)** |
| WorkBuddy Office | 805.53 (3.84×) | — |
| SWE-bench Pro | — | 791.30 (3.19×) |
| τ²-bench | 964.30 (3.20×) | — |

**测试条件**(模型卡原文):SGLang 0.5.22 / **BF16 底模** / FlashAttention 3 / **并发 8**(`--max-running-requests 8`,配套 context 131072)。通用基准:xhigh 推理、贪心+温度采样、最多 4096 新 token;agent 基准:low+xhigh、温度采样、每子集 20 段对话、按录制的工具决策前缀回放、开前缀缓存、每请求最多 32768 新 token。吞吐口径:"total output tokens divided by end-to-end concurrent replay wall time"。**GPU 型号官方未写**——绝对值不可直接对标 170HX(HBM2e 带宽约为 H100 一半,预期绝对吞吐打对折上下,加速比可参考)。论文口径(8B 模型):接受长度 T0 均值 4.82 vs DFlash 3.77(+27.8%),对 DFlash 平均吞吐 +23.3%。

**复现方法**:官方未发 benchmark 脚本,最接近的官方工具是 `scripts/tune_sps.py`(SPS 调优,自带计时吞吐统计:前缀 JSONL 回放、宽度扫 4/8/12/16、每档 3 次计时取中位、默认生成 512 token)。通用测速用 sglang 自带 `bench_serving`(§9 给全套命令)。

来源:https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 · https://github.com/Tencent/DSpine/blob/main/docs/sps_tuning.md · https://arxiv.org/abs/2609.36173

## 4. 170hx sglang 现状(实测)

机器:2026-11-09 13:06 开机(WoL 刚醒),SSH 直连正常。**内核 `6.17.0-35-generic`** ✓(与 cmpunlocker 绑定一致,全程未动)。

```
$ ssh 170hx 'uname -r'        → 6.17.0-35-generic
$ nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
NVIDIA CMP 170HX, 8.0, 610.43.03        ← SM80(GA100/Ampere),驱动 610.43.03(CUDA UMD 13.3)
```
> **SM 8.0 是本单关键硬件事实:FA3 需要 SM90+(Hopper),官方示例的 `--attention-backend fa3` 在这张卡上必炸,必须替换(§8)。**

软件栈(venv `~/sglang-llm-venv`,Python 3.12.3):
```
$ ~/sglang-llm-venv/bin/pip list | grep -iE "sglang|flashinfer|torch|transformers|triton"
sglang 0.5.20 · sglang-kernel 0.4.7 · flashinfer-python 0.6.18
torch 2.13.0 · torchaudio 2.11.0 · torchvision 0.28.0 · transformers 5.12.1 · triton 3.7.1
cuda-toolkit 13.0.3 · cuda-python 13.4.1 · tokenspeed-triton 3.8.10 · torch_memory_saver 0.0.10
```
**与 DSpine 硬要求的差距:sglang 0.5.20→0.5.22、torch 2.13.0→2.14.1、flashinfer 0.6.18→0.7.1、transformers 5.12.1→(0.5.22 适配的 5.18/5.19 系)、补装 ninja + sglang-dspine 插件。** Python 3.12.3 满足 ≥3.11 ✓。

JIT 环境(`~/deploy/cuda_env.sh`,升级时要重验):系统 PATH 的 /usr/bin/nvcc 是 CUDA 12.0(坑1),venv 工具链 nvcc 13.4.92 与 flashinfer 0.6.18 自带 CCCL 只认 ≤13.3(坑2),**结论用 `/usr/local/cuda-13.0`(nvcc 13.0.48)**——flashinfer cu13 轮子按 CUDA 13.0 构建。flashinfer 换 0.7.1 后要重新核对轮子的 CUDA 构建版本是否仍与 13.0 对齐。

连通性(实测):`huggingface.co` HTTP/2 200 ✓ · `hf-mirror.com` 200 ✓ · `pypi.org` 200 ✓(升级/回滚不吃网络亏)。

## 5. 170hx 服务现状(实测)

**起法:systemd 用户单元 `~/.config/systemd/user/sglang-llm.service`(disabled,故意不自启)→ ExecStart=/home/<user>/deploy/launch_user.sh;当前 inactive**(与 ComfyUI 显存互斥,`switch-to-llm.sh` 先确认 ComfyUI 队列空→停 comfyui.service→等显存<2G→start sglang-llm)。无 docker 容器在跑。19997 是 comfy-panel(工具面板),一键启动链路=面板/脚本调 `systemctl --user start sglang-llm`。

**启动参数全文**(`launch_user.sh`,2026-09-19,头两行注释原文:*"SGLang Qwen3.8-27B-FP8 + DFlash2 投机解码" / "sglang 0.5.20 原生含 DFlash2DraftModel,无需补丁" / "显存账(64G 卡):权重29 + 草稿3.6 + mamba + KV + 图缓存 ≈ 0.85 池(~58G 预占)"*):
```bash
MODEL="/mnt/models/Qwen3.8-27B-FP8"    # 软链 -> /mnt/storage/models/llm/qwen3.8-27b-fp8
DRAFT="/mnt/storage/models/llm/qwen3.8-27b-dflash2"
PORT=30020
python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 30020 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT" \
  --speculative-num-draft-tokens 8 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 12 \
  --enable-metrics --enable-cache-report --allow-auto-truncate --log-level info
```
另有一个变体 `launch_obliterated.sh`(模型换 qwen3.8-27b-obliterated-fp8,同 DFLASH 草稿同端口)。**注意:现役配置没有 `--context-length`,吃模型默认 262144(256K)**——与控制台"256K 上下文"线索吻合。

**权重路径验证(控制台第 3 条线索:模型挪过窝)——两个路径都活着 ✓**:
```
$ readlink -f /mnt/models/Qwen3.8-27B-FP8   → /mnt/storage/models/llm/qwen3.8-27b-fp8
qwen3.8-27b-fp8: 29G(实文件,layers-*.safetensors 分片,2026-08-14)
qwen3.8-27b-dflash2: model.safetensors 3.85G + dflash2-drafter-f16.gguf 3.86G(共 7.2G,gguf 是冗余拷贝)
config.json 实测:model_type=qwen3_5,fp8 e4m3 dynamic,modules_to_not_convert 含 model.visual.*(视觉块不量化——该检查点是 VLM 形态 checkpoint)
```
**显存现状**:开机即查时 ComfyUI 占 448MiB,LLM 模式下会被停掉。58G 基线 = launch_user.sh 注释 + 控制台线索双源一致(journalctl 无历史——journal 易失,重启即丢,未找到实测日志佐证,如实说明)。磁盘:**`/mnt/models` 458G 已用 100%(仅剩 3.5G)**;`/mnt/storage`(fnos NAS)19T 空 15T ✓;`/` 系统盘 233G 空 85G ✓(venv 在这,8.7G)。

## 6. 模型下载路径规划

| 项 | 结论 |
|---|---|
| 从哪下 | **首选 HF 直连**(170hx 实测 HTTP/2 200):`tencent/Qwen3.8-27B-DSpine-b16`;备选 hf-mirror.com(实测 200);**魔搭镜像已确认存在**:`modelscope.cn/models/tencent/Qwen3.8-27B-DSpine-b16`(2026-11 上架) |
| 体积 | **4.05 GB**(单文件 model.safetensors + config/LICENSE/README 约 26KB)。另 DSpine 源码仓(git clone,含 CUDA 源码,几十 MB) |
| 下到哪 | **`/mnt/storage/models/llm/qwen3.8-27b-dspine-b16/`**(沿用现有目录约定,15T 空闲)。**严禁下到 /mnt/models(100% 满)** |
| 要不要中转 | **不用**。170hx 三路(HF/镜像站/PyPI)全通,不需要 90hx/nakyo 中转 |
| 命令 | `hf download tencent/Qwen3.8-27B-DSpine-b16 --local-dir /mnt/storage/models/llm/qwen3.8-27b-dspine-b16`(或 `huggingface-cli download`,venv 里已具备) |
| 顺带 | bf16 版底模 `qwen3.8-27b`(52G)本地已有——但 52G 权重+草稿+KV 在 64G 卡上塞不下(§8 预算表),只能走 FP8 底模路线 |

## 7. 升级路径(只出方案,不执行)

**目标**:sglang 0.5.20 → **0.5.22**(严格锁),连带 torch 2.13.0→2.14.1、flashinfer-python→0.7.1、transformers 5.12.1→5.18/5.19 系、sglang-kernel 随 sglang 解析、补 ninja、装 sglang-dspine 0.2.0(源码 `pip install -e`)。

**是否连带 CUDA/驱动变化**:torch 2.14.1 是 cu13 系轮子(pip 自带全套 nvidia-* 运行时),**驱动 610.43.03(CUDA 13.3)足够,内核/驱动零改动** ✓✓(红线:cmpunlocker 绑 6.17.0-35,两者都不碰)。风险点在 JIT:flashinfer 0.7.1 轮子的 CUDA 构建版本要与 `cuda_env.sh` 指的 `/usr/local/cuda-13.0` 对齐(0.6.18 时代踩过 CCCL 头不兼容的雷,见 §4),装完先跑一次小 JIT 冒烟再上服务。

**执行序列**(控制台照抄):
```bash
# ① 快照(回滚锚点,全部只读/追加)
~/sglang-llm-venv/bin/pip freeze > ~/deploy/venv-freeze-0.5.20.txt
cp -a ~/sglang-llm-venv ~/sglang-llm-venv.bak-0520        # 8.7G,/ 有 85G 空闲
cp ~/.config/systemd/user/sglang-llm.service ~/deploy/sglang-llm.service.bak
cp ~/deploy/launch_user.sh ~/deploy/launch_user.sh.bak    # 新配置另写 launch_dspine.sh,不覆盖旧的
# ② 升级(Python 3.12 ✓;uv 或 pip 均可,PyPI 实测通)
source ~/deploy/cuda_env.sh
~/sglang-llm-venv/bin/pip install "sglang==0.5.22" "torch==2.14.1" "flashinfer-python==0.7.1" ninja
~/sglang-llm-venv/bin/pip install -e /path/to/DSpine      # git clone https://github.com/Tencent/DSpine.git
# ③ 验证
~/sglang-llm-venv/bin/python -m sglang.check_env
~/sglang-llm-venv/bin/python -c "import sglang_dspine"    # 插件 entry point 注册验证
# ④ 冒烟:起服务(§8 参数),观察 JIT 编译 + 显存池日志,跑 test_generate
```
**风险与回滚**:
- R0 严格锁版:插件"subclasses SGLang internals",**不许部分升级/不许用 >0.5.22 的 sglang**;sglang 小版本升级可能破坏插件(关注 DSpine 仓更新)。
- R1 torch 2.13→2.14 大版本跨度:装完先 `python -c "import torch; torch.cuda.init()"` 冒烟。
- R2 JIT/CCCL 对齐问题重演(§4):flashinfer 0.7.1 若按 CUDA 13.1+ 构建,需要装对应 nvcc 或等 flashinfer 修;装前先查轮子 metadata。
- R3 transformers 5.12→5.18/5.19:模型加载行为可能变,冒烟时对照生成质量。
- **回滚**:`rm -rf ~/sglang-llm-venv && mv ~/sglang-llm-venv.bak-0520 ~/sglang-llm-venv`(或按 freeze 文件 pip 重装),unit/launch 脚本还原 .bak,模型文件全程未动——5 分钟内可完全回退。升级期间不影响 ComfyUI(venv 与显存都独立);回滚不需要动内核/驱动。

## 8. 集成方案(只出方案,不执行)

**预下载清单**:DSpine 源码仓(clone,~几十MB)+ 草稿模型 4.05G(§6)+ pip 轮子(torch 2.14.1 全家 ~8-10G 走 PyPI 缓存,系统盘 85G 够)。**合计新增磁盘 ~12-14G,全部落在 / 与 /mnt/storage,不碰 100% 满的 /mnt/models。**

**新启动参数全文**(`~/deploy/launch_dspine.sh`,官方 27B 示例 × 170HX 现实(SM80/FP8/64G)适配,改动处标 ★):
```bash
#!/bin/bash
# DSpine 接入:官方示例 + SM80 后端替换 + 沿用现役 FP8/KV 省显存配置
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
  --chunked-prefill-size 2048 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report --allow-auto-truncate --log-level info
# ★fa3→flashinfer(两处):SM80 无 FA3;若草稿滑窗注意力 flashinfer 报错,退 triton
# ★mamba-cache 64→16 + mem-frac 0.80:64 槽 fp32 在 64G 卡上太奢侈,先小后大
# ★float32 官方要求(回滚数值安全);若显存爆且接受率正常,可试回 bfloat16(非官方)
# ★context-length 未设(吃模型默认 256K);若 OOM 先显式压到 131072(官方值)再逐步放大
```
**冒烟矩阵**(按序,失败即停):①官方参数原样(除 fa3)context=131072 → ②flashinfer 草稿后端不通换 triton → ③通过后去 context-length 压力测 256K+调 mamba 槽 → ④(可选)`scripts/tune_sps.py` 调 SPS 表加 `--speculative-dspark-sps-table-path`。

**显存预算表(64G)**:
| 项 | 现役 DFlash2(实测口径) | DSpine 方案 | 说明 |
|---|---|---|---|
| 底模权重 FP8 | 29G | 29G(不变) | du 实测 |
| 草稿权重 bf16 | 3.85G | **4.05G**(+0.2) | HF API 实查 |
| mamba 状态池 | 12 槽 × bf16 | **16 槽 × fp32**(≈+2.7× 本项) | 官方 64 槽装不下,砍到 16;槽=混合层并发上限,≤ max-running-requests 即可 |
| KV cache fp8_e4m3 @256K | 计入 0.85 池 | 同左 | 未设 context-length,与现役同口径 |
| 验证批/激活/CUDA graph | 8-token 验证 | **16-token 验证**(↑) | cuda-graph-max-bs-decode 建议先 8(官方值) |
| 静态池占比 | 0.85(≈54.4G) | **0.80(≈51.2G)** | 先留 3G 余量给翻倍验证批 |
| **运行时总占用** | **~58G**(脚本注释+控制台线索) | **预估 58-61G,塞得下但零余量** | 首启看日志 `max_total_num_tokens`/avail mem 实测校准;OOM 降级顺序:mamba 槽 16→12 → context 256K→131072 → mamba fp32→bf16(最后手段) |

**风险清单**:R1 FA3→SM80 替换无官方背书(issues 里唯一硬件 issue #2 就是 Blackwell 上 flashinfer 兼容性,SM80 反而更成熟,但 DSpine 自带 CUDA kernel——conditional_topk/mixed_sum/sparse_support——要在 SM80 JIT 编译通过才作数);R2 **FP8 底模+DSPINE 官方零表态**(证据链:现役 DFLASH 在同栈 FP8 底模上稳定运行;草稿只吃底模层输出隐态(量化层输出仍是 bf16),逻辑上应通;但 ReplaySSM 新路径未验证);R3 显存零余量,mamba fp32 翻倍是最大变量;R4 官方 benchmark 全部 BF16+未知 GPU,3.33× 绝对值不可平移,以实测 AR 基线为准。

## 9. 测速方案(命令序列,可直接执行)

**原则:升级前先把基线钉死**(0.5.20 栈上测,升级后不可再得),同一组 prompt/并发/温度,三轮对比:AR(关推测)/ DFlash2(回退对照)/ DSpine(新),加第四轮 DSpine+SPS 表(可选)。

```bash
# ── A. 准备(一次性):固定语料,防前缀缓存干扰用 --flush-cache
# A1. 真实负载:现网抓 20 条典型请求(含工具调用/长上下文)存 prompts.jsonl
# A2. 合成负载:固定种子可控长度
# ── B. 基线测试(升级前,0.5.20)─────────────────────────
# B1. DFlash2 现状基线(服务照常起着)
~/sglang-llm-venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30020 \
  --dataset-name random --random-input-len 2048 --random-output-len 512 \
  --num-prompts 32 --concurrency 8 --temperature 0 \
  --output-file ~/deploy/bench-dflash2-pre.json
# B2. AR 基线(关推测):临时脚本去掉 3 个 speculative/mamba 相关 flag 起服务,同命令
#     --output-file ~/deploy/bench-ar-pre.json(测完恢复原服务)
# B3. 顺手留 xhigh/low 两档 reasoning effort 各测一轮(真实负载语料,对应官方口径)
# ── C. 升级后(0.5.22,§7 序列)─────────────────────────
# C1. DFlash2 回退对照:旧 launch_user.sh 起服务,同 bench 命令 → bench-dflash2-post.json
#     (验证升级本身没造成性能回退;0.5.22 对 DFlash 还有块验证优化,可能不降反升)
# C2. DSpine:launch_dspine.sh 起服务,同 bench 命令 → bench-dspine.json
# C3. (可选)SPS 调优后复测 → bench-dspine-sps.json
# ── D. 指标采集(每轮都记)──────────────────────────────
curl -s localhost:30020/metrics | grep -E "spec_accept_length|num_running_reqs"   # 接受率
nvidia-smi --query-gpu=memory.used,power.draw --format=csv -l 5 > mem.log &       # 显存峰值
# ── E. 对比表字段:输出吞吐(tok/s)/ TTFT p50,p99 / TPOT / spec_accept_length
#      / 显存峰值 / 对 AR 加速比;与官方 3.33×(WorkBuddy Web xhigh, 并发8)对参考不对绝对值
```
现役 `~/deploy/bench.sh`(前缀缓存命中 + 512-token 粗测)可作快速回归,不能作正式对比(无并发控制)。**判定门槛(建议,控制台拍板)**:DSpine 对 DFlash2 输出吞吐 ≥+15% 且接受长度 ≥5.5(官方 27B 口径 T0 GSM8K 7.20/DFlash2 5.69)才算值得切;若 FP8 底模下接受长度显著低于官方 bf16 口径(>20%),回退 DFlash2 并向 DSpine 提 issue。

---
### 附:数据佐证索引(170hx 实测命令)
`uname -r`→6.17.0-35-generic · `nvidia-smi --query-gpu=name,compute_cap,driver_version`→CMP 170HX, 8.0, 610.43.03 · `~/sglang-llm-venv/bin/pip list`→sglang 0.5.20/torch 2.13.0/flashinfer 0.6.18/transformers 5.12.1 · `systemctl --user cat sglang-llm`→ExecStart=launch_user.sh, inactive · `cat ~/deploy/launch_user.sh`→参数全文见 §5 · `readlink -f`+`du -sh`→权重路径全存活 · `df -h`→/mnt/models 100%, /mnt/storage 15T 空 · `curl -sI`→HF/hf-mirror/PyPI 全通 · journalctl 无历史显存日志(如实说明,58G 为脚本注释+控制台线索双源)
### 附:来源 URL
https://github.com/Tencent/DSpine · https://raw.githubusercontent.com/Tencent/DSpine/main/README.md · .../pyproject.toml · .../docs/sps_tuning.md · .../sglang_dspine/layers/fp8/coarse.py · https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16(+ /raw/main/config.json, /api/models/.../tree/main) · https://arxiv.org/abs/2609.36173 · https://github.com/sgl-project/sglang/releases/tag/v0.5.22 · https://www.modelscope.cn/models/tencent/Qwen3.8-27B-DSpine-b16 · https://github.com/Tencent/DSpine/issues/2
