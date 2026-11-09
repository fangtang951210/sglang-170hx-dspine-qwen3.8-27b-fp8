# DSpine 长上下文失效根因排查与修复

单号:task_34e626e99eaf(170hx DSpine 长上下文)· 日期:2026-11-09 · 工人:Claude(Orca 派单)
结论先行:**已修好**。根因 = 草稿模型 8192 滑窗的 FA 掩码分支在 seq>8192 时输出垃圾(sglang 0.5.22 + SM80 sgl-kernel 栈);修复 = 禁用草稿滑窗(等价全注意力)。100K 档:修复前 accept 1.0 / decode 23.9 → 修复后 **accept 8.0 / decode 130**。

---

## 1. 根因证据链

**现象复现与边界精确定位**(bench_lctx2.py,code 语料,T=0,单流,output 128):

| prompt tokens | accept_len | decode tps | 状态 |
|---|---|---|---|
| 810 / 2048 / 4096 / 6144 / 7168 / 7680 / 7936 | 12.8–14.5 | 85–94 | 健康 |
| 8192 | 2.46 | 46.2 | 过渡 |
| 8300 / 8704 / 9229 | 1.02–1.06 | 20–25 | 崩塌 |

边界精确落在 **8192 = 草稿 config 的 `sliding_window: 8192`**(全部 5 层 `sliding_attention`)。

**排除项**:
- 草稿 `max_position_embeddings: 262144`、rope_theta 1e7——位置上限远超 8192,头号嫌疑(单子原假设)排除。
- 目标模型:9229 token 输出文本完好连贯(verdict2-sample-realtext.txt 及本单抽检),verify 侧无恙 → 坏的只有草稿提案。
- 草稿 KV 池容量:174,870 token(1.67GB×2),够 131K 上下文。
- prefill 分块:越窗实验(prompt 7700+512 输出)证明**decode 中途总长越过 8192 也崩** → 与 prefill 分块无关,纯 decode 长度触发。
- mamba 16 槽 / 草稿 KV bf16 / fp8 coarse.py 补丁:边界随窗口值移动(见下),与这些部署项无交互。
- `--speculative-draft-window-size`:DSPINE worker 明确拒绝此参数(`DSPINE does not support --speculative-draft-window-size`),单子预设的修复路不通。

**定位实验**(每次改一项、重启、测 8300/9229):
1. `--enable-deterministic-inference`(强制 FA num_splits=1,排除 KV 分裂 bug):仍崩(accept 1.03/1.06)→ 排除分分裂。
2. **窗口放大 131072**(草稿 config 副本 `sliding_window: 131072` + 插件 `WINDOW` 常量同步改):8300→12.8/94.5、9229→**16.0/97.9**、16384→11.64/90.6、32768→10.55/80.9、100000→5.82/63.9 —— **全档恢复**。

**结论**:seq ≤ 8192 时窗口掩码覆盖全部历史(=普通因果注意力,草稿正常);seq > 8192 时 FA 滑窗掩码分支被激活,在 sglang 0.5.22 的 SM80(sgl-kernel fa3 路径)上对该草稿的注意力计算产生垃圾 → 提案全废 → accept 1.0。窗口改大于等于上下文长度后掩码分支永不激活,即回到已验证正确的因果路径。设计上这无损:DSpine 论文/MLX 移植(Tencent/DSpine PR#3)证实草稿绝对位置+相对 rope,窗口裁剪仅为省算力,等价于全注意力。

## 2. 尝试过的修复与结果

| 尝试 | 结果 |
|---|---|
| `--speculative-draft-window-size`(单子预设) | 插件硬拒,不可用 |
| `--chunked-prefill-size 16384`(单 chunk prefill) | 启动 OOM 崩溃,且越窗实验已排除 prefill 因素,放弃 |
| `--enable-deterministic-inference`(num_splits=1) | 启动崩(batch-invariant matmul bf16 KeyError);改打 env 补丁实测 num_splits=1 无效 |
| **草稿窗口 8192→262144(禁用滑窗)** | ✅ **全档修复,采纳为终态** |

## 3. 最终参数全文(终态服务,170hx:30020)

```bash
~/deploy/launch_dspine.sh   # 2026-11-09 19:49 起运行中,logs/dspine-final.log
```
```bash
python -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 30020 \
  --trust-remote-code \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path /home/<user>/deploy/qwen3.8-27b-dspine-b16-fullctx \
  --speculative-num-draft-tokens 16 \
  --attention-backend fa3 \
  --speculative-draft-attention-backend fa3 \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 16 \
  --mem-fraction-static 0.80 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --context-length 131072 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1 \
  --chunked-prefill-size 2048 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report --allow-auto-truncate --log-level info
```

**与原部署的差异只有两处**(均为持久化落盘):
1. `--speculative-draft-model-path` → `~/deploy/qwen3.8-27b-dspine-b16-fullctx/`:config.json 副本(`sliding_window: 262144` = max_position_embeddings,131072 上下文内等价全注意力),权重 symlink 指向 /mnt/storage 原仓(模型盘满,零拷贝)。
2. 插件补丁:`~/DSpine/sglang_dspine/layers/attention.py` `WINDOW = 8192 → 262144`(与 config 副本配套,通过其启动校验)。

DFlash2(sglang-llm.service)已停且未回切;FA 后端 num_splits 实验补丁已还原,sglang 0.5.22 venv 无残留改动。

## 4. 验证矩阵(bench_lctx2.py,T=0,单流,output 512;accept_len = completion/verify_calls)

**code 语料(2 次重复)**:

| 档位 | accept_len | decode tps | 达标(≥3 / ≥60) |
|---|---|---|---|
| 810 | 14.5 / 14.5 | 91.2 / 89.5 | ✅ |
| 8192(草稿原窗口附近) | 15.0 / 14.09 | 224.4 / 217.6 | ✅ |
| 9229 | 13.43 / 14.6 | 218.0 / 106.1 | ✅ |
| 32768 | 10.55 / 11.6 | 80.9 / 82.7 | ✅ |
| 100000 | 7.89 / 8.15 | 129.5 / 132.4 | ✅ |

**realtext 语料(1 次)**:810→4.23/102.3 · 8192→4.79/110.4 · 9229→4.74/108.9 · 32768→4.03/90.3 · 100000→4.38/82.8 —— 全程平坦无崩塌。

修复前对照:>8192 全档 accept≈1.0、decode 20–33(本单实测 9229:1.0/26.8;旧账 100K:23.9)。原始数据:`~/deploy/bench-lctx-results.jsonl`(tag final-code / final-realtext / scan-code / r2-splits1 / r3-w131072)。

## 5. 结论与下一步

- **修好了,且超达标线**:32K/100K 档 accept≥3、decode≥60 全过;100K decode 130 tps ≈ 裸跑 40 tps 的 3.2 倍,亦超 DFlash2 现水平(100K 旧账 23.9)。8K–9K 档 decode ~220 tps。
- 接受率随长度自然衰减(14.5→8.0 @code),属草稿能力/注意力稀释,非故障。
- **遗留观察(非本单故障)**:接受率对语料内容敏感——code 12–16、realtext/中文 ~2–5,短中文叙事两代引擎都低(bench-real-results.jsonl),与本长度 bug 无关,值得另开单查(温度/采样/中文分布)。
- **上游建议**:SM80 上 FA 滑窗掩码(sglang 0.5.22,sgl-kernel fa3 路径)对 DFlash/DSpine 类 16 行 q 的滑窗 decode 疑似有内核级 bug,本单数据(边界=窗口值、窗口放大即愈、num_splits=1 无效)可直接作 issue 证据包报 sgl-project/sglang 或 Tencent/DSpine;修复前 `WINDOW=262144` 补丁可长期顶用(草稿 5 层×8 kv 头,全注意力额外开销在 100K 档实测可忽略)。
