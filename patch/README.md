# 补丁总览

本目录收录把 Tencent/DSpine(sglang-dspine 插件 0.2.0)跑通在 **CMP 170HX(SM80/Ampere)+ FP8 底模**上所需的全部改动。均为实际部署中逐枪验证过的真实 diff,非示意代码。

| # | 补丁 | 文件 | 触发问题 | 性质 |
|---|---|---|---|---|
| 1 | [0001-long-context-sliding-window-fix.patch](0001-long-context-sliding-window-fix.patch) | 草稿 `config.json` 副本 + 插件 `sglang_dspine/layers/attention.py` | seq>8192 后草稿提案全废(accept≈1.0、decode 崩到裸跑以下) | **核心修复** |
| 2 | [0002-sm80-fp8-coarse-retrieval-bf16-fallback.patch](0002-sm80-fp8-coarse-retrieval-bf16-fallback.patch) | 插件 `sglang_dspine/layers/fp8/coarse.py` | `torch._scaled_mm` 要求算力 ≥8.9,SM80 无此内核,启动即崩 | SM80 适配 |

应用顺序:先打 0002(否则服务起不来),再打 0001(否则长上下文废)。两处改动均在插件仓库 `~/DSpine`(editable 安装)内直接生效,改完重启服务。

## 0001 长上下文滑窗修复(核心)

**根因**(完整证据链见 `docs/dspine-longctx-fix.md`):草稿模型 5 层全部是 `sliding_attention`,`sliding_window=8192`。seq≤8192 时滑窗掩码覆盖全部历史,等价普通因果注意力,草稿正常;seq>8192 后 FA 滑窗掩码分支被激活,在 sglang 0.5.22 + SM80(sgl-kernel fa3 路径)上对该草稿的注意力输出为垃圾 → 16 个提案全被目标模型拒绝。

**实锤证据:崩塌边界随窗口值移动**——8192 窗口时边界精确落在 8192;草稿窗口放大到 131072 后,8300/9229/16384/32768/100000 全档恢复(accept 1.0 → 5.8~16.0)。

**改法(两处必须同步,缺一会被插件启动校验拒绝)**:

1. 草稿 config 副本:`sliding_window: 8192 → 262144`(= `max_position_embeddings`,131072 服务上下文内等价全注意力)。做法是建一个零拷贝目录:新 config.json + 其余文件(权重/LICENSE/README)symlink 指向原模型仓,`--speculative-draft-model-path` 指向该副本目录。DSpine 论文/MLX 移植证实草稿用绝对位置+相对 rope,窗口裁剪仅为省算力,放大无损。
2. 插件 `sglang_dspine/layers/attention.py`:`WINDOW = 8192 → 262144`(与 config 副本配套,通过其启动校验)。

**验证矩阵**(bench_lctx2.py,T=0,单流,output 512,code 语料 2 次重复):

| prompt tokens | 修复前 accept / decode tps | 修复后 accept / decode tps |
|---|---|---|
| 810 | 12.8–14.5 / 85–94 | 14.5 / 91.2 |
| 8192 | 2.46 / 46.2(过渡) | 15.0 / 224.4 |
| 9229 | 1.0 / 20–25 | 14.6 / 218.0 |
| 32768 | ~1.0 / ~20 | 11.6 / 82.7 |
| 100000 | ~1.0 / 23.9 | 8.15 / 132.4 |

realtext 语料同样全档平坦无崩塌(100K:4.38 / 82.8)。

**上游价值**:SM80 上 FA 滑窗掩码(sglang 0.5.22,sgl-kernel fa3 路径)对 DFlash/DSpine 类 16 行 q 的滑窗 decode 疑似内核级 bug;本仓库数据(边界=窗口值、窗口放大即愈、num_splits=1 无效)可直接作 issue 证据报 sgl-project/sglang 或 Tencent/DSpine。上游修复前 `WINDOW=262144` 补丁可长期顶用(草稿仅 5 层×8 KV 头,全注意力额外开销在 100K 档实测可忽略)。

**已排除的假修复**:`--speculative-draft-window-size` 被 DSPINE worker 硬拒(`DSPINE does not support --speculative-draft-window-size`),且该 flag 本身只是草稿 KV 显存紧凑化旋钮,不是质量旋钮;`--chunked-prefill-size 16384`(单 chunk prefill)启动 OOM;`--enable-deterministic-inference`(num_splits=1)实测无效。

## 0002 SM80 FP8 粗检索回退

草稿 LM head 的 FP8 粗检索(全词表 248320×5120 先 FP8 粗筛 top-128 再 BF16 精排)调用 `torch.ops.aten._scaled_mm`,该算子要求算力 ≥8.9(SM89/SM90+),CMP 170HX(SM80)没有,CUDA 图捕获预热时直接 `RuntimeError` 崩溃。

**改法**:按算力门控——`cap < 8.9` 时粗检索退普通 BF16 matmul(`hidden @ original_weight.t()`),≥8.9 走原 FP8 路径不变。数值安全:最终分数永远出自 `rescore()` 的原始 BF16 head,粗检索只影响候选集,BF16 粗筛比 FP8 更准。代价:粗筛仅 max_bs×15=120 行,GA100 上 ~1-2ms/步,实测可忽略。原文件留档 `coarse.py.orig`。

注意:`git pull` 升级插件会冲掉本地补丁,升级需重打或先提 PR 给上游。
