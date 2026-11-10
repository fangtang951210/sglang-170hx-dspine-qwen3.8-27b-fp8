# DSpine 256K 极限上下文实测(170hx · CMP 170HX 64G)

日期:2026-11-10 · 工单:170hx DSpine 极限上下文测速 · 服务:sglang 0.5.22 + DSPINE 草稿(Qwen3.8-27B-DSpine-b16-fullctx,滑窗 262144 修复版)

## 一句话结论

**256K 拉满可行但代价惨重**:最终以 `mf 0.90 + mamba 8 槽 + chunked-prefill 1024 + skip-server-warmup` 起到 KV 池 262688 token,单流 261K 请求两口径全部跑通;但 decode 速度腰斩(realtext 106.8→51.8 tps,tool 46.0→23.2 tps),投机接受率从 0.27 崩到 0.037(16 枪几乎全废),显存峰值 64905 MiB(64G 卡剩 631 MiB)。**这台机的 DSpine 长上下文甜点区在 100K 附近;256K 属于"能跑不划算"。**

## 降级链全过程(每级 KV 池 / 显存 / 结局)

| attempt | mem-frac | mamba 槽 | chunked | 其他 | KV 池(token) | 稳态显存 | 结局 |
|---|---|---|---|---|---|---|---|
| 1 | 0.80 | 16 | 2048 | — | 170239 | 59118 MiB | ✓ ready(对照:131072 档池 174870) |
| 2 | 0.85 | 16 | 2048 | — | 209304 | 62310 MiB | ✓ ready |
| 3 | 0.90 | 8 | 2048 | — | **262688** | — | ✗ 启动 warmup(带 mm 数据的请求触发 qwen_vl processor 初始化)吃爆 3.88G 余量,CUDA OOM 崩 |
| 4 | 0.90 | 8 | 2048 | +skip-server-warmup | **262688** | — | ✓ ready,但 261K 请求 prefill 中 OOM 崩(6.5MiB free,24MiB 分不出) |
| 5 | 0.90 | 8 | 1024 | +max-run/graph-bs 降 2 | 264454 | — | ✗ 插件形状 bug(factorized sampler `view(45,16)` vs 480),**max-running/graph-bs 不可降** |
| 6(定稿) | 0.90 | 8 | 1024 | +skip-server-warmup | **262688** | 峰值 64905 | ✓ ready + 261K 请求跑通,两口径测完 |

- 工单降级链原方向(mem-frac 往下 0.80→0.75)是给"OOM 起不来"准备的;实际卡点是反向的——**起来了但 KV 池不够**,所以 mem-frac 往上抬(0.85/0.90)+ mamba 降槽(省 1.26G)+ chunked 2048→1024(砍 prefill workspace,attempt4→attempt6 的救命一招)。
- KV 每 token 成本(主+草稿, bf16)≈ 82 KB;mf 每 +0.05 ≈ +39K token 池。
- skip-server-warmup 是超出工单链的必要补丁:warmup 请求走多模态 processor 初始化,在 3.9G 余量下必炸;跳过后纯文本请求不受影响。
- 环境注:ComfyUI 自启常驻占 ~450-820 MiB,全程未动。

## 极限档实测(单流,concurrency 1,output 512)

words 按 token/词比折算(realtext 1.1516、tool 1.008),prompt+512 不超池 262688:

| 口径 | prompt_tokens | prefill tps | decode tps | TTFT |
|---|---|---|---|---|
| 真实语料 227000 词 | 261383 | 1134.5 | **51.8** | 230.4 s |
| 工具随机词 259300 词 | 261268 | 1128.2 | **23.2** | 231.6 s |

## 与 100K 档对照(dspine-final 基准,131072 稳定态)

| 指标 | 100K 档 | 256K 档 | 变化 |
|---|---|---|---|
| 真实语料 prefill | 1564.6 tps | 1134.5 tps | ↓27% |
| 真实语料 decode | 106.8 tps | 51.8 tps | **↓52%** |
| 真实语料 TTFT | 73.6 s | 230.4 s | 3.1× |
| 工具 decode | 46.0 tps | 23.2 tps | **↓50%** |
| 工具 TTFT | 61.9 s | 231.6 s | 3.7× |
| spec accept_length | 5.075 | **1.55** | ↓69% |
| spec accept_rate | 0.272 | **0.037** | 崩到 1/7 |
| 显存峰值 | ~62.3 GB | **64.905 GB** | 贴边(剩 631 MiB) |

## 结论

1. **这台机 DSpine 的长上下文硬极限 = 262144(context)+ KV 池 262688**,由 `mf 0.90 + mamba 8 + chunked 1024 + skip-warmup` 达成;再无余量(峰值剩 631 MiB),262K 请求是数学极限,不是工程甜点。
2. **代价**:prefill -27%、decode 腰斩、TTFT 3 倍;根因是投机接受率崩塌(0.27→0.037)——超长上下文下草稿模型与主模型分歧拉大,16 枪草稿几乎全废,decode 退化为近裸速度还背验证开销。
3. **工程红线**:`--max-running-requests`/`--cuda-graph-max-bs-decode` 不能从 8 下调(DSpine 插件 factorized sampler 形状假设,attempt5 实锤);mamba 槽 8 = 并发上限 1(5 槽/请求),256K 配置只够单流。
4. **建议**:日常留在 131072 稳定态(已恢复并重启,health 200);确需 256K 单流长文档再切 `launch_dspine_256k.sh`,用完切回。

## 留档(170hx 原始位置 → 本仓库)

- 256K 定稿配置:`~170hx:~/deploy/launch_dspine_256k.sh` → 已收录 [`../patch/launch_dspine_256k.sh`](../patch/launch_dspine_256k.sh)(后续 YARN 修复叠加后即终态)
- 稳定态(已恢复):`~/deploy/launch_dspine.sh`(131072 原版,health 200)
- 启动日志:`~/deploy/logs/dspine-256k-attempt{1..6}.log`、`dspine-restore-131072.log`(机器侧,未收录)
- 显存全程采样(2s 粒度):`~/deploy/logs/dspine-256k-gpu.log`(峰值 64905;机器侧,未收录)
- 测速明细:`~/deploy/bench-tool-results.jsonl` tag `dspine-256k-*` → 已追加进 [`../data/bench-tool-results.jsonl`](../data/bench-tool-results.jsonl);测前后 metrics:`~/deploy/logs/dspine-256k-metrics-{before,after}.txt`(机器侧,未收录)

> 后续:本篇的接受率崩塌(0.27→0.037)在次日定位到第二层根因(草稿 rope 3072 训练位外推)并以 YARN 修复,256K 档恢复 100K 甜点档水平——见 [`dspine-256k-optimize.md`](dspine-256k-optimize.md)。
