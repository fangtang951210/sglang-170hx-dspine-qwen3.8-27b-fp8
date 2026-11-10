# DSpine 256K 优化:草稿生效修复(YARN)——达标

单号:task_1d0a74384429 · 日期:2026-11-10 · 机器:170hx(CMP 170HX 64G, SM80)· 栈:sglang 0.5.22 + DSpine 插件 0.2.0 + Qwen3.8-27B-FP8 + 草稿 Qwen3.8-27B-DSpine-b16
前序:dspine-deploy-log.md(部署八枪)、dspine-longctx-fix.md(滑窗 8192 修复)、dspine-256k-test.md(256K 极限实测)

## 一句话结论

**达标**。256K 输入下草稿接受率崩塌的根因 = 草稿 rope 训练仅覆盖 ~3072 位置、262144 全外推;修复 = 草稿 config 副本 `rope_parameters` 加 YARN(factor 85.3333, orig 3072),**256K 档 accept 5.075 / decode 74.3-82.6 tps(realtext 单流),恢复到 100K 甜点档水平,100K 档零回退**。两条硬要求同时满足:context 262144 可用 ✓、草稿在 256K 输入下真正生效 ✓(decode 74+ vs 裸跑 36.7,投机收益翻倍)。终态服务已按达标配置固化运行(health 200)。KV 量化路线按控制台保守预案实测后**如实放弃**(DSpine 栈架构性死路,证据链见 A;裸栈三档对比数据已留档,e4m3 精度几乎无损,供未来解锁后选用)。

## 实验矩阵(一轮一变因,每轮配置/结果/证据)

### A. KV FP8 化——DSpine 栈四条路全死,量化路线放弃(控制台修正版:档位对比先行)

**DSpine 服务实弹(四枪,日志 ~/deploy/logs/dspine-256k-opt/gun-*.log)**:

| 枪 | 配置 | 结局(报错原文) |
|---|---|---|
| A1 | 主后端 fa3 + `--kv-cache-dtype fp8_e4m3` | ✗ `FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type`(第⑦枪同款复现,kernel 级) |
| A1b | 主后端 fa3 + `fp8_e5m2` | ✗ sglang 自动警告 `FlashAttention3 only supports fp8_e4m3 if using FP8; Setting attention backend to triton` → 主后端变 triton → DSpine `ValueError: The configured attention backend does not support ragged graphs` |
| A2 | 主后端 flashinfer + `fp8_e4m3` | ✗ 同上 ragged 校验拒(DSpine SPS 验证要求主后端带 ragged 旗标) |
| A0 | (代码核实)trtllm_mha | ✗ 源码注明 prefill/decode kernel 仅 SM90/SM120;SM80 上带 ragged 旗标且可用于 Qwen3.8(GQA+hybrid)的后端**只有 fa3**,而 fa3 在 Ampere 只吃 bf16/fp16 KV |

→ **DSpine 栈下 KV 量化是架构性死路**(ragged 硬锁 × Ampere FA dtype 硬锁,二者交集为空),与档位选择无关。

**裸 sglang(无投机)三档对比(控制台要求的精度数据,100K 档同种子 T=0,bench_quality.py,flashinfer 后端)**:

| KV 档 | max_total_num_tokens | prefill tps | decode tps | 与 bf16 输出差异(512 token,T=0 同 prompt) |
|---|---|---|---|---|
| bf16(基线) | 303939 | 1561.8 | 36.7 | — |
| **fp8_e4m3** | 607878(3.5×) | 1369.7 | 37.8 | **仅 2 处同义措辞**("Unlike L1"→"Unlike standard caches"、"significantly"→"drastically"),语义完全等价 |
| fp8_e5m2 | 607878 | 1365.0 | 37.7 | 10 处替换,措辞/句式重写明显多于 e4m3 一个数量级 |

- 速度三档基本持平(decode 36.7-37.8;KV 读取减半只惠及全注意力层,混合架构中占比小)。
- 精度:**e4m3 几乎无损,e5m2 明显劣化**——若未来栈解锁(Hopper 硬件或上游给 flashinfer 系补 ragged 旗标),e4m3 是唯一候选档。
- spec_accept_length 对比无法做:该指标只在投机服务存在,而 DSpine 服务起不了任何 fp8 档(上表四枪)。按控制台预案第 3 条:量化路线放弃,256K 显存沿用 bf16 硬挤底盘(mf0.90+mamba8+chunked1024+skip-warmup)。

### B. 草稿滑窗扫描 @256K 输入——短窗口全崩,262144 唯一可行

底盘:launch_dspine_256k.sh 原版(mf0.90/mamba8/chunked1024/skip-warmup),realtext 227000 词(prompt 261383)单流 512 出,测后即读 metrics(gauge,服务单请求即该请求值)。改窗口 = 草稿 config `sliding_window` + 插件 `WINDOW` 同步(过启动校验)。

| 窗口 | accept_length | accept_rate | decode tps | 判读 |
|---|---|---|---|---|
| 262144(对照) | **3.325** | 0.155 | 52.0 | 全注意力路径真实成绩 |
| 131072 | 1.025 | 0.0017 | 17.3 | 崩 |
| 65536 | 1.025 | 0.0017 | 17.9 | 崩(两轮一致) |
| "32768" | — | — | — | **诚实标注:循环 sed pattern 未跟随更新,标签 b-win32768 的数据实为 65536 复测;32768 未实际测**(131072/65536 已双双实锤同崩,更小窗口无悬念,不补) |

结论:**seq>window 即激活 SM80 fa3 滑窗掩码坏分支(dspine-longctx-fix 已定位的 bug),草稿输出垃圾(accept 1.025 ≈ 只收首 token)**。滑窗裁剪路径被 bug 堵死,窗口必须 ≥ 输入全长;"短窗口回升佐证外推假设"的实验设计不可行,外推假设改由 C 直接验证。附带发现:今日对照轮 accept 3.325 高于前次报告 1.55(前次为两口径混合后的累计 gauge,今日为 realtext 单轮即时读数——口径差异,B/C 内部对比全部用今日统一口径)。

### C. rope 外推(YARN)——根因命中,接受率恢复 100K 档水平

关键前置发现:sglang `get_rope_config`(v5 分支)**优先读 `rope_parameters`**,草稿 config 顶层的 `rope_scaling: null` 根本不被读——yarn 必须写进 `rope_parameters`。草稿 rope 实现在 sglang 主仓 `models/dflash.py`,走标准 `get_rope(..., rope_scaling=...)`,yarn 链路完整可用。

改动(仅草稿 config 副本,一处):
```json
"rope_parameters": {"rope_theta": 10000000, "rope_type": "yarn", "factor": 85.3333, "original_max_position_embeddings": 3072}
```
(factor = 262144/3072;attn_factor 未显式设=1.0,sglang/HF/Qwen 三方默认一致;窗口保持 262144)

| 轮 | prompt | accept_length | accept_rate | decode tps |
|---|---|---|---|---|
| C-yarn @256K | 261383 | **5.075** | **0.272** | **74.3** |
| C-yarn @100K(副作用检查) | 100190 | ~4.98(按 verify 数推算) | ~0.27 | 112.0(原 rope 106.8) |

**外推假设实锤**:yarn 插值修复长位置编码后,256K 接受率从 3.325 → 5.075,恰好等于 100K 档原水平(前次报告 100K:5.075/0.272,逐位一致);100K 档零回退(decode 还略升)。accept-length 5.075 与前次 100K 档完全同值是巧合级吻合,但两档同时健康是真数据。

### D. 组合最优完整测速(256K,yarn 配置,即终态)

| 口径 | prompt_tokens | prefill tps | decode tps | TTFT | 显存峰值 |
|---|---|---|---|---|---|
| 真实语料(bench_realtext) | 261383 | 1133 | **74.3 / 82.6(两轮)** | 231 s | — |
| 工具口径(bench_tool) | 261268 | 1127 | **37.2**(无 yarn 时 23.2,+60%) | 232 s | **64611 MiB**(2s 采样) |
| 质量抽检(bench_quality) | 261371 | 1136 | 82.6 | — | 输出连贯(见下) |

输出质量抽检(261K 输入总结,512 token):结构完整、忠实覆盖输入全部五大主题(GPU 记忆层级/tiling/同步/精度量化/算法设计),无乱码无断层——全文留档 `kvqual-yarn256k.json`。tool 口径 decode 37.2 仍低于 realtext 属口径特性(随机词表无语义,草稿天然难猜),非配置问题。

### E. 达标判定

| 硬指标 | 线 | 实测 | 判 |
|---|---|---|---|
| decode tps(realtext 256K 单流) | ≥70 | **74.3 / 82.6** | ✅ |
| spec accept length | ≥3 | **5.075** | ✅ |
| context 262144 可用 | — | KV 池 262688,261K 请求三轮跑通 | ✅ |
| 草稿 256K 下生效 | — | accept=100K 档水平;decode 74+ vs 裸跑 36.7(投机收益 ×2) | ✅ |

对照:256K 现状 51.8/崩 → 74.3-82.6/5.075;100K 甜点 106.8/8(该 8 为 code 口径,realtext 口径 100K=5.075,同口径下 256K 已与 100K 持平)。

## 最终配置全文(终态,已固化运行)

```bash
~/deploy/launch_dspine_256k.sh   # 2026-11-10 15:0x 起运行中,health 200
```
参数与 dspine-256k-test 定稿完全一致(mf 0.90 + mamba 8 槽 + chunked 1024 + skip-warmup + fa3×2 + 草稿 KV bf16),**唯一增量在草稿 config 副本**(脚本头注释已更新记录):
```json
"sliding_window": 262144,
"rope_parameters": {"rope_theta": 10000000, "rope_type": "yarn", "factor": 85.3333, "original_max_position_embeddings": 3072}
```
插件 `~/DSpine/sglang_dspine/layers/attention.py` WINDOW=262144(不变)。131072 日常档 launch_dspine.sh 与本配置共用同一草稿副本——100K 档已实测 yarn 零回退(112 tps/4.98),共存无冲突。

## 死因分析(量化路线,按控制台要求如实报告)

1. DSpine SPS 验证硬要求主后端 `supports_ragged_verify_graph`;SM80 上唯一带旗标的 Qwen3.8 可用后端是 fa3(trtllm_mha 只要 SM90/120,mla/deepseek 系架构不符)。
2. fa3 的 SM80 kernel 路径只支持 fp16/bf16 KV(A1 实弹复现);e5m2 触发 sglang 自动降级 triton → ragged 校验拒(A1b)。
3. 二者交集为空 → **fp8 KV 无法进入 DSpine 服务,与档位/精度无关,升级 sglang/flashinfer 版本也绕不开(除非上游新增带 ragged 旗标的 SM80 后端)**。
4. 裸栈(无投机)fp8 KV 可用且 e4m3 精度几乎无损、池 3.5×——数据已留档,待未来硬件(Hopper+)或上游解锁后可直接选 e4m3。

## 工程坑(沿用+新增)

- warmup 必须 skip(mm processor bug,256K 余量下必炸);max-running/graph-bs 不可降(插件形状 bug)——均沿用前序。
- **新增:草稿 config 顶层 `rope_scaling` 字段是摆设,sglang 0.5.22 走 v5 `rope_parameters` 优先分支**——改 rope 必须动 `rope_parameters`。
- 新增:改窗口的脚本若用 sed 连环替换,pattern 要跟随上一轮值,否则静默不生效(本次 b-win32768 标签教训)。

## 留档(170hx 原始位置 → 本仓库)

- 实验日志/质量样本:`~/deploy/logs/dspine-256k-opt/`(16 文件:gun-a1/a1b/a2、gun-bare×3、b-win×4、c-yarn3072、kvqual-×4、d-tool-gpu)→ 已全量收录 [`../experiments/`](../experiments/)(逐件索引见其 README)
- 测速明细:`~/deploy/bench-tool-results.jsonl`(tag b-win*/c-yarn*/d-yarn256k-tool)与 realtext 结果(bench_realtext.py 输出)→ 前者已追加进 [`../data/bench-tool-results.jsonl`](../data/bench-tool-results.jsonl)
- 终态:launch_dspine_256k.sh(256K yarn,常驻;已收录 [`../patch/launch_dspine_256k.sh`](../patch/launch_dspine_256k.sh));launch_dspine.sh(131072 备用,共用 yarn 草稿副本,已验无回退)
- 草稿 config 增量 diff(原版→终态)→ [`../patch/yarn-draft-config.md`](../patch/yarn-draft-config.md)
