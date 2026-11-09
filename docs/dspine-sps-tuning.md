# DSpine SPS 调优表实战报告(170hx)

日期:2026-11-09/10 · 机器:170hx(CMP 170HX 64G 解锁,SM80)· 服务:sglang 0.5.22 + DSpine 插件,Qwen3.8-27B-FP8 + Qwen3.8-27B-DSpine-b16(长上下文修复版草稿),端口 30020
结论一句话:**按官方流程实测调优出的 SPS 表对三档真实负载 decode 提升全部 <5%(8K -2.0% / 100K +3.3% / 4×56K 噪声),按判定规则已回退,终态 = DSpine 无表原配置健康运行。**
全部原始产物收录在本仓库 [`tuning/`](../tuning/)(调优表、校准语料、逐候选测量、服务日志、前后 bench 数据),本报告为过程与判定记录。

## 1. 官方调优流程要点(上游 DSpine 仓 docs/sps_tuning.md)

- 草稿 block 固定 16;调优对象 = **batch 1–7 的统一验证宽度 + batch 8 的总预算**(DSpark 按草稿置信度在预算内分配验证)。
- 输入:JSONL,每行 `{"input_ids": [...], "sampling_params": {...}}`,**至少 8 条不同 prefix**,覆盖真实上下文长度/任务/推理深度;**贪心(temperature=0)与温度采样必须分开两次调**;不支持受约束采样。
- 运行:独占一块空闲 GPU,`tune_sps.py --requests xx.jsonl --output sps.json -- <原样 serving 命令>`;host/port/SPS 表参数由调优器注入,每个候选宽度起一台独立测量服。
- 默认扫描:宽度 4/8/12/16 × batch 1–8;每候选先 warm 全部 prefix 再 3 轮计时,按**中位“总输出 token / 总请求墙钟”(含 prefill 与请求开销)**评分;选定组合再起一台服复验一次。响应长度默认 512。
- 产物:`sps.json`(选中策略+全部测量+硬件指纹+命令+语料摘要)、`sps.runs/`(日志与逐请求原始数据)。
- 加载:serving 命令加 `--speculative-dspark-sps-table-path /path/to/sps.json`。**换 GPU/模型/运行时/负载须重调;无表时验证整 16-token 块**。
- 本机落地偏差(有据):tune_sps.py 强制校验 `effective_max_running_requests_per_dp ≥ 最大 batch`;本机 mamba 16 槽/请求 5 槽 → 并发上限 3,故 `--batch-sizes 1 2 3`(官方默认 1–8 在本部署直接报错,bs≥4 生产也到不了);宽度/轮数/响应长度用官方默认。另:调优器起的服 JIT 依赖 venv bin 里的 ninja,须先 `export PATH="$HOME/sglang-llm-venv/bin:$PATH"`(踩过一次,已记)。

## 2. 调优输入与生成的表

**输入** [`tuning/sps-prefixes.jsonl`](../tuning/sps-prefixes.jsonl):8 条全贪心(temperature=0),共 310,872 tokens;语料=测速同源英文技术文章 + DSpine 仓真实 Python 源码,token 级定长,chat 模板烘进 input_ids:

| # | 类型 | tokens | 任务 |
|---|---|---|---|
| 1 | 文章 | 8,984 | 英文总结(对齐 8K 档) |
| 2 | 代码 | 8,984 | 讲解架构 |
| 3 | 混合 | 13,984 | 中文总结(真实用户用法) |
| 4 | 代码 | 19,984 | 写单测 |
| 5 | 混合 | 31,984 | 找 bug |
| 6 | 代码 | 47,984 | 续写实现 |
| 7 | 文章 | 63,984 | 英文总结(对齐 56K 档) |
| 8 | 混合 | 114,984 | 英文总结(对齐 100K 档) |

**产物** [`tuning/dspine-sps-table.json`](../tuning/dspine-sps-table.json)(format `dspine-sps-v1`,含 4 宽度 × 3 batch = 12 组测量 × 3 轮 + 1 轮组合复验):

- 选中策略:`uniform_widths = [12, 8, 16, 16, 16, 16, 16]`(bs1→宽12,bs2→宽8,bs3→宽16;未扫的 bs4–7 回落默认 16),`batch8_tokens = 128`
- 各宽度最优(中位总吞吐,该口径含全量 prefill):bs1 18.46@12 vs 18.39@16(+0.4%);bs2 19.75@8 vs 19.68@16(+0.3%);bs3 21.57@16 ≈ 21.55@12(并列取 16)
- 接受长度随宽度(bs1):2.85@4 / 3.66@8 / 3.83@12 / 3.87@16 —— 本负载接受长度本就 ~3.9,16 宽浪费不大,收窄收益自然小
- 注:测量口径每轮墙钟 ~200s 被 prefill 主导(310K token 语料池超 KV 缓存容量,LRU 淘汰导致跨队列不复用),宽度间 decode 差异被稀释,这与官方“warm 后计时有缓存命中”的预期场景不同,属本机缓存容量约束,如实记录

## 3. 前后三档对比(bench_realtext.py,v2.2 口径,输出 512,贪心,tag sps-前-/sps-后-)

| 档 | 指标 | 前(无表) | 后(带表) | 变化 |
|---|---|---|---|---|
| 单流 8K(9,229 tok) | decode tps | **139.4** | **136.6** | **-2.0%** |
| | prefill tps | 2078.4 | 1864.0 | -10.3%(单发 TTFT 抖动) |
| | 接受长度 Δtok/Δverify | 4.65(512/110) | 4.45(512/115) | 略降(宽 12 截断) |
| 单流 100K(115,155 tok) | decode tps | **105.7** | **109.2** | **+3.3%** |
| | prefill tps | 1565.5 | 1561.8 | -0.2% |
| | 接受长度 | 4.79(512/107) | 4.88(512/105) | 持平 |
| 4 并发 56K(64,498 tok×4) | decode tps(单流均值) | **53.4** | **32.6** | -39.0%(噪声主导,见注) |
| | 各流 decode | 12.8/93.8/94.4/12.7 | 93.1/12.7/12.3/12.3 | 排程形态不同 |
| | prefill tps | 897.6 | 896.9 | -0.1% |
| | 接受长度 | 4.81(2048/426) | 5.09(2048/463) | 持平 |

注:4 并发档 decode 均值被 chunked-prefill 交错排程主导(干净解码 ~93 tps vs 被长 prefill 压着 ~12.5 tps 双峰,取决于几条流赶在上一个 prefill 结束前解码完),前轮 2 条流干净、后轮 1 条;且该档 decode batch=3,选中的策略就是宽 16=与无表完全相同,差异纯属排程方差,不构成信号。历史锚点(dspine-final)8K 140.9 / 100K 106.8 与本次前测偏差 ≤1%,前两档对比可信。

## 4. 判定与终态

- 判定规则(单子):任一档 decode 提升 ≥5% → 保留表固化;全部 <5% 或负 → 回退。
- 实测:8K **-2.0%**、100K **+3.3%**、4×56K 噪声负值,**无一档 ≥+5% → 回退**。
- 终态:launch 脚本已恢复原样(去掉 `--speculative-dspark-sps-table-path`,与调优前逐字一致),服务已按原配置重启,health 200(终验见下),全程及终态均为 DSpine,未回切。
- 表文件保留在 [`tuning/`](../tuning/)(`dspine-sps-table.json` + `runs/` 原始数据)备查:若未来换更大显存卡(mamba 槽↑并发>3)或语料明显变化(如温度采样为主),可按同流程重调再试。

## 5. 证据清单(全部在本仓库)

- 前后 bench 原始行:[`data/bench-tool-results.jsonl`](../data/bench-tool-results.jsonl)(tag sps-前-*/sps-后-*),独立副本 [`tuning/bench-sps-before-after.jsonl`](../tuning/bench-sps-before-after.jsonl)
- metrics 前后快照:[`tuning/sps-metrics/`](../tuning/sps-metrics/)(6 组 × before/after 快照,sglang Prometheus 口径)
- 调优表与测量:[`tuning/dspine-sps-table.json`](../tuning/dspine-sps-table.json) / [`tuning/runs/measurements.json`](../tuning/runs/measurements.json)(逐宽度 × batch × 轮次原始计时与接受长度)
- 服务日志:[`tuning/dspine-sps-table.log`](../tuning/dspine-sps-table.log)(带表实例,含 `DSPine SPS policy: {...}` 加载行)、[`tuning/dspine-final-noSps.log`](../tuning/dspine-final-noSps.log)(终态无表实例)
- 调优器日志:[`tuning/sps-tune.log`](../tuning/sps-tune.log);语料构建:[`tuning/build_sps_prefixes.py`](../tuning/build_sps_prefixes.py);逐候选测量服原始数据:[`tuning/runs/`](../tuning/runs/)
- 终态健康:health **200**,/v1/models = qwen3.8-27b;终态日志 `speculative_dspark_sps_table_path: None`,/proc/cmdline 无 `--speculative-dspark-sps-table-path`(无表时插件打印默认策略行 `uniform_widths=[16×7]`,即整块验证,属正常)
