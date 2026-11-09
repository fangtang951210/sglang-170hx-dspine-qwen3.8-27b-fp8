# DSpine SPS 调优产物(负结果,已回退)

本目录是 2026-11-09/10 在 CMP 170HX 上按官方流程跑完 DSpine SPS 表调优(约 3 小时全程)的**全部原始产物**。结论:三档真实负载 decode 提升全部 <5%,判定负结果,已回退无表配置。负结果也如实收录——同硬件同学可免于重试。完整过程与判定见 [`docs/dspine-sps-tuning.md`](../docs/dspine-sps-tuning.md)。

## 怎么跑的

- 调优器:上游 DSpine 仓 `scripts/tune_sps.py`(**未做任何本地改动**),官方文档 `docs/sps_tuning.md`
- 关键命令形如:`export PATH="$HOME/sglang-llm-venv/bin:$PATH"`(调优器起的服 JIT 依赖 venv bin 里的 ninja,缺了直接崩)之后
  `tune_sps.py --requests sps-prefixes.jsonl --output dspine-sps-table.json --batch-sizes 1 2 3 -- <原样 serving 命令>`
- **为什么 batch 只扫 1–3**:tune_sps.py 强制校验 `effective_max_running_requests_per_dp ≥ 最大 batch`;本机 mamba 16 槽/请求 5 槽 → 实际并发上限 3,官方默认 `--batch-sizes 1–8` 直接报错。宽度 4/8/12/16、3 轮计时、响应 512 均为官方默认。
- 逐候选起服的完整启动参数见 `runs/width*/command.json`(在原 serving 命令上由调优器注入 host/port/表参数)。

## 校准负载说明

- 语料本体:`sps-prefixes.jsonl`(8 行,每行 `{"input_ids": [...], "sampling_params": {...}}`,chat 模板已烘进 input_ids)
- 8 条**全贪心**(temperature=0),共 310,872 tokens;构成 = 测速同源英文技术文章 + DSpine 仓真实 Python 源码,token 级定长(8,984 → 114,984),任务覆盖英文/中文总结、代码讲解、写单测、找 bug、续写,对齐三档真实负载(8K / 56K×4 / 100K)
- 逐条构成见 `sps-prefixes.jsonl.stats.json`;构建脚本 `build_sps_prefixes.py`(语料路径为本机占位,换机重跑需改成自己的语料路径)
- 官方要求贪心与温度采样**分开两次调**;本轮生产负载全贪心,故只调了贪心侧——若你的负载以温度采样为主,须另调,不能直接用这张表

## 文件清单

| 文件/目录 | 内容 |
|---|---|
| `dspine-sps-table.json` | 最终生成的 SPS 表(format `dspine-sps-v1`:选中策略 + 12 组测量 × 3 轮 + 复验 + 硬件指纹 + 命令 + 语料摘要) |
| `sps-prefixes.jsonl` | 校准语料本体(8 条,310,872 tokens) |
| `sps-prefixes.jsonl.stats.json` | 语料逐条构成(类型/长度/任务) |
| `build_sps_prefixes.py` | 语料构建脚本 |
| `sps-tune.log` | tune_sps.py 调优器全程日志(逐宽度逐 batch 逐轮计时与接受长度,末尾选定策略+复验) |
| `bench-sps-before-after.jsonl` | 前后对比 bench 原始行(6 行,tag `sps-前-*` / `sps-后-*`,已并入 `data/bench-tool-results.jsonl`) |
| `sps-metrics/` | 前后各档 bench 前后的 sglang Prometheus metrics 快照(6 组 × before/after) |
| `dspine-sps-table.log` | 带表实例服务日志(含 `DSPine SPS policy: {...}` 加载行) |
| `dspine-final-noSps.log` | 终态无表实例服务日志(回退后健康运行) |
| `runs/` | tune_sps.py 中间产物:`measurements.json`/`selected.json`(汇总)、`width{4,8,12,16}/`(逐候选 `command.json` 启动参数、`requests.jsonl` 逐请求原始计时与生成文本、`server.log` 测量服日志、`sps.json` 该候选表)、`selected/`(选定组合复验服) |

## 如实记录的两个坑(复跑者注意)

1. **测量口径被 prefill 主导**:310K token 语料池超出本机 KV 缓存容量,LRU 淘汰使跨队列 prefix 不复用,每轮 ~200s 墙钟大头是 prefill,宽度间 decode 差异被稀释——与官方“warm 后缓存命中”的预期场景不同。语料池更大时此坑更明显。
2. **接受长度天花板**:本负载接受长度本就 ~3.9,16 宽整块验证浪费不大,收窄到 12/8 自然收益小。官方甜区(小宽度高收益)在这台机的真实负载上不存在;4×56K 档 decode batch=3 时选中策略=宽 16,与无表完全等价,前后差异纯属排程方差。

换更大显存卡(mamba 槽↑、并发>3)或负载形态改变(温度采样为主、语料明显变化)后,可按同流程重调再试。
