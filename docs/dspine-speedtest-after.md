# 170hx DSpine@sglang0.5.22 对比测速报告(对照 DFlash2 基线)

- 工单:170hx DSpine@sglang0.5.22 服务对比测速(端口 30020,与 DFlash2 基线同口径)
- 测试日期:2026-11-09 15:40:30 – 15:53:07 CST(9 轮:8 档主矩阵 + 100K 复测)
- 铁律遵守:全程只测试,未改任何配置;**测完服务保持运行**(终态 health 200,PID 42258 存活,显存 59199 MiB)
- 工具与口径:与基线完全一致——`~/deploy/bench_tool.py`(venv `~/sglang-llm-venv`,sglang **0.5.22**),随机词表+固定结尾、唯一前缀防 prefix cache、temperature 0、真实 usage 计速;结果追加 `~/deploy/bench-tool-results.jsonl`(tag 前缀 `dspine0522-`,共 9 行)
- 基线对照:/Users/sugar/Documents/tmp/dspine-speedtest-baseline.md(2026-11-09 14:04–14:21,sglang 0.5.20 + DFLASH 8 草稿 + fp8 KV)

## 一、核心结论(先说数)

1. **DSpine decode 全面崩盘,所有档位只有基线的 1/3 左右**:单流 43.8→24.3 tps(基线 82.2→78.1),4 并发总 decode 48.8–78.7(基线 145.7–190.4)。8 档 decode 降幅 -46.7% ~ -69.8%。
2. **根因坐实:接受率塌到 1.0**。decode 日志逐批 accept len 1.00–1.02、accept rate 0.00;测后指标 spec_accept_length **1.0125** / rate **0.0008**——每步只收 bonus token,16 个草稿 token 全部白算,草稿前向+验证 16 token 成为纯开销。同口径下 DFlash2 为 2.75 / 0.25。**DSpine 草稿(qwen3.8-27b-dspine-b16)在这套随机词表提示词分布上基本不被目标模型接受。**
3. **prefill 反而全面变快**:+4.9% ~ +20.7%,100K 档 1331→1607 tps,TTFT 75.7s→62.7s(-17.2%)。来源未做归因(可能是 0.5.22 优化或 KV 路径差异),但方向一致、幅度随长度增大。
4. 100K 复测两轮 24.3 / 24.4,极差 0.1 tps(±0.4%)——不是方差,是稳态。
5. 两个预告差异均如实发生且被数据坐实(详见第六节):KV 池没超(峰值 0.65),4 并发靠排队消化;mamba 上限 3 有启动日志原话,实测 running 峰值 2、queue 峰值 3,无一请求被拒。

## 二、环境与启动核查(只读核对,未动)

- 主机 170hx(CMP 170HX 64G 解锁,Ubuntu 24.04),sglang **0.5.22**(pip 确认)
- 运行参数(get_server_info + 启动日志 `/tmp/dspine-launch-8.log`):DSPINE 投机解码、草稿 `/mnt/storage/models/llm/qwen3.8-27b-dspine-b16`(DSpineDraftModel 3.86 GB)、num_draft_tokens **16**、steps 1、topk 1、draft KV bfloat16、chunked_prefill 2048、fa3、schedule fcfs、**max_running_requests=8(被 mamba 钳到 3,见下)**、mem_fraction 0.8
- 资源池(启动日志 15:31):权重 28.97 GB;**目标 KV bfloat16,#tokens 174870**(K 5.34+V 5.34 GB);草稿 KV bf16 同 token 数(1.67+1.67 GB);mamba 池 16 槽(conv 0.05GB+ssm 2.39GB)
- **mamba 钳制原话(启动日志 15:31:38)**:`max_running_requests is capped to 3 by the mamba state cache (max_mamba_cache_size=16, 5 state slots per request)`
- 注意:本服务**不在 systemd unit 里**(sglang-llm unit 显示 failed 是旧单元残留),进程在 `session-278.scope`,日志落 `/tmp/dspine-launch-8.log`(15:29 启动,15:32 就绪)。journalctl 检查改用该日志文件,等效。

## 三、单流 5 档实测(output 512,temperature 0)

| 档位 | prompt tokens | prefill tps | TTFT | decode tps | 备注 |
|---|---|---|---|---|---|
| 8K | 8113 | 1781.1 | 4.56s | **43.8** | |
| 16K | 16179 | 2072.5 | 7.81s | **29.6** | |
| 32K | 32295 | 1973.8 | 16.36s | **28.5** | |
| 56K | 56463 | 1827.7 | 30.89s | **26.7** | |
| 100K | 100799 | 1607.2 | 62.72s | **24.3** | 主档 |
| 100K 复测 | 100799 | 1604.9 | 62.81s | **24.4** | 方差 ±0.1 tps |

decode 随长度单调走低(43.8→24.4),与基线"全长度区间平稳 75-83"形成鲜明对比;completion tokens 353–512(temperature 0 提前 EOS,与基线同现象,计速按真实 usage,口径一致)。

## 四、4 并发 3 档实测(decode_tps_total 为 4 流合计)

| 档位 | prompt tokens/请求 | prefill tps(均值) | TTFT max | decode 单流分布 | **decode 总 tps** |
|---|---|---|---|---|---|
| 4c-16K | ~16160 | 808.1 | 53.3s | 16.6/18.5/27.5/16.1 | **78.7** |
| 4c-32K | ~32276 | 901.6 | 87.5s | 14.0/24.1/14.3/24.1 | **76.5** |
| 4c-56K | ~56449 | 885.7 | 145.6s | 9.3/9.1/8.2/22.2 | **48.8** |

4c-56K 四个请求 TTFT 分别为 32.3 / 61.8 / 114.2 / 145.6s——第 3、4 个请求等前序释放槽位才进,排队线性劣化比基线更早出现(基线 4c-56K 第 3 个 112.4s 起就进预填了)。

## 五、与 DFlash2 基线逐档对比(DSpine 相对 DFlash2,负数=退步)

单流:

| 档位 | prefill tps | Δ | TTFT | Δ | decode tps | Δ | 显存峰值 MiB(ΔFlash2→DSpine) |
|---|---|---|---|---|---|---|---|
| 8K | 2021.3→1781.1 | **-11.9%** | 4.01→4.56s | **+13.5%** | 82.2→43.8 | **-46.7%** | 57901→59195(+1294) |
| 16K | 1974.8→2072.5 | **+5.0%** | 8.19→7.81s | **-4.7%** | 82.4→29.6 | **-64.1%** | 57901→59195(+1294) |
| 32K | 1816.7→1973.8 | **+8.6%** | 17.78→16.36s | **-8.0%** | 83.3→28.5 | **-65.8%** | 57901→59197(+1296) |
| 56K | 1611.8→1827.7 | **+13.4%** | 35.03→30.89s | **-11.8%** | 74.7→26.7 | **-64.3%** | 57903→59199(+1296) |
| 100K | 1331.4→1607.2 | **+20.7%** | 75.71→62.72s | **-17.2%** | 78.1→24.3 | **-68.9%** | 57905→59199(+1294) |
| 100K(对基线 r2 80.9) | — | — | — | — | 80.9→24.4 | **-69.8%** | — |

4 并发:

| 档位 | prefill tps(均值) | Δ | TTFT max | Δ | decode 总 tps | Δ | 显存峰值 Δ |
|---|---|---|---|---|---|---|---|
| 4c-16K | 908.3→808.1 | **-11.0%** | 40.0→53.3s | **+33.3%** | 166.8→78.7 | **-52.8%** | 57905→59199(+1294) |
| 4c-32K | 887.6→901.6 | **+1.6%** | 78.5→87.5s | **+11.4%** | 190.4→76.5 | **-59.8%** | 57907→59199(+1292) |
| 4c-56K | 811.3→885.7 | **+9.2%** | 147.4→145.6s | **-1.2%** | 145.7→48.8 | **-66.5%** | 57907→59199(+1292) |

一句话:**prefill 全线变快、decode 全线崩到 1/3,显存常驻高 ~1.3 GB**。

## 六、两个预告差异的实际表现

**a) KV 预算 174870 token vs 4 并发 56K 合计 ~225K 输入——未超池,靠排队消化,无拒绝**
- 全程 token_usage(占 KV 池比例)峰值 **0.65**(=113.7K / 174870,出现在 4c-56K,2 条 56K 长驻 + 1 条预填交错时)
- 没有出现限流/拒绝/retract:12 个并发请求全部 200 完成,日志 grep warn/retract/evict/oom/fragment/abort **零命中**(仅 1 条 triton `make_block_ptr` 弃用 UserWarning,良性)
- 原因:并发被 mamba 钳在 ≤3 驻留,4×56K 根本不会同时进池——**KV 超池风险被 mamba 排队间接挡住了**,代价转嫁为 TTFT 排队

**b) mamba 16 槽 × 5 槽/请求 = 并发上限 3——启动日志原话坐实,4 并发必有排队,实测如下**
- 启动日志:`max_running_requests is capped to 3 by the mamba state cache (max_mamba_cache_size=16, 5 state slots per request)`
- 实测(10s 间隔抓 `num_running_reqs`/`num_queue_reqs`):三档 4 并发 **running 峰值均只有 2,queue 峰值 3**;decode 运行时日志 `mamba num: 8, mamba usage: 0.50`(2 请求 × 4 槽实际占用,准入按 5 槽预留)
- 4c-56K 时间线:15:50:26 起 1 预填 + 3 排队 → 2 并行解码 → 第 1 条完成后第 3 条才进预填 → 第 2 条完成后第 4 条才进;最终 TTFT 32.3/61.8/114.2/145.6s,第 4 请求排队 ~2 分钟
- **实际并发行为结论:4 并发被串行化为"≤2 解码并行 + 1 预填交错",第 4 请求纯排队;无失败无降级,但并发扩展性为零(总 decode 反而低于单流×2)**

## 七、接受率(测前/测后)与异常观察

**接受率:**
- 测前(15:38:34,首测前):spec_accept_length **3.5**(服务 15:32 就绪后预热+此前未知流量的累计均值,rate 未抓)
- 测后(15:53:35,9 轮完成):spec_accept_length **1.0125**,spec_accept_rate **0.0008**
- 对照:DFlash2 基线同口径 2.75 / 0.25;decode 批日志全程 accept len 1.00–1.02
- 解读:DSpine 在本测速口径(随机词表+哲学 essay 结尾,temperature 0)下草稿几乎全被拒;测前 3.5 说明它并非在所有提示词分布上都这么差(预热流量是什么内容不可考)——**接受率强烈依赖工作负载,是 DSpine 当前最大的真实代价**

**其他观察:**
- 单流 decode 随长度衰减明显(8K 43.8 → 100K 24.3),基线则是平的 75-83;草稿验证成本(16 token/步)随上下文变长的开销放大
- gen throughput 日志交叉验证:单流 ~27.8 tps、2 并行 ~43.7-44.8 tps,与 bench 数字自洽
- 显存:空载 58951(基线 57593,**常驻高 1358 MiB**,bf16 KV + 3.86GB 草稿所致);全程峰值 59199,平坦无增长;测后空闲仍 59199(池预分配+radix 保留,非泄漏迹象);温度峰值 73℃、功耗峰值 342.7W,与基线相当,无过热
- prefill 提升未做归因(0.5.22 版本差异或 KV dtype 路径差异均可能),只报数不猜死因

## 八、raw JSONL 原样引用(本次 9 行,源:170hx:~/deploy/bench-tool-results.jsonl)

```json
{"tag": "dspine0522-1s-8k", "words": 8000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 8113, "prefill_tps_avg": 1781.1, "decode_tps_each": [43.8], "decode_tps_avg": 43.8, "decode_tps_total": 43.8, "ttft_ms_max": 4555.1, "detail": [{"seed": 1, "prompt_tokens": 8113, "completion_tokens": 425, "prefill_ms": 4555.1, "decode_ms": 9699.4, "prefill_tps": 1781.1, "decode_tps": 43.8, "ttft_ms": 4555.1}]}
{"tag": "dspine0522-1s-16k", "words": 16000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 16179, "prefill_tps_avg": 2072.5, "decode_tps_each": [29.6], "decode_tps_avg": 29.6, "decode_tps_total": 29.6, "ttft_ms_max": 7806.4, "detail": [{"seed": 1, "prompt_tokens": 16179, "completion_tokens": 353, "prefill_ms": 7806.4, "decode_ms": 11944.3, "prefill_tps": 2072.5, "decode_tps": 29.6, "ttft_ms": 7806.4}]}
{"tag": "dspine0522-1s-32k", "words": 32000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 32295, "prefill_tps_avg": 1973.8, "decode_tps_each": [28.5], "decode_tps_avg": 28.5, "decode_tps_total": 28.5, "ttft_ms_max": 16361.7, "detail": [{"seed": 1, "prompt_tokens": 32295, "completion_tokens": 512, "prefill_ms": 16361.7, "decode_ms": 17983.2, "prefill_tps": 1973.8, "decode_tps": 28.5, "ttft_ms": 16361.7}]}
{"tag": "dspine0522-1s-56k", "words": 56000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 56463, "prefill_tps_avg": 1827.7, "decode_tps_each": [26.7], "decode_tps_avg": 26.7, "decode_tps_total": 26.7, "ttft_ms_max": 30892.3, "detail": [{"seed": 1, "prompt_tokens": 56463, "completion_tokens": 512, "prefill_ms": 30892.3, "decode_ms": 19187.6, "prefill_tps": 1827.7, "decode_tps": 26.7, "ttft_ms": 30892.3}]}
{"tag": "dspine0522-1s-100k", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1607.2, "decode_tps_each": [24.3], "decode_tps_avg": 24.3, "decode_tps_total": 24.3, "ttft_ms_max": 62718.3, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 62718.3, "decode_ms": 21105.3, "prefill_tps": 1607.2, "decode_tps": 24.3, "ttft_ms": 62718.3}]}
{"tag": "dspine0522-1s-100k-r2", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1604.9, "decode_tps_each": [24.4], "decode_tps_avg": 24.4, "decode_tps_total": 24.4, "ttft_ms_max": 62807.7, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 62807.7, "decode_ms": 21016.4, "prefill_tps": 1604.9, "decode_tps": 24.4, "ttft_ms": 62807.7}]}
{"tag": "dspine0522-4c-16k", "words": 16000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 16160, "prefill_tps_avg": 808.1, "decode_tps_each": [16.6, 18.5, 27.5, 16.1], "decode_tps_avg": 19.7, "decode_tps_total": 78.7, "ttft_ms_max": 53287.5, "detail": [{"seed": 1, "prompt_tokens": 16179, "completion_tokens": 355, "prefill_ms": 10104.3, "decode_ms": 21331.7, "prefill_tps": 1601.2, "decode_tps": 16.6, "ttft_ms": 10104.3}, {"seed": 2, "prompt_tokens": 16155, "completion_tokens": 512, "prefill_ms": 17600.9, "decode_ms": 27723.1, "prefill_tps": 917.9, "decode_tps": 18.5, "ttft_ms": 17600.9}, {"seed": 3, "prompt_tokens": 16157, "completion_tokens": 404, "prefill_ms": 53287.5, "decode_ms": 14716.0, "prefill_tps": 303.2, "decode_tps": 27.5, "ttft_ms": 53287.5}, {"seed": 4, "prompt_tokens": 16150, "completion_tokens": 339, "prefill_ms": 39384.5, "decode_ms": 21082.9, "prefill_tps": 410.1, "decode_tps": 16.1, "ttft_ms": 39384.5}]}
{"tag": "dspine0522-4c-32k", "words": 32000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 32276, "prefill_tps_avg": 901.6, "decode_tps_each": [14.0, 24.1, 14.3, 24.1], "decode_tps_avg": 19.1, "decode_tps_total": 76.5, "ttft_ms_max": 87461.3, "detail": [{"seed": 1, "prompt_tokens": 32295, "completion_tokens": 512, "prefill_ms": 17796.0, "decode_ms": 36584.5, "prefill_tps": 1814.7, "decode_tps": 14.0, "ttft_ms": 17796.0}, {"seed": 2, "prompt_tokens": 32261, "completion_tokens": 512, "prefill_ms": 33105.9, "decode_ms": 21258.0, "prefill_tps": 974.5, "decode_tps": 24.1, "ttft_ms": 33105.9}, {"seed": 3, "prompt_tokens": 32271, "completion_tokens": 512, "prefill_ms": 72038.4, "decode_ms": 35688.9, "prefill_tps": 448.0, "decode_tps": 14.3, "ttft_ms": 72038.4}, {"seed": 4, "prompt_tokens": 32279, "completion_tokens": 322, "prefill_ms": 87461.3, "decode_ms": 13339.6, "prefill_tps": 369.1, "decode_tps": 24.1, "ttft_ms": 87461.3}]}
{"tag": "dspine0522-4c-56k", "words": 56000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 56449, "prefill_tps_avg": 885.7, "decode_tps_each": [9.3, 9.1, 8.2, 22.2], "decode_tps_avg": 12.2, "decode_tps_total": 48.8, "ttft_ms_max": 145585.7, "detail": [{"seed": 1, "prompt_tokens": 56463, "completion_tokens": 475, "prefill_ms": 32319.2, "decode_ms": 50818.9, "prefill_tps": 1747.0, "decode_tps": 9.3, "ttft_ms": 32319.2}, {"seed": 2, "prompt_tokens": 56437, "completion_tokens": 481, "prefill_ms": 61758.2, "decode_ms": 52768.3, "prefill_tps": 913.8, "decode_tps": 9.1, "ttft_ms": 61758.2}, {"seed": 3, "prompt_tokens": 56437, "completion_tokens": 409, "prefill_ms": 114223.5, "decode_ms": 49604.8, "prefill_tps": 494.1, "decode_tps": 8.2, "ttft_ms": 114223.5}, {"seed": 4, "prompt_tokens": 56460, "completion_tokens": 407, "prefill_ms": 145585.7, "decode_ms": 18297.5, "prefill_tps": 387.8, "decode_tps": 22.2, "ttft_ms": 145585.7}]}
```

## 九、附:采样与方法备忘

- 各档窗口(CST):8K 15:40:30 / 16K 15:41:03 / 32K 15:41:43 / 56K 15:42:32 / 100K 15:43:38 / 100K-r2 15:45:12 / 4c16K 15:46:57 / 4c32K 15:48:19 / 4c56K 15:50:23,档间 sleep 10s
- nvidia-smi 原始采样(2s 间隔):170hx `/tmp/dspine-after-nvsmi.csv`;并发/队列/token_usage 采样(10s 间隔):`/tmp/dspine-after-reqs.log`;本报告峰值与并发行为按各档时间窗切片(`/tmp/dspine-after-analyze.py`,一次性)
- 接受率抓取:`curl -s localhost:30020/metrics | grep spec_accept`(测前 15:38:34 / 测后 15:53:35)
- 日志告警检查:`/tmp/dspine-launch-8.log`(本服务非 systemd unit,见第二节)grep warn/retract/evict/oom/fragment/abort,零命中

*报告完。服务保持运行未关;未改任何配置;170hx 上新增 /tmp/dspine-after-*.{csv,log,sh,py,pid} 采样残留,可删。*
