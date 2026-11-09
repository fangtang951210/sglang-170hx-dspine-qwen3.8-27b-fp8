# 170hx sglang 推理服务专项测速基线报告

- 工单:170hx sglang(Qwen3.8-27B-FP8 + DFlash2 草稿,端口 30020)专项测速 + "76 vs 100+"异常排查
- 测试日期:2026-11-09 14:04 – 14:21 CST(服务 14:01:44 启动,14:03:45 就绪,总启动约 2 分钟)
- 铁律遵守:全程只测试,未改任何配置;**测完服务保持运行**(终态 active / health 200 / 显存 57907 MiB 占用)
- 工具:`~/deploy/bench_tool.py`(venv `~/sglang-llm-venv`),口径=用户网页测速工具 v2.2 同构:随机词表+固定结尾、唯一前缀防 prefix cache、enable_thinking=false、temperature 0、真实 usage 计速。结果落 `~/deploy/bench-tool-results.jsonl`

## 一、核心结论(先说数)

1. **单流 decode 全长度区间稳定在 75-83 tps**(1K→100K 都这样),100K 三轮 78.1/80.9/78.0,方差 ±2-3 tps,与控制台昨日初测 75.7/76.0 高度复现。
2. **76 不是今天的异常**:服务无劣化、无告警、无配置漂移、显存全程平稳——这就是当前软硬件栈(sglang 0.5.20 + DFLASH 8 草稿 + FP8 Marlin + fp8 KV)的稳态水平。
3. **本机找不到任何"100+"的历史实测记录**(结果文件里只有今天控制台的 5 行)。"单流 100+"记忆无法在数据中核实;最接近 100+ 的是今天 13:35 官方 bench(2K 输入×8 并发)的单流体感值:mean TPOT 10.6ms ≈ 94 tps、mean ITL 9.6ms ≈ 104 tps——并发批处理摊薄权重读取后体感确实能过 100。
4. 4 并发总 decode:16K 166.8 / 32K 190.4 / 56K 145.7 tps(控制台 56K 144.0 复现)。
5. DFlash 接受率:spec_accept_length **0.0(测前)→ 2.75(测后)**,accept rate 0.25。

## 二、测试环境与启动核查(排查步骤 e)

- 主机 170hx,CMP 170HX 64G 解锁,Ubuntu 24.04,sglang **0.5.20**
- 启动链一致性(逐项核对):systemd unit `ExecStart=/home/<user>/deploy/launch_user.sh` → 脚本内容 → 运行时 `/proc/<MainPID>/cmdline` → 今天 13:20 的 `launch_user.sh.bak`/`sglang-llm.service.bak`,**四方逐字节一致,今天没改过任何配置**
- 关键运行参数(来自启动日志 server_args 与 cmdline):DFLASH 投机解码、草稿 `qwen3.8-27b-dflash2`、num_draft_tokens 8、kv fp8_e4m3、mem_fraction_static 0.85、chunked_prefill 2048、flashinfer、mamba cache 12 槽(extra_buffer 策略)、max_running_requests 48
- 资源池(启动日志):权重 28.97 GB;KV 池 450101 tokens(K+V 13.74 GB,fp8);草稿 KV 4.3 GB;mamba 池约 2.67 GB;CUDA 图捕完后可用 7.59 GB
- 服务先前 "failed" 状态原因:控制台 13:51:49 用 SIGINT 停服,sglang 关闭路径以 KeyboardInterrupt/exit 1 收尾,systemd 记为 failed;日志同时有 "Application shutdown complete",**属正常停服的显示问题,非故障**
- **注意(排查证据①)**:启动日志明确打印 `Your GPU does not have native support for FP8 computation ... Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.`——GA100 无原生 FP8 算力,FP8 权重走 Marlin 反量化。

## 三、测速矩阵(8 档主矩阵 + 排查加测,共 13 轮)

单流(output 512,temperature 0):

| 档位 | 实测 prompt tokens | prefill tps | TTFT | decode tps | 备注 |
|---|---|---|---|---|---|
| 1K | 1057 | 1655.1 | 0.64s | **81.5** | 加测(排查用) |
| 2K | 2066 | 1878.4 | 1.10s | **79.3** | 加测(排查用) |
| 8K | 8113 | 2021.3 | 4.01s | **82.2** | 衰减参照 |
| 16K | 16179 | 1974.8 | 8.19s | **82.4** | |
| 32K | 32295 | 1816.7 | 17.78s | **83.3** | |
| 56K | 56463 | 1611.8 | 35.03s | **74.7** | |
| 100K | 100799 | 1331.4 | 75.71s | **78.1** | 主档 |
| 100K 复测① | 100799 | 1330.0 | 75.79s | **80.9** | 排查步骤 d |
| 100K 复测② | 100799 | 1324.0 | 76.13s | **78.0** | 排查步骤 d |
| 8K 末段对照 | 8113 | 2008.0 | 4.04s | **80.9** | 加测:开局 82.2 vs 收尾 80.9 |

4 并发(decode_tps_total 为 4 流合计;单流内各请求受 chunked prefill 交错影响差异大):

| 档位 | prompt tokens/请求 | prefill tps(均值) | TTFT max | decode 单流分布 | **decode 总 tps** |
|---|---|---|---|---|---|
| 4c-16K | ~16160 | 908.3 | 40.0s | 38.0/30.6/31.1/67.1 | **166.8** |
| 4c-32K | ~32276 | 887.6 | 78.5s | 12.3/83.5/21.0/73.6 | **190.4** |
| 4c-56K | ~56449 | 811.3 | 147.4s | 6.7/71.6/8.9/58.5 | **145.7** |

说明:并发档中先进入 decode 的请求会与后续请求的 chunked prefill 交错,单流 decode 数字被拉低(如 6.7),总吞吐才是有效口径;completion_tokens 在 temperature 0 下有时提前 EOS(<512,如 8K 档 440),decode tps 一律按真实 usage token 计,口径诚实。

## 四、显存/功耗采样(nvidia-smi 每 2s,全程 14:04:36-14:18:44)

服务就绪后基线 **57593 MiB**;各档峰值:

| 档位 | 显存峰值 MiB | GPU util 峰值 | 功耗峰值 W | 温度峰值 ℃ |
|---|---|---|---|---|
| 1s-8K | 57901 | 97 | 303.5 | 45 |
| 1s-16K | 57901 | 100 | 307.3 | 50 |
| 1s-32K | 57901 | 100 | 331.6 | 55 |
| 1s-56K | 57903 | 100 | 366.3 | 61 |
| 1s-100K | 57905 | 100 | 373.8 | 69 |
| 4c-16K | 57905 | 100 | 315.3 | 69 |
| 4c-32K | 57907 | 100 | 349.5 | 73 |
| 4c-56K | 57907 | 100 | 368.2 | 73 |
| 1s-100K-r2 | 57907 | 100 | 367.8 | 70 |
| 1s-100K-r3 | 57907 | 100 | 378.4 | 72 |
| 1s-8k-r2 | 57907 | 97 | 307.2 | 63 |

全程会话峰值 57907 MiB,与基线差仅 ~314 MiB 且后期恒定(池预分配顶格)。**无显存增长/碎片化迹象,温度功耗正常(峰值 378W/73℃,无过热压制迹象)。**

## 五、DFlash 接受率

- 测前(服务就绪后、首测前):`sglang:spec_accept_length = 0.0`(刚启动,仅预热请求,gauge 未累计)
- 测后(全部 13 轮完成):`sglang:spec_accept_length = 2.75`,`sglang:spec_accept_rate = 0.25`
- 含义:每次目标模型前向平均收下 2.75 token(含 bonus);8 个草稿 token 收下约 2 个

## 六、"76 vs 100+"异常排查记录(只列证据,不下死结论)

| # | 假设 | 证据 | 判定倾向 |
|---|---|---|---|
| a | 控制台早前测试偶然偏慢 | 早前 5 行(dflash2- 前缀):单流 100K 75.7/76.0、8K 78.1、4c56K 总 144.0;本次复测 78.1/80.9/78.0、82.2、145.7——完全复现,还略快 3-6% | **排除**:控制台数据不是偶然,服务一贯如此 |
| b | 长上下文 decode 随测试进行劣化(碎片/cache 池劣化) | 100K 三轮 78.1→80.9→78.0(不降反波动);8K 开局 82.2 vs 全部测完后的末段 80.9;显存峰值恒定 57907 | **排除**:无劣化 |
| c | 服务端告警(缓存驱逐/retract/mamba 槽不足) | 测试窗口 journal 告警计数 **0**(grep warn/retract/evict/oom/fragment/abort;先前命中的 "Prefill batch" 行是 `#cached-token…full` 的正则误匹配,非告警)。mamba usage 全程 0.33 无压力 | **排除** |
| d | 测量方差大,76 是坏点 | 100K 三轮极差仅 2.9 tps(±2%) | **排除**:76 是稳态不是坏点 |
| e | 启动参数漂移 | systemd unit == launch_user.sh == 运行时 cmdline == 今天 13:20 的 .bak,逐字节一致 | **排除** |
| f | 上下文长度导致(用户记忆来自短上下文) | 加测 1K/2K 单流:81.5/79.3——短上下文也到不了 100+ | **对单流不成立** |
| g | 用户记忆来自并发/不同口径 | 今天 13:35 官方 bench_serving(2K 输入,并发 8,`bench-dflash2-pre.json`):output_throughput 162.8 tps,mean TPOT 10.64ms(≈94 tps/流),mean ITL 9.63ms(≈104 tps/流)——并发下批处理摊薄权重读取,单流体感确实 >100 | **可能**:体感/口径来源 |
| h | 旧版本更快(如 sglang 升 0.5.20 前 DFlash 接受率更高) | 结果文件里无任何 100+ 历史记录可对照;机器上找不到升级前的测速数据(明说:搜不到) | **无法证实也无法排除** |
| i | 硬件层(FP8 无原生算力/带宽瓶颈) | GA100 无原生 FP8→Marlin weight-only(启动日志明示);带宽推算:29GB 权重 ÷ ~1.49TB/s ≈ 51 tps 基线,×接受 2.75 ≈ 理论上限,实测 76-83 落在合理区间(草稿前向+KV 读取+批内损耗吃掉差额) | 与现状自洽 |

**综合**:76-83 是该服务在当前栈上的真实稳态;不存在"今天变慢了"的证据。要定位"100+"从哪来,需要用户指认来源(哪台机器/哪个工具/什么时期/单流还是并发),本机数据已无法进一步回溯。

## 七、其他观察

- journal 常规行:mamba usage 恒 0.33(4/12 槽口径,单请求也显示 0.33,属统计口径,非压力);chunked prefill 每 chunk 2048 token,单流 prefill 实时吞吐 1700-2100 tok/s(8K 档)随长度衰减到 1300+(100K 档,注意力二次项),偶见的 18 万 tok/s 行是 chunk 收尾的统计毛刺
- 4 并发档 TTFT 随排队位次线性劣化(56K 档第 4 个请求 TTFT 147s)——fcfs+chunked prefill 的正常行为,长上下文并发体验主要受此制约
- 服务重启后首次 8K 测试(82.2)与末段(80.9)一致,无"冷启动慢"现象(CUDA 图捕获在启动阶段完成)

## 八、raw JSONL 原样引用(本次 13 行,源:170hx:~/deploy/bench-tool-results.jsonl)

```json
{"tag": "dspine-1s-8k", "words": 8000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 8113, "prefill_tps_avg": 2021.3, "decode_tps_each": [82.2], "decode_tps_avg": 82.2, "decode_tps_total": 82.2, "ttft_ms_max": 4013.7, "detail": [{"seed": 1, "prompt_tokens": 8113, "completion_tokens": 440, "prefill_ms": 4013.7, "decode_ms": 5354.3, "prefill_tps": 2021.3, "decode_tps": 82.2, "ttft_ms": 4013.7}]}
{"tag": "dspine-1s-16k", "words": 16000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 16179, "prefill_tps_avg": 1974.8, "decode_tps_each": [82.4], "decode_tps_avg": 82.4, "decode_tps_total": 82.4, "ttft_ms_max": 8192.6, "detail": [{"seed": 1, "prompt_tokens": 16179, "completion_tokens": 512, "prefill_ms": 8192.6, "decode_ms": 6216.6, "prefill_tps": 1974.8, "decode_tps": 82.4, "ttft_ms": 8192.6}]}
{"tag": "dspine-1s-32k", "words": 32000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 32295, "prefill_tps_avg": 1816.7, "decode_tps_each": [83.3], "decode_tps_avg": 83.3, "decode_tps_total": 83.3, "ttft_ms_max": 17776.6, "detail": [{"seed": 1, "prompt_tokens": 32295, "completion_tokens": 512, "prefill_ms": 17776.6, "decode_ms": 6146.8, "prefill_tps": 1816.7, "decode_tps": 83.3, "ttft_ms": 17776.6}]}
{"tag": "dspine-1s-56k", "words": 56000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 56463, "prefill_tps_avg": 1611.8, "decode_tps_each": [74.7], "decode_tps_avg": 74.7, "decode_tps_total": 74.7, "ttft_ms_max": 35031.6, "detail": [{"seed": 1, "prompt_tokens": 56463, "completion_tokens": 413, "prefill_ms": 35031.6, "decode_ms": 5526.0, "prefill_tps": 1611.8, "decode_tps": 74.7, "ttft_ms": 35031.6}]}
{"tag": "dspine-1s-100k", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1331.4, "decode_tps_each": [78.1], "decode_tps_avg": 78.1, "decode_tps_total": 78.1, "ttft_ms_max": 75711.8, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 75711.8, "decode_ms": 6557.9, "prefill_tps": 1331.4, "decode_tps": 78.1, "ttft_ms": 75711.8}]}
{"tag": "dspine-4c-16k", "words": 16000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 16160, "prefill_tps_avg": 908.3, "decode_tps_each": [38.0, 30.6, 31.1, 67.1], "decode_tps_avg": 41.7, "decode_tps_total": 166.8, "ttft_ms_max": 39967.9, "detail": [{"seed": 1, "prompt_tokens": 16179, "completion_tokens": 512, "prefill_ms": 9326.0, "decode_ms": 13474.0, "prefill_tps": 1734.8, "decode_tps": 38.0, "ttft_ms": 9326.0}, {"seed": 2, "prompt_tokens": 16155, "completion_tokens": 462, "prefill_ms": 16585.3, "decode_ms": 15092.6, "prefill_tps": 974.1, "decode_tps": 30.6, "ttft_ms": 16585.3}, {"seed": 3, "prompt_tokens": 16157, "completion_tokens": 462, "prefill_ms": 31063.9, "decode_ms": 14847.7, "prefill_tps": 520.1, "decode_tps": 31.1, "ttft_ms": 31063.9}, {"seed": 4, "prompt_tokens": 16150, "completion_tokens": 403, "prefill_ms": 39967.9, "decode_ms": 6006.6, "prefill_tps": 404.1, "decode_tps": 67.1, "ttft_ms": 39967.9}]}
{"tag": "dspine-4c-32k", "words": 32000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 32276, "prefill_tps_avg": 887.6, "decode_tps_each": [12.3, 83.5, 21.0, 73.6], "decode_tps_avg": 47.6, "decode_tps_total": 190.4, "ttft_ms_max": 78492.5, "detail": [{"seed": 1, "prompt_tokens": 32295, "completion_tokens": 512, "prefill_ms": 19012.5, "decode_ms": 41596.2, "prefill_tps": 1698.6, "decode_tps": 12.3, "ttft_ms": 19012.5}, {"seed": 2, "prompt_tokens": 32261, "completion_tokens": 512, "prefill_ms": 35804.8, "decode_ms": 6133.5, "prefill_tps": 901.0, "decode_tps": 83.5, "ttft_ms": 35804.8}, {"seed": 3, "prompt_tokens": 32271, "completion_tokens": 512, "prefill_ms": 59821.5, "decode_ms": 24389.9, "prefill_tps": 539.5, "decode_tps": 21.0, "ttft_ms": 59821.5}, {"seed": 4, "prompt_tokens": 32279, "completion_tokens": 324, "prefill_ms": 78492.5, "decode_ms": 4399.6, "prefill_tps": 411.2, "decode_tps": 73.6, "ttft_ms": 78492.5}]}
{"tag": "dspine-4c-56k", "words": 56000, "concurrency": 4, "output_len": 512, "prompt_tokens_avg": 56449, "prefill_tps_avg": 811.3, "decode_tps_each": [6.7, 71.6, 8.9, 58.5], "decode_tps_avg": 36.4, "decode_tps_total": 145.7, "ttft_ms_max": 147423.2, "detail": [{"seed": 1, "prompt_tokens": 56463, "completion_tokens": 512, "prefill_ms": 36285.3, "decode_ms": 76251.0, "prefill_tps": 1556.1, "decode_tps": 6.7, "ttft_ms": 36285.3}, {"seed": 2, "prompt_tokens": 56437, "completion_tokens": 512, "prefill_ms": 70192.0, "decode_ms": 7150.2, "prefill_tps": 804.0, "decode_tps": 71.6, "ttft_ms": 70192.0}, {"seed": 3, "prompt_tokens": 56437, "completion_tokens": 365, "prefill_ms": 112392.8, "decode_ms": 41192.9, "prefill_tps": 502.1, "decode_tps": 8.9, "ttft_ms": 112392.8}, {"seed": 4, "prompt_tokens": 56460, "completion_tokens": 372, "prefill_ms": 147423.2, "decode_ms": 6357.5, "prefill_tps": 383.0, "decode_tps": 58.5, "ttft_ms": 147423.2}]}
{"tag": "dspine-1s-100k-r2", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1330.0, "decode_tps_each": [80.9], "decode_tps_avg": 80.9, "decode_tps_total": 80.9, "ttft_ms_max": 75786.1, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 75786.1, "decode_ms": 6327.0, "prefill_tps": 1330.0, "decode_tps": 80.9, "ttft_ms": 75786.1}]}
{"tag": "dspine-1s-100k-r3", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1324.0, "decode_tps_each": [78.0], "decode_tps_avg": 78.0, "decode_tps_total": 78.0, "ttft_ms_max": 76134.4, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 76134.4, "decode_ms": 6565.4, "prefill_tps": 1324.0, "decode_tps": 78.0, "ttft_ms": 76134.4}]}
{"tag": "dspine-1s-8k-r2", "words": 8000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 8113, "prefill_tps_avg": 2008.0, "decode_tps_each": [80.9], "decode_tps_avg": 80.9, "decode_tps_total": 80.9, "ttft_ms_max": 4040.3, "detail": [{"seed": 1, "prompt_tokens": 8113, "completion_tokens": 402, "prefill_ms": 4040.3, "decode_ms": 4968.2, "prefill_tps": 2008.0, "decode_tps": 80.9, "ttft_ms": 4040.3}]}
{"tag": "dspine-1s-2k", "words": 2000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 2066, "prefill_tps_avg": 1878.4, "decode_tps_each": [79.3], "decode_tps_avg": 79.3, "decode_tps_total": 79.3, "ttft_ms_max": 1099.9, "detail": [{"seed": 1, "prompt_tokens": 2066, "completion_tokens": 415, "prefill_ms": 1099.9, "decode_ms": 5232.0, "prefill_tps": 1878.4, "decode_tps": 79.3, "ttft_ms": 1099.9}]}
{"tag": "dspine-1s-1k", "words": 1000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 1057, "prefill_tps_avg": 1655.1, "decode_tps_each": [81.5], "decode_tps_avg": 81.5, "decode_tps_total": 81.5, "ttft_ms_max": 638.6, "detail": [{"seed": 1, "prompt_tokens": 1057, "completion_tokens": 407, "prefill_ms": 638.6, "decode_ms": 4996.8, "prefill_tps": 1655.1, "decode_tps": 81.5, "ttft_ms": 638.6}]}
```

对照:控制台早前 5 行(dflash2- 前缀,2026-11-09 13:41-13:51 会话)关键值——单流 100K:75.7 / 76.0;单流 8K:78.1;4c 56K 总:144.0;另有一行 `dflash2-0.5.20-单流100K` 是口径坏点(prefill 0/decode 6.3,首字判定失败把全程算进 decode),不计。

## 九、附:采样与方法备忘

- 各档窗口(CST):8K 14:04:49 / 16K 14:05:17 / 32K 14:05:50 / 56K 14:06:31 / 100K 14:07:30 / 4c16K 14:09:11 / 4c32K 14:10:16 / 4c56K 14:11:57 / 100K-r2 14:14:51 / 100K-r3 14:16:30 / 8K-r2 14:18:10,档间 sleep 10s
- nvidia-smi 原始采样:170hx `/tmp/dspine-nvsmi.csv`(425 行,2s 间隔);本报告峰值按各档时间窗取 max
- 接受率抓取命令:`curl -s localhost:30020/metrics | grep spec_accept_length`
- 日志告警检查:`journalctl --user -u sglang-llm --since "14:04:30"` 过滤 warn/retract/evict/oom/fragment/abort,计数 0

*报告完。服务保持运行未关;未改任何配置;170hx 上仅新增 /tmp/dspine-nvsmi.csv 与 .pid(采样残留,可删)。*
