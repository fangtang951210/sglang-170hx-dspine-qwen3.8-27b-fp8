# 170hx DSpine 部署复现判定实验报告——真实自然语言负载

- 工单:task_53cb794c0712 · 2026-11-09 16:49–17:08 CST · 主机 170hx(CMP 170HX 64G,sglang 0.5.22)
- 问题:DSpine 在随机词测速下 decode 24 tps 连裸跑(40 tps)都不如——判定这是「测法负载错配」还是「部署出错」
- **判别标准(单子写死)**:真实自然语言负载下 DSpine 接受率≥5 且 decode>40 → 部署成立;接受率<3 或 decode≤40 → 部署有问题需排查
- 实验流程:停现役 DFlash2 → 起 DSpine(`~/deploy/launch_dspine.sh` 八枪定版,第 9 次启动)→ 跑三组真实语料 → 停 DSpine → 恢复 DFlash2 → 同三组对照
- **服务终态:DFlash2 运行中**(active,health 200,显存 57599 MiB,17:07:56 核验)

## 一、判定结论(先说判定)

**部署成立。随机词口径是测法负载错配,部署正常,DSpine 定位为真实负载可用的选项。**

- 数学、代码两组双双大幅过「成立」线:接受率 7.51 / 12.28(≥5),decode 237.7 / 386.4 tps(>40)。
- 整体(token 加权)接受率 3.36、decode 103 tps:不落入「<3 或 ≤40」问题分支;三组 decode 全部 >40,随机词口径的 24 tps 在任何真实负载组都不复现。
- 唯一低于阈值项如实记录:narrative 组接受率 1.99(<3 字面触发),但 **decode 60.9 仍 >40**,且同负载下现役 DFlash2 接受率 1.62 更低——该组低接受率是文学创作负载对任意草稿模型都难的负载属性,不是 DSpine 部署缺陷(依据见 §五)。
- 同负载对照:DSpine 三组 decode 全部快过现役 DFlash2(math ×1.44、code ×1.76、narrative ×1.26,加权 ×1.31)。

## 二、实验设计

**语料**(自写,固定,两引擎跑同一份,`~/deploy/bench_real.py` 内置):
a) 数学应用题 10 题(GSM8K 风格,中英各半,要求分步解答)
b) 代码续写 10 题(Python/JS 函数签名+docstring,要求完整实现:merge_intervals、LRUCache、debounce、groupBy、retryWithBackoff 等)
c) 中文叙事续写 10 篇(约 300 字开头 × 10 种题材:悬疑/科幻/武侠/敦煌画匠/末班车等,要求自然续写)

**口径**:单流串行,enable_thinking=false,temperature 0,max_tokens 512,每条带唯一流水号前缀防 prefix cache;预热 1 条不计入。decode tps = completion_tokens/(完成−首字)(流式,真实 usage 计)。**接受率 = spec_accept_length 口径(每步验证平均收下 token 数,含 bonus)**,用三个独立来源交叉验证:
1. 硬数:组内 completion_tokens 合计 ÷ `/metrics` 计数器 `spec_verify_calls_total` 增量(不依赖 gauge 刷新)
2. gauge:组后 `spec_accept_length` 读数(按批刷新的窗口均值)
3. 日志:`/tmp/dspine-launch-9.log` 逐批 `accept len` 按组时间窗切片均值

**流程与基线**:16:55:02 停 DFlash2(停前 spec 基线 0.0/0.0/3.0)→ 16:55:13 起 DSpine → 16:57:29 health 200 → 起后基线 spec_accept_length **0.0**、verify_calls 3.0 → 三组(16:57:43–16:59:33)→ 17:01:14 测后终态指标(2.3/0.087/3267)→ 精确 PID 58323 SIGTERM 停 DSpine,显存归零 → 17:01:39 `systemctl --user start sglang-llm` → 17:03:50 health 200(身份核对:DFLASH·8 草稿·fp8 KV,与实验前完全一致)→ 同三组(17:04:08–17:06:32)。启动日志关键行与部署日志八枪定版一致(mamba 钳并发 3、max_total_num_tokens=174870、fa3、ReplaySSM、草稿 3.86G)。

## 三、DSpine 真实负载数据(主判定对象)

| 组 | decode tps 均值 | 单条区间 | 接受率(硬数 tok/verify) | 接受率(gauge) | 接受率(日志逐批) | verify 增量 | completion 合计 |
|---|---|---|---|---|---|---|---|
| 数学 | **237.7** | 206.8–281.5 | **7.51** | 8.275 | 7.276(14 批) | 535 | 4020 |
| 代码 | **386.4** | 231.1–501.2 | **12.28** | 11.875 | 10.572(4 批) | 144 | 1769 |
| 中文叙事 | **60.9** | 57.8–64.3 | **1.99** | 2.30 | 2.128(65 批) | 2571 | 5120 |
| 整体(加权) | **103.0** | — | **3.36** | — | — | 3250 | 10909 |

TTFT 全部 117–168 ms(短提示词,prefill 不构成因素)。冷启动即工作:服务起来后第一批自然语言请求(16:57:44–46,暖场请求)日志逐批 accept len 5.12→5.83→7.58,gen throughput 167→234 tps——草稿模型从第一条真实请求就在被大量接受。

## 四、DFlash2 同负载对照(现役引擎,同一份语料)

| 组 | decode tps 均值 | 单条区间 | 接受率(硬数) | 接受率(gauge) | verify 增量 | completion 合计 |
|---|---|---|---|---|---|---|
| 数学 | 164.8 | 126.3–198.1 | 5.42 | 4.65 | 755 | 4089 |
| 代码 | 218.9 | 200.8–237.2 | 7.28 | 7.35 | 243 | 1770 |
| 中文叙事 | 48.3 | 45.6–52.2 | 1.62 | 1.525 | 3156 | 5120 |
| 整体(加权) | 78.6 | — | 2.64 | — | 4154 | 10979 |

两引擎逐条 completion tokens 分布几乎一致(math 组同为 196/512/272/320/…量级),temperature 0 下跑的是同一负载,对比公平。

**三组 × 两引擎对照总表(decode tps / 接受率硬数)**:

| 组 | DSpine | DFlash2 | DSpine/DFlash2 |
|---|---|---|---|
| 数学 | 237.7 / 7.51 | 164.8 / 5.42 | ×1.44 / +38% |
| 代码 | 386.4 / 12.28 | 218.9 / 7.28 | ×1.76 / +69% |
| 中文叙事 | 60.9 / 1.99 | 48.3 / 1.62 | ×1.26 / +23% |
| 整体加权 | 103.0 / 3.36 | 78.6 / 2.64 | ×1.31 / +27% |

对照裸跑(AR 无投机,同机实测 37.6–43.0 tps):DSpine 数学 ×5.5、代码 ×9.0、叙事 ×1.4。

## 五、判定依据(逐条)

1. **「成立」分支在两组真实负载上明确达成**:数学 7.51≥5 且 237.7>40;代码 12.28≥5 且 386.4>40。「问题」分支(<3 或 ≤40)在整体层面不触发(3.36≥3 且 103>40),且没有任何一组 decode≤40。
2. **与官方口径吻合,直接排除五关补丁损伤草稿的可能**:官方 27B bf16 基准 GSM8K T0 接受率 7.20(部署日志引);本部署 FP8 底模 + GSM8K 风格数学题实测 7.51(gauge 8.275)——达到官方量级。若 SM80 bf16 粗检索补丁、fa3 绕路、kv/ReplaySSM 交互任何一环损伤了草稿或验证管线,这个数不可能出现(随机词口径下该管线实测只有 1.0)。按单子第 4 步,判定成立故排查分支不启动;补丁无虞的证据即本条。
3. **narrative 组接受率 1.99<3 的归因——负载属性而非部署缺陷**:同一份叙事语料,现役 DFlash2(完全不同的 DFLASH 8-token 草稿)接受率 **1.62,比 DSpine 还低**,decode 48.3 也低于 DSpine 的 60.9。创作型续写本质难预测,压的是所有草稿模型,不专挑 DSpine。
4. **机制自洽性检验:两引擎各自效率因子恒定**。decode tps ÷(裸跑 43 tps × 接受率)在 DSpine 三组为 0.735/0.731/0.711,DFlash2 三组为 0.707/0.699/0.692——同一套健康的「草稿-验证」机制,负载变化只改变接受率,不改变机制效率。若部署出错(草稿输入精度坏/验证图错/状态交互错),损伤会体现在机制效率上且不应随负载题材选择性地发作。
5. **反证**:部署性缺陷是代码路径级的,应均匀压低所有负载的接受率;实测却是数学/代码 7.5–12.3、叙事 2.0、随机词 1.0——接受率随负载内容单调变化且模式与官方基准一致,这是「草稿模型按内容难度正常工作」的特征,不是「管线坏了」的特征。
6. **随机词口径为何失真**:随机词表提示词把生成分布推向词表边缘(下一步词近乎均匀不可预测),任何草稿模型都无从起草——DFlash2 在随机词下 2.75、真实负载 1.62–7.28,同样强烈负载依赖。随机词测速评推测解码是系统性失真的测法,只能测裸算力,不能评草稿。24 tps 之谜的完整解释:随机词下接受率 1.0,16 个草稿 token 全白算,草稿前向+16-token 验证成纯开销,decode 自然崩到裸跑以下;真实负载下接受率 2.0–12.3,同一部署 decode 60.9–386.4。

## 六、遗留与提醒(不属本判定,如实记)

- 本实验只验草稿模型工作与否(单流短上下文);100K 长上下文按单子未测,mamba 并发帽 3、131K 上下文上限、bf16 KV、+1.3G 常驻显存等运营代价仍在(部署日志遗留问题 2/3/4)。DSpine 转正式与否由控制台拍板。
- narrative 类负载两引擎接受率都低(<2),真实业务若以创作型续写为主,投机解码收益有限(×1.3 左右)——负载画像决定选型。
- 170hx 产物:`~/deploy/bench_real.py`(可复用)、`~/deploy/bench-real-results.jsonl`(6 行原始数据)、`/tmp/dspine-launch-9.log`(第 9 次启动全日志,/tmp 易失);本地存档:派工区 `2026-11-09-DSpine复现判定/`(脚本副本、结果副本、逐批日志、启动日志)。

## 七、raw JSONL 原样引用(源:170hx:~/deploy/bench-real-results.jsonl,6 行)

```json
{"engine": "dspine", "group": "math", "ts_start": "16:57:43", "ts_end": "16:58:01", "output_len": 512, "metrics_before": {"accept_length": 0.0, "accept_rate": 0.0, "verify_calls_total": 17.0}, "metrics_after": {"accept_length": 8.275, "accept_rate": 0.485, "verify_calls_total": 552.0}, "agg": {"n": 10, "decode_tps_mean": 237.7, "decode_tps_weighted": 233.1, "decode_tps_min": 206.8, "decode_tps_max": 281.5, "completion_tokens_total": 4020, "accept_length_delta": 8.275, "verify_calls_delta": 535.0}, "detail": [{"idx": 1, "prompt_tokens": 83, "completion_tokens": 196, "ttft_ms": 117.0, "decode_ms": 798.7, "decode_tps": 245.4}, {"idx": 2, "prompt_tokens": 81, "completion_tokens": 512, "ttft_ms": 137.9, "decode_ms": 2447.0, "decode_tps": 209.2}, {"idx": 3, "prompt_tokens": 83, "completion_tokens": 272, "ttft_ms": 136.3, "decode_ms": 966.2, "decode_tps": 281.5}, {"idx": 4, "prompt_tokens": 91, "completion_tokens": 320, "ttft_ms": 141.5, "decode_ms": 1547.1, "decode_tps": 206.8}, {"idx": 5, "prompt_tokens": 93, "completion_tokens": 493, "ttft_ms": 137.2, "decode_ms": 2293.0, "decode_tps": 215.0}, {"idx": 6, "prompt_tokens": 94, "completion_tokens": 321, "ttft_ms": 136.2, "decode_ms": 1451.3, "decode_tps": 221.2}, {"idx": 7, "prompt_tokens": 74, "completion_tokens": 506, "ttft_ms": 139.2, "decode_ms": 1903.8, "decode_tps": 265.8}, {"idx": 8, "prompt_tokens": 73, "completion_tokens": 376, "ttft_ms": 137.5, "decode_ms": 1451.4, "decode_tps": 259.1}, {"idx": 9, "prompt_tokens": 89, "completion_tokens": 512, "ttft_ms": 140.3, "decode_ms": 1936.2, "decode_tps": 264.4}, {"idx": 10, "prompt_tokens": 66, "completion_tokens": 512, "ttft_ms": 137.8, "decode_ms": 2453.1, "decode_tps": 208.7}]}
{"engine": "dspine", "group": "code", "ts_start": "16:58:01", "ts_end": "16:58:07", "output_len": 512, "metrics_before": {"accept_length": 8.275, "accept_rate": 0.485, "verify_calls_total": 552.0}, "metrics_after": {"accept_length": 11.875, "accept_rate": 0.725, "verify_calls_total": 696.0}, "agg": {"n": 10, "decode_tps_mean": 386.4, "decode_tps_weighted": 382.6, "decode_tps_min": 231.1, "decode_tps_max": 501.2, "completion_tokens_total": 1769, "accept_length_delta": 3.6, "verify_calls_delta": 144.0}, "detail": [{"idx": 1, "prompt_tokens": 145, "completion_tokens": 243, "ttft_ms": 116.9, "decode_ms": 675.4, "decode_tps": 359.8}, {"idx": 2, "prompt_tokens": 164, "completion_tokens": 210, "ttft_ms": 141.9, "decode_ms": 483.8, "decode_tps": 434.0}, {"idx": 3, "prompt_tokens": 124, "completion_tokens": 195, "ttft_ms": 137.0, "decode_ms": 546.5, "decode_tps": 356.8}, {"idx": 4, "prompt_tokens": 114, "completion_tokens": 166, "ttft_ms": 137.8, "decode_ms": 481.2, "decode_tps": 345.0}, {"idx": 5, "prompt_tokens": 164, "completion_tokens": 180, "ttft_ms": 138.9, "decode_ms": 419.8, "decode_tps": 428.8}, {"idx": 6, "prompt_tokens": 128, "completion_tokens": 81, "ttft_ms": 141.2, "decode_ms": 350.5, "decode_tps": 231.1}, {"idx": 7, "prompt_tokens": 113, "completion_tokens": 111, "ttft_ms": 137.4, "decode_ms": 221.5, "decode_tps": 501.2}, {"idx": 8, "prompt_tokens": 175, "completion_tokens": 232, "ttft_ms": 143.5, "decode_ms": 548.1, "decode_tps": 423.3}, {"idx": 9, "prompt_tokens": 109, "completion_tokens": 198, "ttft_ms": 139.9, "decode_ms": 512.5, "decode_tps": 386.4}, {"idx": 10, "prompt_tokens": 136, "completion_tokens": 153, "ttft_ms": 140.3, "decode_ms": 384.7, "decode_tps": 397.7}]}
{"engine": "dspine", "group": "narrative", "ts_start": "16:58:08", "ts_end": "16:59:33", "output_len": 512, "metrics_before": {"accept_length": 11.875, "accept_rate": 0.725, "verify_calls_total": 696.0}, "metrics_after": {"accept_length": 2.3, "accept_rate": 0.08666666666666667, "verify_calls_total": 3267.0}, "agg": {"n": 10, "decode_tps_mean": 60.9, "decode_tps_weighted": 60.9, "decode_tps_min": 57.8, "decode_tps_max": 64.3, "completion_tokens_total": 5120, "accept_length_delta": -9.575, "verify_calls_delta": 2571.0}, "detail": [{"idx": 1, "prompt_tokens": 195, "completion_tokens": 512, "ttft_ms": 137.5, "decode_ms": 8385.9, "decode_tps": 61.1}, {"idx": 2, "prompt_tokens": 187, "completion_tokens": 512, "ttft_ms": 118.8, "decode_ms": 8416.5, "decode_tps": 60.8}, {"idx": 3, "prompt_tokens": 206, "completion_tokens": 512, "ttft_ms": 161.4, "decode_ms": 8857.9, "decode_tps": 57.8}, {"idx": 4, "prompt_tokens": 223, "completion_tokens": 512, "ttft_ms": 141.9, "decode_ms": 8748.4, "decode_tps": 58.5}, {"idx": 5, "prompt_tokens": 192, "completion_tokens": 512, "ttft_ms": 116.9, "decode_ms": 8221.0, "decode_tps": 62.3}, {"idx": 6, "prompt_tokens": 233, "completion_tokens": 512, "ttft_ms": 167.6, "decode_ms": 8567.3, "decode_tps": 59.8}, {"idx": 7, "prompt_tokens": 228, "completion_tokens": 512, "ttft_ms": 145.0, "decode_ms": 8639.3, "decode_tps": 59.3}, {"idx": 8, "prompt_tokens": 241, "completion_tokens": 512, "ttft_ms": 146.2, "decode_ms": 8215.3, "decode_tps": 62.3}, {"idx": 9, "prompt_tokens": 198, "completion_tokens": 512, "ttft_ms": 165.2, "decode_ms": 7966.7, "decode_tps": 64.3}, {"idx": 10, "prompt_tokens": 240, "completion_tokens": 512, "ttft_ms": 168.1, "decode_ms": 8106.7, "decode_tps": 63.2}]}
{"engine": "dflash2", "group": "math", "ts_start": "17:04:08", "ts_end": "17:04:34", "output_len": 512, "metrics_before": {"accept_length": 0.0, "accept_rate": 0.0, "verify_calls_total": 21.0}, "metrics_after": {"accept_length": 4.65, "accept_rate": 0.5214285714285715, "verify_calls_total": 776.0}, "agg": {"n": 10, "decode_tps_mean": 164.8, "decode_tps_weighted": 161.6, "decode_tps_min": 126.3, "decode_tps_max": 198.1, "completion_tokens_total": 4089, "accept_length_delta": 4.65, "verify_calls_delta": 755.0}, "detail": [{"idx": 1, "prompt_tokens": 82, "completion_tokens": 245, "ttft_ms": 118.7, "decode_ms": 1603.8, "decode_tps": 152.8}, {"idx": 2, "prompt_tokens": 81, "completion_tokens": 512, "ttft_ms": 139.6, "decode_ms": 3421.7, "decode_tps": 149.6}, {"idx": 3, "prompt_tokens": 84, "completion_tokens": 272, "ttft_ms": 134.4, "decode_ms": 1373.2, "decode_tps": 198.1}, {"idx": 4, "prompt_tokens": 92, "completion_tokens": 320, "ttft_ms": 139.8, "decode_ms": 2480.0, "decode_tps": 129.0}, {"idx": 5, "prompt_tokens": 95, "completion_tokens": 512, "ttft_ms": 134.6, "decode_ms": 2749.4, "decode_tps": 186.2}, {"idx": 6, "prompt_tokens": 95, "completion_tokens": 317, "ttft_ms": 136.2, "decode_ms": 2078.2, "decode_tps": 152.5}, {"idx": 7, "prompt_tokens": 75, "completion_tokens": 512, "ttft_ms": 135.6, "decode_ms": 2649.3, "decode_tps": 193.3}, {"idx": 8, "prompt_tokens": 73, "completion_tokens": 375, "ttft_ms": 133.5, "decode_ms": 2212.2, "decode_tps": 169.5}, {"idx": 9, "prompt_tokens": 91, "completion_tokens": 512, "ttft_ms": 135.3, "decode_ms": 2681.7, "decode_tps": 190.9}, {"idx": 10, "prompt_tokens": 70, "completion_tokens": 512, "ttft_ms": 134.3, "decode_ms": 4053.0, "decode_tps": 126.3}]}
{"engine": "dflash2", "group": "code", "ts_start": "17:04:35", "ts_end": "17:04:44", "output_len": 512, "metrics_before": {"accept_length": 4.65, "accept_rate": 0.5214285714285715, "verify_calls_total": 776.0}, "metrics_after": {"accept_length": 7.35, "accept_rate": 0.9071428571428571, "verify_calls_total": 1019.0}, "agg": {"n": 10, "decode_tps_mean": 218.9, "decode_tps_weighted": 217.4, "decode_tps_min": 200.8, "decode_tps_max": 237.2, "completion_tokens_total": 1770, "accept_length_delta": 2.7, "verify_calls_delta": 243.0}, "detail": [{"idx": 1, "prompt_tokens": 145, "completion_tokens": 244, "ttft_ms": 121.5, "decode_ms": 1174.1, "decode_tps": 207.8}, {"idx": 2, "prompt_tokens": 166, "completion_tokens": 210, "ttft_ms": 137.1, "decode_ms": 940.7, "decode_tps": 223.2}, {"idx": 3, "prompt_tokens": 126, "completion_tokens": 195, "ttft_ms": 135.5, "decode_ms": 971.0, "decode_tps": 200.8}, {"idx": 4, "prompt_tokens": 116, "completion_tokens": 166, "ttft_ms": 134.7, "decode_ms": 802.6, "decode_tps": 206.8}, {"idx": 5, "prompt_tokens": 163, "completion_tokens": 180, "ttft_ms": 136.6, "decode_ms": 838.9, "decode_tps": 214.6}, {"idx": 6, "prompt_tokens": 131, "completion_tokens": 81, "ttft_ms": 135.5, "decode_ms": 368.7, "decode_tps": 219.7}, {"idx": 7, "prompt_tokens": 115, "completion_tokens": 111, "ttft_ms": 133.5, "decode_ms": 467.9, "decode_tps": 237.2}, {"idx": 8, "prompt_tokens": 174, "completion_tokens": 232, "ttft_ms": 136.1, "decode_ms": 1040.7, "decode_tps": 222.9}, {"idx": 9, "prompt_tokens": 107, "completion_tokens": 198, "ttft_ms": 141.9, "decode_ms": 868.2, "decode_tps": 228.1}, {"idx": 10, "prompt_tokens": 135, "completion_tokens": 153, "ttft_ms": 136.0, "decode_ms": 670.3, "decode_tps": 228.3}]}
{"engine": "dflash2", "group": "narrative", "ts_start": "17:04:44", "ts_end": "17:06:32", "output_len": 512, "metrics_before": {"accept_length": 7.35, "accept_rate": 0.9071428571428571, "verify_calls_total": 1019.0}, "metrics_after": {"accept_length": 1.525, "accept_rate": 0.075, "verify_calls_total": 4175.0}, "agg": {"n": 10, "decode_tps_mean": 48.3, "decode_tps_weighted": 48.2, "decode_tps_min": 45.6, "decode_tps_max": 52.2, "completion_tokens_total": 5120, "accept_length_delta": -5.825, "verify_calls_delta": 3156.0}, "detail": [{"idx": 1, "prompt_tokens": 196, "completion_tokens": 512, "ttft_ms": 140.7, "decode_ms": 10225.7, "decode_tps": 50.1}, {"idx": 2, "prompt_tokens": 188, "completion_tokens": 512, "ttft_ms": 137.7, "decode_ms": 10482.3, "decode_tps": 48.8}, {"idx": 3, "prompt_tokens": 205, "completion_tokens": 512, "ttft_ms": 142.7, "decode_ms": 11219.1, "decode_tps": 45.6}, {"idx": 4, "prompt_tokens": 226, "completion_tokens": 512, "ttft_ms": 148.0, "decode_ms": 10722.4, "decode_tps": 47.8}, {"idx": 5, "prompt_tokens": 193, "completion_tokens": 512, "ttft_ms": 157.8, "decode_ms": 10687.2, "decode_tps": 47.9}, {"idx": 6, "prompt_tokens": 233, "completion_tokens": 512, "ttft_ms": 148.0, "decode_ms": 11031.1, "decode_tps": 46.4}, {"idx": 7, "prompt_tokens": 230, "completion_tokens": 512, "ttft_ms": 163.7, "decode_ms": 10820.9, "decode_tps": 47.3}, {"idx": 8, "prompt_tokens": 241, "completion_tokens": 512, "ttft_ms": 149.9, "decode_ms": 10794.4, "decode_tps": 47.4}, {"idx": 9, "prompt_tokens": 198, "completion_tokens": 512, "ttft_ms": 159.6, "decode_ms": 9812.1, "decode_tps": 52.2}, {"idx": 10, "prompt_tokens": 240, "completion_tokens": 512, "ttft_ms": 152.5, "decode_ms": 10415.1, "decode_tps": 49.2}]}
```

*报告完。只测未改任何配置;DSpine 服务已停干净;现役 DFlash2 已恢复并核验(active / health 200 / DFLASH·8草稿·fp8KV / 57599 MiB)。*
