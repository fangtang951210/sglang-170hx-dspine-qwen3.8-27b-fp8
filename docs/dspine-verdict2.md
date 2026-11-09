# 170hx DSpine 接受率两极分化裁决实验报告(verdict2)

- 工单:task_b5135bab3519 · 2026-11-09 18:33–18:54 CST · 主机 170hx(CMP 170HX 64G,sglang 0.5.22,DSpine 第 10 次启动)
- 裁决问题:判定实验测得数学 7.51 / 代码 12.28 接受率(decode 237/386),文章总结负载 1.025(decode 23.9)——用户质疑两者必有一假:要么之前测错,要么 DSpine 压根没工作
- 口径(按控制台中途修正):全部走用户工具框架——`bench_verdict.py` = `bench_tool.py v2.2` 只加 `--corpus` 参数换 gen_prompt(五负载变体;realtext 分支与 `bench_realtext.py` 逐字节一致已验证),其余(唯一前缀防缓存 / temperature 0 / enable_thinking=false / 流式首字-完成计时 / 真实 usage 计速 / --words/--concurrency/--output-len/--tag / jsonl 追加)一字不动。每组 8 条 = 工具连续 8 次调用(每次 --concurrency 1),--words 8000 --output-len 512,组间 sleep 10,五组背靠背同服务零重启零改动;预热 1 条代码后记基线 A0。tag 前缀 `裁决-`。禁自写请求循环,遵办。

## 一、裁决结论(先说判决)

**两批数据都是真的,谁也不假。极化的判别变量是提示词长度,不是负载题材——DSpine 草稿在 ~9K token 上下文以上整体失效(代码/数学也塌),短上下文(≤810 token 实测)工作正常。**

按单子 rubric 形式上落「G1/G2 也低 = 系统问题」分支(8K 词档五负载全灭),但「前数据不可信」这一半**不成立**——本单在**同一实例、同一配置、相隔数分钟**内把两批前数据都复现了:

- 短提示词诊断(测完五组后、停服前补测,零改动):代码 306 token → decode **441.8 tps**、接受率窗口 1.025→**5.625**(硬数 1005/138 = **7.28**)——复现判定实验量级(386.4 tps / 12.28);
- 8K 词文章总结组:接受率 1.10 / decode 32.4——复现真实语料测速(1.025 / 32.8)。

即:「DSpine 没工作」错(短档在工作),「之前测错」也错(两档各自复现)。**它只在短上下文工作**——这才是两极分化的根因。mamba/SSM 草稿对长上下文失效的机制方向与部署日志遗留的 mamba 槽位/ReplaySSM 复杂度一致,精确根因排查不在本单范围。

## 二、主表(五组,8K 词档,背靠背)

| 组 | 负载 | prompt tokens | 接受率(硬数 Σct/Δverify) | gauge Ai | 任务口径 ΔAi | decode tps 均值 | decode tps 加权 |
|---|---|---|---|---|---|---|---|
| G1 | 代码续写 | 16354 | **1.04** | 1.000 | 0.000 | 29.7 | 30.0 |
| G2 | 数学应用题 | 10459 | **1.01** | 1.000 | 0.000 | 29.7 | 29.6 |
| G3 | 文章总结 | 9229 | **1.10** | 1.025 | +0.025 | 32.4 | 32.4 |
| G4 | 常识问答 | 9400 | **1.08** | 1.050 | +0.025 | 32.0 | 31.9 |
| G5 | 中译英 | 19858 | **1.04** | 1.025 | −0.025 | 29.8 | 29.8 |
| 整体 | — | — | **1.065**(13066/12267) | 1.025 | — | — | — |

- 每组 8 条,verify 增量 1040–3782,样本量充分;TTFT 4–8 s、prefill ~2000 tps(8K–20K token 提示词),prefill 不构成 decode 判读因素。
- **ΔAi 口径说明(如实)**:spec_accept_length gauge 是窗口均值,本实验全程钉在 1.0 附近,故 ΔAi ≈ 0 无信息量;各组真实接受率以硬数(Σcompletion_tokens ÷ Δspec_verify_calls_total 计数器)为准——与判定实验同做法。G3 组 1.10 与原真实语料测速的 1.025 同档。
- 五组接受率 1.01–1.10:每步验证平均只收下 bonus 那一个 token,16 个草稿 token 全白算;decode 29.6–32.4 tps 全部低于裸跑实测 37.6–43.0(判定实验同机参照)——8K 词档下 DSpine 比不开投机还慢 ~25%。

## 三、同实例短提示词诊断组(判别变量的直接证据)

| 诊断 | corpus | prompt tokens | completion | decode tps |
|---|---|---|---|---|
| 短代码 | code,--words 150 | 306 | 377 | **441.8** |
| 短数学 | math,--words 150 | 211 | 512 | 152.1 |
| 中代码 | code,--words 400 | 810 | 116 | **449.5** |

诊断窗口硬数接受率 = (377+512+116)/(14307−14169) = 1005/138 = **7.28**(gauge 1.025 → 5.625)。

同一 DSpine 实例、同一配置、同一工具,唯一变量是提示词长度:**810 token → 接受率 7.28、decode 441.8;9229 token → 接受率 1.10、decode 32.4**。崩塌边界在 (810, 9229] 区间内,未二分(见遗留)。

## 四、生成质量样本(证明非乱码;同种子同构请求抓取,temperature 0,各 ~100 字)

- **G1 代码**:```python import asyncio import random  class TransientError(Exception):     pass  def merge_interva…(完整补全函数实现,语法正常)
- **G2 数学**:Here is the step-by-step solution for the last problem presented in the text.  **Problem:** Sarah bo…(正确选题并分步解答)
- **G3 总结**:Based on the text provided, here is a detailed summary of the key points regarding GPU memory hierar…(结构化分点总结)
- **G4 常识**:Based on the text provided, here is a detailed explanation of how to choose a solid-state drive (SSD…(就末段问题详答)
- **G5 中译英**:The autumn on the western Sichuan Plateau is the favorite season for photography enthusiasts. Beneat…(川西高原段落,译文流畅准确)

五组输出全部正常——底模生成质量无恙,塌的只是草稿接受率,排除「服务产出乱码级故障」。

## 五、运营含义(供控制台拍板)

1. **DSpine 不适合接替现役**:真实业务凡提示词到 8K 词量级(长文总结/长文翻译/RAG 带上下文)零收益且负收益(decode ~30 < 裸跑 ~40);只有短交互(几百 token,如对话续写、短代码补全)吃到 5–10× 加速。
2. 判定实验的「数学/代码高接受」结论**仅在短上下文成立**,不能外推到 8K+ 档;真实语料测速的 1.025 是 8K+ 档的普遍值,不是总结负载特有——本单 G1/G2 在同档同样 1.0。
3. 若业务负载画像以短 prompt 为主,DSpine 仍是有效选项;混合负载则不如现役 DFlash2(其在同负载对照中 8K 档未测,但 100K 随机词历史 79.9 tps 远高于 DSpine 长档 ~24-32)。

## 六、过程与恢复(服务终态:DFlash2 运行中 ✓)

18:33 停 sglang-llm(现役 DFlash2,停前指标 0.0/0.0/3.0)显存归零 → 18:34 起 DSpine(`launch_dspine.sh` 第 10 次,PID 77168,启动日志与八枪定版一致:DSPINE 草稿 b16·fa3·context 131072·mem 0.80)→ 18:35:54 health 200 → 18:36:27–18:50:32 预热+五组背靠背(零重启零改动)→ 18:50:40 诊断组 3 条 → 18:51:16 SIGTERM 精确 PID,显存归零 → 18:51:49 `systemctl --user start sglang-llm` → **18:54:03 health 200,显存 57597 MiB(与停前一致),身份核验 DFLASH·speculative_num_draft_tokens 8·kv fp8_e4m3·DFlash2DraftModel 3.69G 加载行**——现役完全恢复。

## 七、产物

- 170hx:`~/deploy/bench_verdict.py`(统一 --corpus 工具,可复用)、`verdict_driver.sh`、`verdict2_sample.py`、`verdict2_analysis.py`、`verdict2-driver.jsonl`(13 个指标事件)、`verdict2-analysis.json`、`logs/verdict2-run.log`、`verdict2-sample-*.txt` ×5、`/tmp/dspine-launch-10.log`(易失,已存档);`bench-tool-results.jsonl` 追加 44 行(tag `裁决-*`,含预热与诊断)
- 本地存档:派工区 `2026-11-09-DSpine裁决实验/`(脚本 ×3、报告、驱动日志、事件流、五样本、启动日志、结果 jsonl、分析 json)+ 本报告 `/Users/sugar/Documents/tmp/dspine-verdict2.md`

## 八、遗留(不属本判定,如实记)

- 崩塌边界未二分:(810, 9229] token 之间;控制台要精确阈值可再开工单(同框架 --words 1000/2000/4000 一轮即出)。
- 短数学诊断 decode 152.1 低于短代码 441.8,方向与判定实验(237.7 vs 386.4)一致但系单条样本未成组。
- DSpine 长上下文失效的精确根因(mamba 状态池/ReplaySSM/粗检索补丁哪一环)未排查,本单只裁决真伪。

*报告完。只测未改任何配置;DSpine 已停干净;现役 DFlash2 已恢复并三重核验(active / health 200 / 身份一致)。*
