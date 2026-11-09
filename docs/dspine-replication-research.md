# DSpine 长上下文复现失败 · 外部情报调研报告

- 工单:task_275d26393e6a(外部情报线,纯联网查证,未碰任何内网机器/服务)
- 日期:2026-11-09 · 调研人:外部情报工人
- 问题:DSpine(sglang 0.5.22,Qwen3.8-27B-FP8,CMP 170HX/SM80)短提示词(≤810 tok)接受率 7.28、decode 449 tps 完美;提示词 ≥9K tok 后一切负载接受率崩至 1.0、decode 29-32 tps(裸跑 ~40,负收益)。崩塌边界在 810-9229 之间。官方宣称 3.3x(WorkBuddy xhigh,并发 8)。

---

## 1. 草稿模型规格:tencent/Qwen3.8-27B-DSpine-b16

**结论:草稿模型是一个 5 层、全部层为滑窗注意力的小模型,`sliding_window = 8192`(每层只能看见最近 8192 token);论文训练最大序列长度仅 3,072 token;"b16" = block size 16(每步并行起草 16 个 token)。官方对长输入的处理没有任何特殊设计说明——8192 就是架构硬边界,恰落在实测崩塌区间(810-9229)正中间。**

证据:
- config.json(实测抓取):`"num_hidden_layers": 5`,layer_types 全部 `sliding_attention`;`"sliding_window": 8192`,`"use_sliding_window": true`;`"max_position_embeddings": 262144`(位置编码上限很大,但注意力每层只看最近 8192);架构 `DFlashDraftModel`(DSpine 复用 sglang 的 DFlash 家族加载器);`mask_token_id: 248070`、`draft_block_causal: true`;dflash_config 含 `target_layer_ids: [1,16,31,46,61]`——**草稿条件于目标模型第 1/16/31/46/61 层隐状态**(prefix code bridge,rank 512/code 1024 的压缩桥接),不是独立语言模型。
  来源:https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/raw/main/config.json
- HF 模型卡原话:"This checkpoint is a five-layer draft model for Qwen3.8-27B, **with block size 16 and an 8,192-token sliding window**."。全文无任何长提示词处理、截断策略、长上下文性能的说明。
  来源:https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16
- 论文(arXiv 2609.36173,Appendix A Training setup):"a maximum training sequence length of **3,072 tokens**, and six epochs (3,655 steps)",训练数据 ShareGPT+目标再生成。Appendix C:"All methods receive identical prompts without a system message and generate at most 2,048 new tokens"。**通篇不讨论滑窗、不讨论长上下文表现**——论文级草稿是在 ≤3K 序列上训出来的。
  来源:https://arxiv.org/abs/2609.36173 · https://arxiv.org/html/2609.36173v1
- 对照:前代官方草稿 z-lab/Qwen3.8-27B-DFlash2 的 config `sliding_window = 2048`——DFlash 家族的草稿本来就是"小滑窗"设计哲学。
  来源:https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json

## 2. 官方 benchmark 条件:3.3x 是怎么测出来的

**结论:官方全部证据都在短提示词/局部长上下文负载上。通用基准(GSM8K/MATH-500/HumanEval/MBPP/MT-Bench)提示词都是百级 token;agent 基准(WorkBuddy/SWE-bench Pro/τ²-bench)接受率 4.7-5.6、加速 2.7-3.84x,但官方从未公布提示词长度分布。对"长上下文(>8K 输入)下表现如何"官方零声明、零数据——既没说行,也没说不行。**

证据:
- HF 卡 Evaluation 节:Runtime = **SGLang 0.5.22, BF16, FlashAttention 3, 并发 8**;通用基准 "at most 4,096 new tokens"(限制的是生成长度,提示词本身是标准短题);agent 基准 "20 conversations per subset, tool-call prefixes replayed in order, prefix caching, up to 32,768 new tokens per request"。3.33x = WorkBuddy Web xhigh;最高 3.84x = WorkBuddy Office low。
  来源:https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16
- GitHub README:论文侧只在 Qwen3-4B/8B 上测 7 个数学/代码/对话基准(平均接受长度 7.83@T=0);Qwen3-8B 官方启动示例 `--context-length 8192`——**小模型配方把上下文钉在 8192,正好等于滑窗大小**;27B 配方 context 131072 但无长上下文性能说明。
  来源:https://github.com/Tencent/DSpine
- WorkBuddy Bench 仓库与数据集卡:260 个任务(Code 80/Web 70/Office 50/Security 60),数据集卡**没有**任何轨迹 token 长度统计;官方评测用的 CodeBuddy 沙箱轨迹长度分布未公开。(注:agent 会话逐轮累积上下文,官方接受率 4-5 说明其负载在官方环境可用,但无法据此断定其单请求输入超过 8192。)
  来源:https://github.com/Tencent/workbuddy-bench · https://huggingface.co/datasets/tencent/workbuddy-bench
- 项目页 https://dspine-project.github.io :无 FAQ、无长上下文/使用限制的任何声明。

## 3. GitHub issues / discussions:有没有人报过同样现象

**结论:没有一模一样的报告(DSpine 开源仅 2 天),但同家族(DFlash 系草稿 + Qwen3.8 混合线性注意力目标)存在多个"接受率钉死 1.00"的未修复 issue,其中 #43514 就是同款目标模型 Qwen3.8-27B,触发因素含"长提示词多 chunk 进场"。官方对 DSpine 长上下文问题目前零回应、零修复。**

- Tencent/DSpine 仓库:仅 2 个 issue(#1 训练预算质疑、#2 RTX 6000 PRO flashinfer 支持),均与长上下文无关。来源:https://github.com/Tencent/DSpine/issues
- sgl-project/sglang:标题含 "DSPINE" 的 issue 为 0。但同家族:
  - **#43514**(2026-10-10,open):**Qwen3.8-27B + DFlash2**,v0.5.19/0.5.20,H20×4,**flashinfer 后端**,非 FA3——"accept rate drops to exactly 0.00 / accept length 1.00",触发需两要素同时:语法约束流量 + **~50K 新增长提示词多 chunk 进场**;整批毒化、跨请求持续;上报者定位 verify 路径 stale vocab mask,**修复 PR #43518("dflash verify stal…")至今未合入上游**;官方无回复。
    来源:https://github.com/sgl-project/sglang/issues/43514
  - **#49274**(2026-11-09,open):Kimi-K3.1-Pro(hybrid mamba)+ DFlash2,MI300X/AITER:长提示词后"accept len 1.00, accept rate 0.00"钉死 5.9 小时,重启才恢复;关联 #38319(chunked-prefill radix 竞态,修复 PR #38355 未合入)、#48964(v0.5.23 百万 token 请求后回归)。
    来源:https://github.com/sgl-project/sglang/issues/49274
  - 相关但较远:#46752(DSPARK/DFLASH 忽略 repetition/frequency penalty)、#49199(草稿层无量纲名导致量化排除失效)、#46806(权重热更新可静默覆盖草稿权重)。
- 判别要点:这些 bug 都是**概率触发、依赖负载历史、跨请求锁存**;而我们的现象是**提示词长度的确定性函数**(裁决实验:先跑完 8K 档五组全灭,同实例紧接短提示词诊断组立刻恢复 7.28,无锁存)——说明我们踩的更可能是确定性的架构边界(§1),而非上述状态污染类 bug。但它们证明了:该代码路径在非 FA3/非 Hopper 组合下确有未修缺陷,不能排除"滑窗 mask 在绕 FA3 的内核上 >8192 时构造错误"把降级放大成崩塌。

## 4. sglang 0.5.22 实现细节

**结论:DSPINE 走 `dflash_worker_v2` 工作器;官方启动命令不设 `--speculative-draft-window-size`,草稿 KV 与目标共享全序列分配、靠注意力 mask 实现滑窗(window_left=8191);该 flag 是显存优化的紧凑 KV 旋钮(须 ≥16),不是质量旋钮。0.5.22(tag 2026-10-15)已含 DFLASH verify-mask-policy 对 wrapped 后端的修复(#43054),但官方文档明言"不同注意力后端的接受率不可同表比较"——官方数字全部产自 FA3+BF16。**

- 算法注册与文件:v0.5.22 `python/sglang/srt/speculative/` 下无独立 DSPINE 文件,由 `dflash_worker_v2.py`(128KB)/`dflash_info_v2.py`/`dflash_utils.py` 承载,草稿 checkpoint 的 `dflash_config` 字段驱动行为(DSPINE=b16 新一代,DFlash2=b8 前代,同一 worker)。
  来源:https://github.com/sgl-project/sglang/tree/v0.5.22/python/sglang/srt/speculative
- 草稿输入构造:`dflash_worker_v2.py` — `self.draft_window_size = get_spec().speculative_draft_window_size`;不设时 `use_compact_draft_cache=False`,注释原文:"Without draft windowing, the draft worker **aliases the target request->token mapping** and allocation state… With draft windowing… **draft attention sees only the recent window**"。即默认路径草稿照样对全序列建 KV,滑窗由注意力层 mask 实现;滑窗大小来自 checkpoint config(`get_dflash_attention_sliding_window_size`,HF 窗口含当前 token,sglang 存 window_left=sliding_window-1=8191,`dflash_utils.py:418-435`)。
- `--speculative-draft-window-size` 官方文档定义:"Draft KV sliding-window size. **Must be >= speculative-num-draft-tokens when set**",默认 None;另一处:"DFlash-only draft KV sliding-window size"。用途是省草稿 KV 显存(紧凑缓存+私	req→token 表),**不是**让草稿"看见"更多上下文——设了也不能修复超窗质量问题。
  来源:https://github.com/sgl-project/sglang/blob/v0.5.22/docs/docs/advanced_features/speculative_decoding.mdx
- verify mask 路径:`resolve_dflash_verify_mask_policy`(dflash_utils.py:208)按后端类名查跳过表(FA3/FlashInfer/Triton/TRTLLM 等免 custom mask);wrapped 后端(Hybrid/Tbo)解析修复 #43054 于 2026-10-14 合并,**v0.5.22 tag 于 2026-10-15,已包含**(排除"缺这个修复"作为根因)。
  来源:https://github.com/sgl-project/sglang/pull/43054 · tag 日期实测核验
- 0.5.23(2026-11-02)在其后又合入一批 DFlash 相关修复:("[Spec] Fix DFLASH verify mask policy for wrapped attention backends" #43054 在列说明 0.5.23 与 0.5.22 窗口有重叠)、"[Fix] deepseek-v4.1: Start the PD decode draft SWA window at the prefill draft valid-from" #45037、"[Perf] speculative: Fuse draft write-window gathering and KV translation" #44972 等——**草稿 SWA 窗口相关代码在 0.5.22 之后仍在活跃修**,长上下文+SWA 是热修补区,但 §3 的核心 issue 均未关闭。
  来源:https://github.com/sgl-project/sglang/releases
- 官方文档原话(重要):"**Backends also differ in attention numerics, so acceptance measured on different backends does not belong in one table.**" 官方 DSpine 数字全部是 FA3+BF16;我们 SM80 上 fa3 实为绕行/FA2 内核(启动日志 attention_backend=fa3、draft fa3、speculative_attention_mode=prefill、chunked_prefill_size=2048),属官方未测试组合。
- 内存机制佐证(GGUF 量化卡):混合线性注意力目标要"为每个起草 token 保存一份循环状态副本"(~150MB/份,16-token 块比 8-token 贵 ~1.2GB)——这解释了为何 170HX 64G 装不下官方 `--max-mamba-cache-size 64` 而被迫降到 16(实测把并发钳到 3)。
  来源:https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF

## 5. 理论层面:草稿-目标漂移随上下文增长

**结论:这是推测解码的已知现象,有专门论文(LongSpec, ICML 2025)研究:草稿在短序列上训练、在长上下文下使用,接受率会显著崩塌。DSpine 草稿训练序列仅 3,072 token、推理时却要对付 9K-20K 输入,超出训练分布 3-6 倍;加上 8192 滑窗使早期上下文完全不可见,5 层小草稿(且其上下文表征依赖目标模型隐状态桥接)没有冗余容量补偿,崩到 1.0 与理论预期一致。**

- LongSpec: Long-Context Lossless Speculative Decoding(ICML 2025):明确指出短序列训练的草稿在长上下文下失效,需长序列训练草稿才能维持接受率。
  来源:https://icml.cc/virtual/2025/51846
- 工程实践:"Get 3× Faster LLM Inference with Speculative Decoding" 一文指出 **EAGLE-3 草稿头未针对长上下文优化,长提示词下接受率因分布失配骤降**(该文 URL 经搜索引擎聚合跳转,原始域名未能直接核验,标题可检索);"When should you NOT be using speculative decoding"(2026-06):短答案+长上下文 RAG 场景 SD 常输给裸解码,context 一长 acceptance 掉得又快又狠;社区讨论原话:"Long context and you win against speculative decoding because it drops sharp"。
  来源:https://x.com/drummatick/article/2068370473650647202(直接可访)· 其余两篇经 WebSearch 检索标题可得
- 反向参考(长上下文下 SD 仍可赚):Together AI 博客证明大 batch 长上下文服务场景 SD 仍可提升吞吐——但前提是草稿与负载匹配,与本案不矛盾。
  来源:https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference
- 量化到本案:acceptance=1.0 且 draft block=16 时,每步 = 草稿前向 + 16-token 验证全废,纯开销;判定实验已验证效率因子恒定(~0.72-0.73),decode 29-32 < 裸跑 37.6-43.0 的负收益机制自洽(与判定/裁决报告互证,非本报告新证据)。

## 6. 社区实践:非 H 卡 / FP8 / 长上下文跑 DSpine

**结论:sglang 侧 A100/3090/4090 + FP8 + 长上下文跑 DSpine 的经验帖,截至今日(开源第 2 天)一篇都没有——搜不到就是搜不到。唯一的非官方实战数据在 llama.cpp fork 的 GGUF 量化卡及其 HF discussions:AMD 卡(RX 7800 XT/R9700 ROCm)与 DGX Spark 上,短上下文代码负载 ~2.2x,但 prose/聊天/图像负载 4-23% **负优化**,n-max(草稿块)大了直接掉速——社区已经在非官方环境复现"负载/环境不匹配→负收益"模式。**

- aj9o9/Qwen3.8-27B-DSpine-b16-GGUF(第三方,llama.cpp 需作者 fork 的 dspine 分支,主库尚不支持):上下文预算表显示 24G 卡上 n-max 15 只剩 65,536 上下文(n-max 7 → 98,304;DFlash2 → 102,400;无投机 → 176,128);HF discussions 实测:RX 7800 XT(Vulkan)n-max 2 得 47.4 tok/s,n-max 15 掉 19.1;DGX Spark 上 DFlash2 在 math/prose/summary 反超 DSpine;R9700(ROCm)代码 ~2.2x 但 prose/chat/图像慢 4-23%。量化作者还注明 16-token 块的循环状态副本开销,以及"output can diverge from non-draft runs even at temperature 0"。
  来源:https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF
- Reddit r/LocalLLM 讨论帖存在(Tencent released DSpine, a parallel 16-token draft model),但 reddit 抓取被反爬拦截,评论内容未能核验,如实注明。
  来源:https://www.reddit.com/r/LocalLLM/comments/1xrjntb/
- AICoder 等新闻稿只有发布信息,无实战数据(略)。

---

## 最可能根因排序(按证据强度)

**Top 1(证据:强)草稿模型架构与训练边界:8192 滑窗 + 训练序列 3,072。**
提示词超 8192 后草稿对窗口外上文彻底失明,且 9K-20K 输入超出其 ≤3K 训练分布 3-6 倍;5 层小草稿(依赖目标隐状态桥接)无冗余补偿,接受率断崖至 1.0。与全部现象同构:边界吻合(810✓/9229✗,8192 居中)、负载无关、确定性触发、无跨请求锁存、短提示词完美。弱点:官方 agent 基准(接受率 4-5)未公布提示词长度,若其上下文确超 8K 则说明官方环境下滑窗裁剪尚可工作——无法证伪也无法证实。
**Top 2(证据:中强)非官方后端组合(绕 FA3/SM80+FP8)× 0.5.22 长提示词路径的未修 bug。**
官方数字全部 FA3+BF16+H 系;文档明言后端数值差异致接受率不可比;同家族"accept 钉 1.00"未修 issue 两枚(#43514 同款目标模型、触发含长提示词多 chunk;#49274 hybrid mamba),修复 PR(#43518/#38355)未合入。我们 chunked_prefill_size=2048,9K 提示词=多 chunk 进场,正落在该 bug 家族触发面上。弱点:这些 bug 均为概率性/锁存型,与我们的确定性干净崩塌形态不同,更像是"放大器"而非"扳机"。
**Top 3(证据:中)配方偏离的累积效应:mamba cache 16(官方 64,并发被钳到 3)+ FP8 量化底模(官方 BF16)。**
64G 卡装不下官方 64 槽是被动的(16-token 块的循环状态开销大);FP8-on-SM80 走反量化路径。两者不产生 8192 阈值,但改变数值路径与调度形态,放大负收益幅度,且全部落在官方零测试区。

**复合判断**:崩塌**阈值**最可能来自 8192 滑窗(Top1),崩塌**深度**(掉到 1.0 而非 3-4)可能由 Top2 的环境特异性路径放大。给出三个可分离二者的判别实验(给实地线):
1. **边界二分**:2000/2049/3000/6000/8190/8192/8194/9000 token 提示词逐档测。若边界=8192→滑窗实锤;若边界≈2048(chunked_prefill_size)→多 chunk 状态 bug 实锤(810 与 9229 之间两者都可能)。
2. **改 chunk 再测**:`--chunked-prefill-size 32768`(让 9K-20K 提示词单 chunk 进场)。若长提示词恢复接受率→Top2(chunk 竞态)升为第一根因;若不变→Top1 锁死。
3. **DFlash2 同负载对照**:现役 DFlash2(SWA 仅 2048)跑同批 9K-20K 提示词。若 DFlash2 也崩(且边界≈2048)→家族性滑窗行为;若 DFlash2 在 16K 仍 4+ 而 DSpine 8K 崩→DSpine b16 特有。

## 官方推荐做法清单(逐条带来源)

1. **长上下文(输入 >8K)负载:官方没有任何背书、数据或声明**(模型卡/README/论文/项目页四处置核均无)——官方等于没说行,也没说不行;当前把 DSpine 用在 >8K 输入属用户自担区。
2. 短上下文按官方配方才有宣称数字:**SGLang 0.5.22 + BF16 + FA3 + `--max-mamba-cache-size 64` + 并发 8 + flashinfer 采样**(HF 卡 Evaluation 节),偏离任一项(FA3 绕行/FP8/16 槽)即进入无支持配置,官方明言"不同后端接受率不可同表比较"。
3. `--speculative-draft-window-size`(须 ≥16)只是草稿 KV 显存紧凑化旋钮,官方 DSpine 命令不设;设它**不解决**超窗质量问题(文档语义核验)。
4. 无现成修复可装:核心相关 issue(#43514/#49274)与修复 PR(#43518/#38355)截稿时均未合入上游;0.5.23 含若干 DFlash/SWA 修复但非对症。**建议把本案(170HX/SM80/非 FA3 真核/FP8/边界二分数据)报给 sgl-project/sglang 与 Tencent/DSpine**,填补非 H 系空白。
5. llama.cpp fork(GGUF)是唯一有非官方硬件实测的路线,仍实验性,且同样显示负载不匹配时负优化。
6. 实操建议(按判别实验结果):若确认 8192 边界→DSpine 定位为"短上下文加速器",长提示词流量回退 DFlash2(其 2048 滑窗若实测同样崩,则长上下文一律裸跑)或按负载分流;若 chunk 实验翻案→追 #43518 上游进度。

## 附:本报告未能查到的事项(如实)

- WorkBuddy/SWE-bench Pro/τ²-bench 官方评测轨迹的单请求输入 token 分布(未公布)。
- Reddit 讨论帖评论正文(反爬拦截,仅确认帖子存在)。
- sglang 侧 A100/3090/4090/FP8/长上下文跑 DSpine 的第三方经验帖(开源 2 天,尚不存在)。
- "DSPINE" 在 sglang issue 区零命中(issue 全文搜索仅覆盖标题+正文索引,讨论区未逐一翻页,159 页 DFlash 结果只看了第一页)。
