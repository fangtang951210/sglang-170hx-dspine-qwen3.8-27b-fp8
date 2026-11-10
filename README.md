# sglang + DSpine(CMP 170HX / SM80 / FP8)复现 Qwen3.8-27B 投机解码全记录:部署踩坑 → 测速 → 发现长上下文滑窗掩码 bug → 修复 → 256K 达成 → 数据

> 关键词:sglang · DSpine · speculative decoding · CMP 170HX · SM80 / Ampere · FP8 · Qwen3.8-27B · long context · sliding window bug · **YARN rope extrapolation fix · 256K context** · SPS tuning (negative result)

一张矿卡(NVIDIA CMP 170HX,SM80/Ampere,64G 显存)上用 **FP8 底模**复现腾讯 DSpine 并行起草投机解码的完整实战记录:sglang 0.5.20→0.5.22 严格升级 → 八枪闯过五关部署成功 → 测速发现 decode 全线崩到裸跑以下 → 五负载裁决+外部调研锁定方向 → **定位到 sglang 0.5.22 + SM80 上 FA 滑窗掩码分支输出垃圾的确定性 bug(边界=草稿 config 的 `sliding_window: 8192`)→ 两处同步改 262144 修复,长上下文全档恢复**。随后把上下文拉到 256K:先以 mf 0.90+mamba 8 槽+chunked 1024 硬挤进 KV 池 262688(代价:接受率 0.27→0.037、decode 腰斩),再**定位第二层根因——草稿 rope 训练仅覆盖 ~3072 位置、262144 全程 85 倍外推,草稿 config `rope_parameters` 写 YARN(factor 85.3333)修复,256K 接受率恢复 100K 甜点档水平(5.075)、decode 74–83 tps(投机收益 ×2)**。全部数据同机同工具实测,含原始 JSONL 与真实 diff,欢迎复现。SPS 调优按官方流程实测:三档负载提升全部 <5%,判定负结果已回退(负结果也是数据,见下文专节)。

## 测速:网页测速工具 v2.2 本体实测(单流,短上下文梯度)

> 数据来源:用户亲手用网页测速工具本体实测,截图存 [`screenshots/final-benchmark-user.png`](screenshots/final-benchmark-user.png)。条件:单流,备注 `qwen3.8-27b-fp8-170hx*1`,超时 30s,提示词 8 档递增(609→7833 tok)。

| 提示词 tok | 预填充 tok/s | 输出 tok/s |
|---|---|---|
| 609 | 1897.20 | 273.30 |
| 1642 | 1929.04 | 295.13 |
| 2673 | 2118.23 | 262.81 |
| 3704 | 2029.03 | 217.56 |
| 4737 | 2073.54 | 259.74 |
| 5768 | 2053.47 | 220.80 |
| 6799 | 2088.34 | 210.06 |
| 7833 | 2057.96 | 220.07 |
| **汇总(8 档)** | **平均 2030.85** | **平均 244.93**(P50 240.27 / P90 279.85 / P95 287.49) |

## 附:脚本口径交叉验证

主表为用户工具本体实测,下表为同构脚本口径(含长上下文与并发档),两者测量负载不同,不可直接对比。

| 场景(Qwen3.8-27B-FP8 + DSpine b16 草稿) | decode 吞吐 | 修复前 | 对照 |
|---|---|---|---|
| 短上下文真实负载(810 tok 代码续写,单流) | **449.5 tok/s**(接受率 7.28) | 同水平(短档本就正常) | 裸跑 AR 同机 ~40 |
| 100K 真实语料(115K tok 文章总结,单流) | **106.8 tok/s** | 23.9(崩) | 裸跑 100K:37.9 |
| 56K×4 并发真实语料 | **133.4 tok/s**(4 流合计) | 60.8(崩) | — |
| 裸跑对照(纯 AR,无投机解码,8K–100K) | 37.6–43.0 tok/s | — | 带宽上界推算 ~51 |

补充口径(同数据文件 `data/bench-tool-results.jsonl`,tag `dspine-final-*` / `裁决-诊断-*` / `ar-裸跑-*`):8K 真实语料单流 140.9;短代码 306 tok 441.8;修复后验证矩阵(bench_lctx2.py,code 语料)9229 tok:accept 14.6 / 218 tok/s,100K:accept 8.2 / 132 tok/s。注意:随机词表合成语料会系统性低估投机解码(接受率被压到 ~1.0,修复后 100K 随机词仍只有 46.0 tok/s)——评推测解码请用真实语料,详见 `docs/dspine-real-load-verdict.md` 第五节。

## 修复核心方法(三行)

1. 草稿 config 副本 `sliding_window: 8192 → 262144`(= `max_position_embeddings`,上下文 131072 内等价全注意力),权重 symlink 零拷贝,`--speculative-draft-model-path` 指向副本目录;
2. 插件 `sglang_dspine/layers/attention.py` 的 `WINDOW` 常量同步 `8192 → 262144`(与 config 副本配套,过其启动校验);
3. 根因实锤:sglang 0.5.22 + SM80(sgl-kernel fa3 路径)的 FA **滑窗掩码分支在 seq>窗口值时对该草稿输出垃圾**——崩塌边界精确等于窗口值(8192),窗口放大到 131072 后 8300~100000 全档恢复,边界随窗口值移动;窗口≥上下文后掩码分支永不激活,回到已验证正确的因果注意力路径,故无损。

真实 diff 与完整证据链见 [`patch/`](patch/) 与 [`docs/dspine-longctx-fix.md`](docs/dspine-longctx-fix.md)。已排除的假修复:`--speculative-draft-window-size`(DSPINE 硬拒)、`--chunked-prefill-size 16384`(OOM)、deterministic/num_splits=1(无效)。

## 256K 达成:第二层根因(YARN rope 外推)与终态

滑窗修好后把上下文拉到 262144(显存硬挤:`mem-frac 0.90 + mamba 8 槽 + chunked-prefill 1024 + skip-server-warmup`,KV 池 262688,前序降级链六次尝试见 [`docs/dspine-256k-test.md`](docs/dspine-256k-test.md)),256K 档接受率仍崩(0.27→0.037,decode 腰斩到 51.8)。第二层根因与修复:

1. **根因**:草稿 rope 训练仅覆盖 ~3072 位置,262144 输入下位置编码**全程 85 倍外推**,草稿与主模型分歧拉大、提案全废(窗口扫描证明不是滑窗问题:131072/65536 档 accept 均 1.025,262144 档 3.325,短窗口只会更崩);
2. **修复**:草稿 config 副本 `rope_parameters` 写 YARN——`{"rope_type": "yarn", "factor": 85.3333, "original_max_position_embeddings": 3072}`(factor = 262144/3072)。**坑:顶层 `rope_scaling` 字段是摆设**,sglang 0.5.22 的 `get_rope_config` 优先读 `rope_parameters`,yarn 写错地方静默不生效;
3. **效果**:256K 档 accept **5.075 / 0.272**(与 100K 甜点档完全同值),100K 档零回退(decode 还略升 106.8→112.0),131072 日常档与 256K 档共用同一草稿副本无冲突。

**256K 终态测速**(单流 261K realtext / 工具口径,终态配置全文 [`patch/launch_dspine_256k.sh`](patch/launch_dspine_256k.sh),diff 见 [`patch/yarn-draft-config.md`](patch/yarn-draft-config.md),报告 [`docs/dspine-256k-optimize.md`](docs/dspine-256k-optimize.md)):

| 口径 | prefill tps | decode tps | TTFT | 显存峰值 |
|---|---|---|---|---|
| realtext 261383 tok | 1133 | **74.3 / 82.6(两轮)** | 231 s | — |
| 工具随机词 261268 tok | 1127 | 37.2(无 yarn 时 23.2,+60%) | 232 s | 64611 MiB(剩 ~600 MiB) |

对照裸跑 AR 256K decode 36.7 tps:投机收益 ×2;输出质量抽检(261K 输入 512 token 总结)结构完整无乱码,样本见 [`experiments/kvqual-yarn256k.json`](experiments/kvqual-yarn256k.json)。

**KV 量化路线在此栈上是架构性死路,如实放弃**:DSpine SPS 验证硬要求主后端带 ragged 旗标,而 SM80 上唯一满足的后端 fa3 只吃 bf16/fp16 KV——**ragged 旗标 × Ampere FA dtype 交集为空**,与档位无关,升级版本绕不开。裸栈(无投机)fp8 KV 三档对比已留档(池 3.5×、e4m3 精度几乎无损),待 Hopper 硬件或上游补 ragged 后端后可直接选 e4m3(死因链与三档数据:[`docs/dspine-256k-optimize.md`](docs/dspine-256k-optimize.md) A 节,原始四枪日志 [`experiments/`](experiments/))。

## SPS 调优(负结果,已回退)

修完长上下文后,按 DSpine 官方 `tune_sps.py` 流程把 SPS 验证宽度表完整调了一遍(约 3 小时全程,负结果如实收录,同硬件同学可免重试):

- **结论:三档均无提升,已回退**。调出的表对三档真实负载 decode 提升全部 <5%——单流 8K **-2.0%**、单流 100K **+3.3%**、4×56K 纯排程噪声(该档 decode batch=3 时选中策略=宽 16,与无表完全等价)。按判定规则(任一档 ≥+5% 才保留)回退,终态恢复无表原配置,health 200。
- **原因分析**:本机 mamba 16 槽 → 实际并发上限 3,调优只能扫 `--batch-sizes 1 2 3` 的窄区间(官方默认 1–8 在本部署直接报错,bs≥4 生产也到不了);且本负载草稿接受长度本就 ~3.9,16 宽整块验证浪费不大,收窄无肉——官方甜区(小宽度高收益)在这台机不存在。另:310K token 校准语料池超 KV 缓存容量,测量墙钟被 prefill 主导,宽度间差异进一步被稀释。
- **适用条件**:换更大显存卡(mamba 槽↑、并发>3)或负载形态改变(温度采样为主、语料明显变化)后,可按同流程重调再试。
- 完整报告:[`docs/dspine-sps-tuning.md`](docs/dspine-sps-tuning.md);全部原始产物(SPS 表 JSON、校准语料、逐候选测量原始数据、服务日志、前后 bench):[`tuning/`](tuning/),前测速对照行已并入 [`data/bench-tool-results.jsonl`](data/bench-tool-results.jsonl)(tag `sps-前-*` / `sps-后-*`)。

## 环境

**硬件**:NVIDIA CMP 170HX(解锁 64G 显存)· GA100/SM 8.0 · HBM2e ~1493 GB/s · Xeon E5-2686 v4 · Ubuntu 24.04 · 驱动 610.43.03

**版本锁(DSpine 插件 pyproject 严格 `==`,一项不能偏)**:

```
sglang==0.5.22        torch==2.14.1        flashinfer-python==0.7.1
sglang-kernel==0.4.9  sglang-dspine==0.2.0(Tencent/DSpine editable 安装)
transformers==5.19.0  triton==3.8.0        Python 3.12
```

**模型**:底模 Qwen3.8-27B-FP8(28.97G)+ 草稿 tencent/Qwen3.8-27B-DSpine-b16(4.05G,5 层×sliding_attention,b16=每步并行起草 16 token)

**部署五关**(完整踩坑记录见 `docs/dspine-deploy-log.md`):① ninja 不在 PATH(JIT 编译崩)→ 补 export;② 草稿滑窗注意力后端插件硬锁字面 `fa3` → 实测 sglang 0.5.22 的 fa3 在 Ampere 可走通(sglang-kernel 包装,非裸 FA3);③ 插件限 8 并发 → 补 `--max-running-requests 8` 三件套;④ 草稿 KV 池必须 BF16(量化池被校验拒)→ `--speculative-draft-kv-cache-dtype bfloat16`;⑤ FP8 粗检索 `_scaled_mm` 无 SM80 内核 → 打 `patch/0002` BF16 回退;另 Ampere FA 不吃 fp8 KV → 底模 KV 退 bf16(最终启动参数全文见 `docs/dspine-deploy-log.md` §6 与 `docs/dspine-longctx-fix.md` §3)。

## 目录

| 路径 | 内容 |
|---|---|
| [`docs/`](docs/) | 14 份报告原文(时间序):调研方案 → 模型下载验收 → 部署日志(五关) → DFlash2 基线测速 → DSpine 修复前测速(崩盘) → 裸跑 AR 对照 → 真实负载数据 → 五负载裁决(verdict2) → 外部调研(根因 Top3+官方口径) → **长上下文修复报告** → 测速工具填法 → **SPS 调优(负结果)** → **256K 极限首测(降级链)** → **256K 优化(YARN 修复+量化死因)** |
| [`experiments/`](experiments/) | 256K 优化 16 件原始实验日志:A 轮 KV 量化四枪死因日志 + 裸栈三档对比与质量样本(`kvqual-*.json`)· B 轮滑窗扫描 · C 轮 YARN 服务日志 · D 轮显存采样 · `README.md` 逐件索引 |
| [`tuning/`](tuning/) | SPS 调优全部原始产物(负结果):最终表 `dspine-sps-table.json` · 校准语料与构建脚本 · 调优器日志 · 逐候选测量原始数据(`runs/`)· 带表/终态无表服务日志 · 前后对比 bench 与 metrics 快照 |
| [`patch/`](patch/) | 四件:两枚真实 diff(长上下文滑窗修复 · SM80 FP8 粗检索 BF16 回退)+ **256K YARN 草稿 config 增量(`yarn-draft-config.md`)** + **256K 终态启动脚本(`launch_dspine_256k.sh`)** |
| [`screenshots/`](screenshots/) | `final-benchmark-user.png` 用户实测截图 · `tool-howto-fullpage.png` / `tool-howto-closeup.png` 网页测速工具跑通与填法 |
| [`data/bench-tool-results.jsonl`](data/bench-tool-results.jsonl) | 全程 106 行原始测速数据(DFlash2 基线 / DSpine 修复前后 / 裸跑 AR / 裁决 / 最终验证 / SPS 调优前后 / 256K 首测 / 256K 优化 B·C·D 轮,tag 分段) |

## 已知限制(如实)

- **FP8 底模偏离官方 BF16 口径**:官方 benchmark 全部 BF16+FA3+未知 GPU(3.33× 不可平移绝对值);本机 FP8-on-SM80 走 Marlin 反量化路径。实测短上下文接受率 7.28–12.28 已达官方量级(GSM8K T0 官方 7.20)。
- **SM80 无真 FA3**:插件锁字面 `fa3`,sglang-kernel 的 fa3 后端在 Ampere 走包装内核(非 Hopper-only 的裸 FA3),属官方未测试组合;正因此踩到滑窗掩码分支的坑(也是本仓库核心发现)。
- **mamba 16 槽钳并发**:官方 64 槽在 64G 卡装不下(mamba fp32×16 槽被 ReplaySSM 按每请求 5 槽吃),实际并发上限 3(非 8);bf16 KV 使 token 预算 174,870、上下文显式压到 131072。256K 档换 mamba 8 槽 + mf 0.90 把预算抬到 262,688,但并发上限降为 1(单流专用档)。
- 接受率对语料敏感:code 12–16 / 数学 7.5 / 英文 realtext 4–5 / 中文叙事 ~2——创作型负载投机收益有限,负载画像决定选型。

## 复现

所有数字均为 2026-11-09/10 单机实测,原始数据、启动参数全文、验证方法与脚本口径都在本仓库;`patch/` 可直接应用。同硬件同学欢迎按 `docs/` 时间线复跑对照,对上游(sgl-project/sglang、Tencent/DSpine)报 issue 时可直接引用 `data/`、`experiments/` 与 `patch/` 的证据链(滑窗:边界=窗口值、窗口放大即愈、num_splits=1 无效;256K:rope 外推=YARN 可修、`rope_scaling` 顶层字段不被读、KV 量化 ragged×Ampere dtype 交集为空)。SPS 调优流程与判定口径见 [`tuning/README.md`](tuning/README.md),256K 实验逐件索引见 [`experiments/README.md`](experiments/README.md)。
