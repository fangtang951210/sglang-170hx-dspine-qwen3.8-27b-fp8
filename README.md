# sglang + DSpine(CMP 170HX / SM80 / FP8)复现 Qwen3.8-27B 投机解码全记录:部署踩坑 → 测速 → 发现长上下文滑窗掩码 bug → 修复 → 数据

> 关键词:sglang · DSpine · speculative decoding · CMP 170HX · SM80 / Ampere · FP8 · Qwen3.8-27B · long context · sliding window bug

一张矿卡(NVIDIA CMP 170HX,SM80/Ampere,64G 显存)上用 **FP8 底模**复现腾讯 DSpine 并行起草投机解码的完整实战记录:sglang 0.5.20→0.5.22 严格升级 → 八枪闯过五关部署成功 → 测速发现 decode 全线崩到裸跑以下 → 五负载裁决+外部调研锁定方向 → **定位到 sglang 0.5.22 + SM80 上 FA 滑窗掩码分支输出垃圾的确定性 bug(边界=草稿 config 的 `sliding_window: 8192`)→ 两处同步改 262144 修复,长上下文全档恢复**。全部数据同机同工具实测,含原始 JSONL 与真实 diff,欢迎复现。

## 最终测速(修复后,单机实测)

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
| [`docs/`](docs/) | 11 份报告原文(时间序):调研方案 → 模型下载验收 → 部署日志(五关) → DFlash2 基线测速 → DSpine 修复前测速(崩盘) → 裸跑 AR 对照 → 真实负载数据 → 五负载裁决(verdict2) → 外部调研(根因 Top3+官方口径) → **长上下文修复报告** → 测速工具填法 |
| [`patch/`](patch/) | 两枚真实 diff:长上下文滑窗修复(config+WINDOW 常量)· SM80 FP8 粗检索 BF16 回退 |
| [`screenshots/`](screenshots/) | `final-benchmark-user.png` 用户实测截图 · `tool-howto-fullpage.png` / `tool-howto-closeup.png` 网页测速工具跑通与填法 |
| [`data/bench-tool-results.jsonl`](data/bench-tool-results.jsonl) | 全程 91 行原始测速数据(DFlash2 基线 / DSpine 修复前后 / 裸跑 AR / 裁决 / 最终验证,tag 分段) |

## 已知限制(如实)

- **FP8 底模偏离官方 BF16 口径**:官方 benchmark 全部 BF16+FA3+未知 GPU(3.33× 不可平移绝对值);本机 FP8-on-SM80 走 Marlin 反量化路径。实测短上下文接受率 7.28–12.28 已达官方量级(GSM8K T0 官方 7.20)。
- **SM80 无真 FA3**:插件锁字面 `fa3`,sglang-kernel 的 fa3 后端在 Ampere 走包装内核(非 Hopper-only 的裸 FA3),属官方未测试组合;正因此踩到滑窗掩码分支的坑(也是本仓库核心发现)。
- **mamba 16 槽钳并发**:官方 64 槽在 64G 卡装不下(mamba fp32×16 槽被 ReplaySSM 按每请求 5 槽吃),实际并发上限 3(非 8);bf16 KV 使 token 预算 174,870、上下文显式压到 131072。
- 接受率对语料敏感:code 12–16 / 数学 7.5 / 英文 realtext 4–5 / 中文叙事 ~2——创作型负载投机收益有限,负载画像决定选型。

## 复现

所有数字均为 2026-11-09 单机实测,原始数据、启动参数全文、验证方法与脚本口径都在本仓库;`patch/` 可直接应用。同硬件同学欢迎按 `docs/` 时间线复跑对照,对上游(sgl-project/sglang、Tencent/DSpine)报 issue 时可直接引用 `data/` 与 `patch/` 的证据链(边界=窗口值、窗口放大即愈、num_splits=1 无效)。
