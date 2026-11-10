# 0003 · 256K 草稿 YARN rope 外推修复(草稿 config 副本增量)

> 配套启动脚本全文:[`launch_dspine_256k.sh`](launch_dspine_256k.sh)(256K 终态,常驻可用)。完整证据链见 [`../docs/dspine-256k-optimize.md`](../docs/dspine-256k-optimize.md),原始实验日志见 [`../experiments/`](../experiments/)。

## 根因(一句话)

草稿(Qwen3.8-27B-DSpine-b16)的 rope 训练仅覆盖 ~3072 位置(`original_max_position_embeddings: 3072`),而 `max_position_embeddings: 262144` 意味着 256K 输入下草稿位置编码**全程 85 倍外推**——滑窗修好(0001)之后 256K 档接受率仍崩在 3.3(100K 档为 5.075),根因就是外推,不是滑窗。

## 改动全文(草稿 config 副本,一处对象)

在 [0001](0001-long-context-sliding-window-fix.patch) 的零拷贝副本目录(`--speculative-draft-model-path` 指向的那个 `config.json`)上叠加以下 diff,**净 diff 相对 tencent/Qwen3.8-27B-DSpine-b16 原版**:

```diff
--- a/qwen3.8-27b-dspine-b16/config.json        (原版 tencent/Qwen3.8-27B-DSpine-b16)
+++ b/qwen3.8-27b-dspine-b16-fullctx/config.json (部署副本,0001+0003 叠加终态)
@@ -86,10 +86,12 @@
   "rms_norm_eps": 1e-06,
   "rope_parameters": {
     "rope_theta": 10000000,
-    "rope_type": "default"
+    "rope_type": "yarn",
+    "factor": 85.3333,
+    "original_max_position_embeddings": 3072
   },
   "rope_scaling": null,
   "rope_theta": 10000000,
-  "sliding_window": 8192,
+  "sliding_window": 262144,
   "tie_word_embeddings": false,
```

两块改动的分工:

| 字段 | 归属 | 作用 |
|---|---|---|
| `sliding_window: 8192 → 262144` | **0001(前序)** | 禁用滑窗裁剪,绕开 SM80 fa3 滑窗掩码坏分支(seq>窗口即输出垃圾);窗口必须 ≥ 输入全长,任何更小窗口实测全崩(131072/65536 档 accept 均为 1.025) |
| `rope_parameters` 三行 | **0003(本补丁)** | YARN 插值修复长位置编码外推:`factor = 262144 / 3072 = 85.3333`,`original_max_position_embeddings = 3072`(草稿实际训练长度) |

`attn_factor` 未显式设置(=1.0),sglang / HF / Qwen 三方默认一致,无需写。

## 两个坑(踩过才知道)

1. **顶层 `rope_scaling` 字段是摆设**:草稿 config 顶层本来就有 `"rope_scaling": null`,看起来是写 yarn 的地方——但 sglang 0.5.22 的 `get_rope_config`(v5 分支)**优先读 `rope_parameters`**,顶层 `rope_scaling` 根本不被读。yarn 必须写进 `rope_parameters` 对象,写错地方静默不生效。草稿 rope 实现在 sglang 主仓 `models/dflash.py`,走标准 `get_rope(..., rope_scaling=...)`,yarn 链路完整可用。
2. **改窗口的脚本若用 sed 连环替换,pattern 要跟随上一轮值**:否则静默不生效(本次 b-win32768 标签的教训,该轮数据实为 65536 复测,见 experiments/README.md 诚实标注)。

## 效果(256K 输入,realtext 261383 tok 单流,512 出)

| 轮 | 配置 | accept_length | accept_rate | decode tps |
|---|---|---|---|---|
| 滑窗修复后对照 | 0001(无 yarn) | 3.325 | 0.155 | 52.0 |
| **+ YARN(本补丁)** | 0001+0003 | **5.075** | **0.272** | **74.3–82.6** |
| 100K 档(副作用检查) | 同上,@100K | ~4.98 | ~0.27 | 112.0(原 106.8) |

- 256K 接受率恢复到 **100K 甜点档完全同值**(前次报告 100K 档即 5.075/0.272),外推假设实锤。
- 100K 档零回退(decode 还略升),131072 日常档与 256K 档可共用同一草稿副本。
- 对照裸跑 AR(无投机)256K decode 36.7 tps:投机收益 ×2。

## 与 0001 的关系

0001 解决"草稿在 seq>窗口时输出垃圾"(内核级 bug 绕开),0003 解决"草稿在超长位置上外推失准"(模型训练特性补偿)。两者独立必要:只打 0003 不打 0001,窗口仍 8192,seq>8192 即触发滑窗坏分支(对照 B 轮:窗口必须 ≥ 输入全长,131072/65536 在 256K 输入下 accept 均 1.025);只打 0001 不打 0003,256K 档 accept 卡 3.3。**256K 完整配置 = 0002(SM80 粗检索)→ 0001(滑窗)→ 0003(YARN)+ 启动脚本 `launch_dspine_256k.sh`。**
