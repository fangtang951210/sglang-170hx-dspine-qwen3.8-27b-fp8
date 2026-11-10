# 256K 优化实验原始日志(2026-11-10,170hx 单机实测)

本目录是 [`../docs/dspine-256k-optimize.md`](../docs/dspine-256k-optimize.md) 报告的原始证据,16 个文件对应实验矩阵 A/B/C/D 四轮,一轮一变因。同日的前序(256K 极限首测、降级链六次尝试)见 [`../docs/dspine-256k-test.md`](../docs/dspine-256k-test.md)。测速数值行已并入 [`../data/bench-tool-results.jsonl`](../data/bench-tool-results.jsonl)(tag 见下表)。日志内 traceback 路径已统一脱敏为 `~/`(原绝对路径含用户名)。

## A 轮:KV FP8 化四枪全死(DSpine 栈)

| 文件 | 配置(server_args 实录) | 结局 |
|---|---|---|
| `gun-a1.log` | DSpine + fa3 + `fp8_e4m3` | ✗ `FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type`(kernel 级) |
| `gun-a1b.log` | DSpine + `fp8_e5m2`(启动配 fa3,sglang 自动警告降级 triton,server_args 已是 triton) | ✗ 主后端变 triton → `ValueError: The configured attention backend does not support ragged graphs` |
| `gun-a2.log` | DSpine + flashinfer + `fp8_e4m3` | ✗ 同上 ragged 校验拒(DSpine SPS 验证要求主后端带 ragged 旗标) |
| (代码核实) | trtllm_mha | ✗ prefill/decode kernel 仅 SM90/SM120,SM80 无;未起服务故无日志 |

→ **死因链:ragged 旗标硬锁 × Ampere FA dtype 硬锁,交集为空**,与档位无关,升级版本绕不开(除非上游新增带 ragged 旗标的 SM80 后端)。

## A 轮附:裸 sglang(无投机)三档对比

| 文件 | KV 档 | 质量样本 |
|---|---|---|
| `gun-bare-bf16.log` | bf16 基线(`kv_cache_dtype: auto`) | `kvqual-bf16.json` |
| `gun-bare.log` | `fp8_e4m3`(池 3.5×,精度几乎无损) | `kvqual-e4m3.json` |
| `gun-bare-e5m2.log` | `fp8_e5m2`(措辞漂移明显多于 e4m3 一个数量级) | `kvqual-e5m2.json` |

三枪均为 flashinfer 后端、100K 档、T=0 同种子,质量样本为同 prompt 512 token 输出全文,可直接 diff 对比精度。结论:e4m3 几乎无损(仅 2 处同义措辞),未来栈解锁(Hopper 或上游补 ragged)后是唯一候选档。

## B 轮:草稿滑窗扫描 @256K 输入(短窗口全崩)

底盘 = `launch_dspine_256k.sh` 原版,realtext 261383 tok 单流,改窗口 = 草稿 config `sliding_window` + 插件 `WINDOW` 同步:

| 文件 | 窗口 | accept_length | decode tps |
|---|---|---|---|
| `b-win262144.log` | 262144(对照) | 3.325 | 52.0 |
| `b-win131072.log` | 131072 | 1.025 | 17.3 |
| `b-win65536.log` | 65536 | 1.025 | 17.9 |
| `b-win32768.log` | ⚠️ **诚实标注:循环 sed pattern 未跟随更新,该轮实际跑的仍是 65536(与 b-win65536 复测一致);32768 未实际测** | 1.025 | 17.9 |

→ seq>window 即激活 SM80 fa3 滑窗掩码坏分支(0001 定位的 bug),窗口必须 ≥ 输入全长。

## C 轮:YARN rope 外推(根因命中)

| 文件 | 内容 |
|---|---|
| `c-yarn3072.log` | 草稿 config `rope_parameters` 加 yarn(factor 85.3333, orig 3072)后完整服务日志,含 @256K(accept 5.075 / decode 74.3)与 @100K 副作用检查(~4.98 / 112.0)两轮,尾部有逐 batch accept len/rate 实时行 |
| 质量样本 | `kvqual-yarn256k.json`:yarn 256K 终态 261371 tok 输入的 512 token 总结输出全文(结构完整、忠实覆盖输入五大主题、无乱码) |

## D 轮:组合最优完整测速(256K 终态)

| 文件 | 内容 |
|---|---|
| `d-tool-gpu.log` | 工具口径测速全程显存采样(2s 粒度,每行一值 MiB),峰值 64611 |

realtext 口径两轮(74.3 / 82.6 tps)与工具口径(37.2,较无 yarn 的 23.2 +60%)数值见 jsonl tag `d-yarn256k-tool` 与报告 D 节。

## 数值对照表(jsonl tag)

| tag | 轮 | 数据 |
|---|---|---|
| `dspine-256k-*`(2 行) | 前序(256k-test) | 无 yarn 终态两口径测速 |
| `b-win*`(4 行) | B | 窗口扫描 |
| `c-yarn3072` / `c-yarn3072-100k` | C | yarn 256K / 100K 副作用 |
| `d-yarn256k-tool` | D | yarn 256K 终态工具口径 |
