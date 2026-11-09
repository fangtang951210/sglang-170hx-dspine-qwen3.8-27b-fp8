# 170hx sglang 裸跑(纯 AR,无投机解码)单流 decode 验证实验报告

- 工单:验证推算「CMP 170HX 单流 decode 物理上限 ≈51 tps(1493GB/s 显存带宽 ÷ 29GB FP8 权重)」——若实测显著高于 51 则推算作废
- 日期:2026-11-09 16:09–16:23 CST · 主机 170hx· sglang 0.5.22(`~/sglang-llm-venv`)
- 流程:停手动 DSPINE 服务 → 起 AR 服务(16:09:45 启动,16:12:19 health 200,约 2.5 分钟)→ 4 轮测速 → 停 AR → 恢复现役 `sglang-llm`(DFlash2)
- 测速口径:与基线完全同构——`~/deploy/bench_tool.py`,随机词表+固定结尾+唯一前缀防 prefix cache,enable_thinking=false,temperature 0,真实 usage 计速;单流 output 512;档间 sleep 10;结果已追加 `~/deploy/bench-tool-results.jsonl`(tag 前缀 `ar-裸跑-`)

## 一、结论(先说判定)

**51 tps 推算成立(未作废)。** 纯 AR 单流 decode 实测 **37.6–43.0 tps**,4 轮全部显著低于 51 上界,一次都没有越线。用户预设的作废条件(实测显著高于 51)未触发。

| 长度 | AR 实测 decode tps | ÷51 上界 | 说明 |
|---|---|---|---|
| 8K | 43.0 | 84% | KV 读取占比小,最接近纯权重带宽 |
| 56K | 40.2 | 79% | 随上下文增长下滑 |
| 100K | 37.9 | 74% | 两轮复测一致 |
| 100K 复测 | 37.6 | 74% | 极差仅 0.3 tps,稳态 |

实测效率 = 上界的 74–84%,缺口来自:KV cache 读取(随长度增长,解释 43→38 的下滑)、lm_head/大词表、采样与 kernel 开销、GDN/线性注意力状态读写。**"每 token 全量读一遍 29GB 权重"的带宽瓶颈模型与实测自洽。**

## 二、AR 配置参数全文

`~/deploy/launch_ar.sh`(底本 `~/deploy/launch_user.sh`,删除全部 `--speculative-*` 三项与 `--mamba-cache` 槽数,mamba cache 参数按 hybrid GDN 模型所需保留、槽数回默认),运行 cmdline 逐字核对一致:

```
/home/<user>/sglang-llm-venv/bin/python -m sglang.launch_server
  --model-path /mnt/models/Qwen3.8-27B-FP8
  --served-model-name qwen3.8-27b
  --host 0.0.0.0 --port 30020
  --trust-remote-code
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.85
  --attention-backend flashinfer
  --chunked-prefill-size 2048
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --mamba-ssm-dtype bfloat16
  --mamba-radix-cache-strategy extra_buffer
  --enable-metrics
  --enable-cache-report
  --allow-auto-truncate
  --log-level info
```

相对底本删掉的 4 个参数:`--speculative-algorithm DFLASH`、`--speculative-draft-model-path`、`--speculative-num-draft-tokens 8`、`--max-mamba-cache-size 12`(槽数回默认)。

运行时核验(启动日志):`speculative_draft_model_path=None`;mamba 槽数默认 **157**(每请求 5 状态槽,`max_running_requests` 自动封 31);KV 池 fp8_e4m3 共 **425,244 tokens**(K+V 12.98GB);decode CUDA 图捕完(含 bs=1),图捕后可用 7.74GB。注:venv 内装有 sglang-dspine 插件(日志有 AROUND hook 注册行),但未启用投机,不影响 AR 路径。

## 三、4 轮测速数据(单流,output 512,temperature 0)

| 轮 | tag | prompt tokens | prefill tps | TTFT | decode tps |
|---|---|---|---|---|---|
| 1 | ar-裸跑-1s-8k | 8113 | 1860.6 | 4.36s | **43.0** |
| 2 | ar-裸跑-1s-56k | 56463 | 1655.3 | 34.11s | **40.2** |
| 3 | ar-裸跑-1s-100k | 100799 | 1359.4 | 74.15s | **37.9** |
| 4 | ar-裸跑-1s-100k-r2 | 100799 | 1353.9 | 74.45s | **37.6** |

completion tokens:422/447/512/512(temperature 0 偶发提前 EOS,decode tps 一律按真实 usage 计,口径与基线一致)。prefill 速度与 DFlash2 基线同档(投机不加速 prefill,符合预期)。

nvidia-smi 采样(全程 2s 间隔,`/tmp/ar-nvsmi.csv`,136 行):

| 轮 | 显存峰值 | util 峰值 | 功耗峰值 | 温度峰值 |
|---|---|---|---|---|
| 8K | 57637 MiB | 100% | 299.4 W | 47℃ |
| 56K | 57637 MiB | 100% | 350.7 W | 61℃ |
| 100K | 57639 MiB | 100% | 365.1 W | 69℃ |
| 100K-r2 | 57641 MiB | 100% | 372.4 W | 72℃ |

显存平稳无增长;功耗/温度形态与 DFlash2 基线相同(峰值 372W vs 基线 378W)。

## 四、三方对照:AR 裸跑 vs DFlash2 基线 vs 51 tps 推算

DFlash2 基线取自 `/Users/sugar/Documents/tmp/dspine-speedtest-baseline.md`(2026-11-09 上午,0.5.20 栈,同工具同口径):

| 长度 | ① AR 裸跑 | ② 51 tps 推算(带宽上界) | ③ DFlash2 实测 | ③÷① 实测加速 | ③ 越过 51? |
|---|---|---|---|---|---|
| 8K | 43.0 | 51 | 82.2 | ×1.91 | 是(+61%) |
| 56K | 40.2 | 51 | 74.7 | ×1.86 | 是(+46%) |
| 100K | 37.9 / 37.6 | 51 | 78.1 / 80.9 / 78.0 | ×2.06 | 是(+53%) |

三方串起来的完整逻辑:

1. **AR 从未越过 51**——带宽上界成立。每 decode step 必须全量读 29GB 权重,29GB ÷ 1493GB/s ≈ 51.5 tps,实测最高 43(84%),越不过去。
2. **DFlash2 稳定 75–83,反而越过了 51**——不矛盾,恰恰是投机解码的意义:每次目标模型前向平均收下 2.75 token(基线 spec_accept_length=2.75,accept rate 0.25),相当于把一次权重读取摊到多个 token 上,理论上限 ≈51×2.75≈141 tps,实测 78–83(实现 55–59%),草稿模型前向+验证开销吃掉差额。
3. **实测加速比 ×1.86–2.06 与接受率 2.75 自洽**——若 AR 真能跑到 100+(即推算作废的情形),DFlash2 的 78–83 反而无法解释。三方互相咬合,推算成立。

## 五、最终服务状态(已恢复现役)

- 实验服务已停(GPU 清零、端口释放),`~/deploy/launch_ar.sh` 保留可复用
- **现役 `sglang-llm` 已恢复**:16:19:57 systemctl start,16:23:25 health 200(约 3.5 分钟);unit active,cmdline 核验 `speculative-algorithm DFLASH`、`num-draft-tokens 8`、`port 30020`、`mem-fraction-static 0.85`,显存 57597 MiB——0.5.22 栈 DFlash2 正常起服,未触发"起不来"兜底分支
- 实验前状态:手动 DSPINE 栈(PID 42258,`/tmp/dspine-launch-8.log`)已按精确 PID SIGTERM 优雅停止;systemd unit 的 "failed" 为 14:26 旧停服残留,现已归位 active
- 170hx 残留物(可删):`/tmp/ar-launch.log`、`/tmp/ar-nvsmi.csv`、`/tmp/ar-rounds.txt`

## 六、raw JSONL 原样引用(源:170hx:~/deploy/bench-tool-results.jsonl 末 4 行)

```json
{"tag": "ar-裸跑-1s-8k", "words": 8000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 8113, "prefill_tps_avg": 1860.6, "decode_tps_each": [43.0], "decode_tps_avg": 43.0, "decode_tps_total": 43.0, "ttft_ms_max": 4360.3, "detail": [{"seed": 1, "prompt_tokens": 8113, "completion_tokens": 422, "prefill_ms": 4360.3, "decode_ms": 9818.1, "prefill_tps": 1860.6, "decode_tps": 43.0, "ttft_ms": 4360.3}]}
{"tag": "ar-裸跑-1s-56k", "words": 56000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 56463, "prefill_tps_avg": 1655.3, "decode_tps_each": [40.2], "decode_tps_avg": 40.2, "decode_tps_total": 40.2, "ttft_ms_max": 34110.1, "detail": [{"seed": 1, "prompt_tokens": 56463, "completion_tokens": 447, "prefill_ms": 34110.1, "decode_ms": 11109.5, "prefill_tps": 1655.3, "decode_tps": 40.2, "ttft_ms": 34110.1}]}
{"tag": "ar-裸跑-1s-100k", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1359.4, "decode_tps_each": [37.9], "decode_tps_avg": 37.9, "decode_tps_total": 37.9, "ttft_ms_max": 74149.0, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 74149.0, "decode_ms": 13514.6, "prefill_tps": 1359.4, "decode_tps": 37.9, "ttft_ms": 74149.0}]}
{"tag": "ar-裸跑-1s-100k-r2", "words": 100000, "concurrency": 1, "output_len": 512, "prompt_tokens_avg": 100799, "prefill_tps_avg": 1353.9, "decode_tps_each": [37.6], "decode_tps_avg": 37.6, "decode_tps_total": 37.6, "ttft_ms_max": 74451.2, "detail": [{"seed": 1, "prompt_tokens": 100799, "completion_tokens": 512, "prefill_ms": 74451.2, "decode_ms": 13611.0, "prefill_tps": 1353.9, "decode_tps": 37.6, "ttft_ms": 74451.2}]}
```

*报告完。只测不改配置;现役服务已恢复并核验;DFlash2 在 0.5.22 下正常起服。*
