#!/bin/bash
# DSpine 投测服务:sglang 0.5.22 + DSPINE 草稿 tencent/Qwen3.8-27B-DSpine-b16(2026-11-09 冒烟通过)
# 官方示例 × SM80 现实适配(与方案 §8 初版的差异,8 枪迭代定版,详见 dspine-deploy-log.md):
#   - 主/草稿 attention-backend 都必须 fa3(DSpine ragged 验证硬要求;SM80 走 sglang-kernel FA 路径可行)
#   - KV 不许 fp8(Ampere FA 只吃 bf16/fp16):底模随模型 bf16,草稿显式 bf16
#   - 必须带 --max-running-requests 8(插件上限);实际被 mamba 16 槽/请求5槽 压到并发 3
#   - mamba 64→16 槽 fp32、mem-frac 0.80、context 显式 131072
#   - 插件补丁:~/DSpine/sglang_dspine/layers/fp8/coarse.py SM80 粗检索 bf16 回退(原文件 .orig)
#   - 长上下文修复(2026-11-09):草稿滑窗 8192 在 seq>8192 时走 FA 滑窗掩码分支,SM80 上输出垃圾→接受率 1.0。
#     修复=禁用滑窗:草稿 config 副本 sliding_window 262144(=max_position_embeddings,131072 上下文内等价全注意力),
#     配套插件补丁 ~/DSpine/sglang_dspine/layers/attention.py WINDOW=262144。详见 dspine-longctx-fix 报告。
#   - 256K 接受率修复(2026-11-10,dspine-256k-optimize 报告):草稿 rope 训练仅覆盖 ~3072 位置,262144 全外推导致
#     接受率崩塌(3.3/52tps)。修复=草稿 config 副本 rope_parameters 加 yarn(factor 85.3333, orig 3072):
#     256K 档 accept 5.075/decode 74-83tps(=100K 甜点档水平),100K 档零回退。窗口必须=262144,任何更小窗口
#     在 seq>window 时触发 SM80 fa3 滑窗掩码坏分支(accept 1.025 实测)。
source "$HOME/deploy/cuda_env.sh"
export PATH="$HOME/sglang-llm-venv/bin:$PATH"
exec "$HOME/sglang-llm-venv/bin/python" -m sglang.launch_server \
  --model-path /mnt/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 30020 \
  --trust-remote-code \
  --speculative-algorithm DSPINE \
  --speculative-draft-model-path "$HOME"/deploy/qwen3.8-27b-dspine-b16-fullctx \
  --speculative-num-draft-tokens 16 \
  --attention-backend fa3 \
  --speculative-draft-attention-backend fa3 \
  --linear-attn-backend triton \
  --enable-linear-replayssm-spec \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 8 \
  --mem-fraction-static 0.90 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --context-length 262144 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1 \
  --chunked-prefill-size 1024 \
  --sampling-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --enable-cache-report --skip-server-warmup --allow-auto-truncate --log-level info
