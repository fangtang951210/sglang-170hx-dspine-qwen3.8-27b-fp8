#!/usr/bin/env python3
# 用途:生成 DSpine SPS 调优器输入 prefixes JSONL——真实语料(代码+文章混合,8 条不同长度/任务/推理深度,全部贪心 temperature=0)
# 归属:2026-11-09 DSpine SPS 调优实战(170hx,控制台派单)
# 用法:~/sglang-llm-venv/bin/python ~/deploy/build_sps_prefixes.py --output ~/deploy/sps-prefixes.jsonl
# 一次性/可复用:可复用(改 TARGETS 或语料基底重跑即可)
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_realtext import ARTICLE, SUFFIX  # 与测速口径同源语料

from transformers import AutoTokenizer

MODEL = "/mnt/models/Qwen3.8-27B-FP8"
MAX_CTX = 131072
GEN = 512

# (目标token数, 语料类型, 任务说明)——长度覆盖 8K/56K/100K 三档真实使用,任务覆盖总结/代码理解/找bug/续写/单测/中文总结
TARGETS = [
    (9000,   "art",  SUFFIX),                                                        # 对齐 bench 8K 档:文章+英文总结
    (9000,   "code", "\nExplain what this code does, its architecture, and any subtle issues."),
    (14000,  "mix",  "\n用中文详细总结上面的要点,并给出你的评价。"),                       # 中文任务=真实用户用法
    (20000,  "code", "\nWrite unit tests for the most important functions above."),
    (32000,  "mix",  "\nReview the material above carefully, find any bugs or inconsistencies, and explain them."),
    (48000,  "code", "\nContinue the implementation in the same style, completing the unfinished parts."),
    (64000,  "art",  SUFFIX),                                                        # 对齐 bench 56K 档
    (115000, "mix",  SUFFIX),                                                        # 对齐 bench 100K 档
]

def read_bases():
    readme = (Path.home() / "DSpine" / "README.md").read_text(errors="ignore")
    code_parts = [p.read_text(errors="ignore") for p in sorted((Path.home() / "DSpine" / "sglang_dspine").rglob("*.py"))]
    code = "\n\n".join(code_parts)
    return {"art": ARTICLE + "\n\n" + readme, "code": code}

def compose_words(base, n_words, kind):
    words = base.split()
    fill = (words * (n_words // len(words) + 2))[:n_words]
    if kind == "mix":  # 前半文章后半代码,中间再垫一段文章,贴近"读材料+看代码"的真实上下文
        half = n_words // 2
        art = (ARTICLE.split() * (half // len(ARTICLE.split()) + 2))[:half]
        code_words = base.split()
        cod = (code_words * (half // len(code_words) + 2))[:n_words - half]
        fill = art + [ARTICLE[:0]] + cod
        fill = [w for w in fill if w != ""]
    return " ".join(fill)

def to_ids(tok, content):
    enc = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False)
    if not isinstance(enc, (list, tuple)):
        try:
            enc = enc["input_ids"]
        except Exception:
            enc = list(enc.ids) if hasattr(enc, "ids") else list(enc)
    return [int(x) for x in enc]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="<user>/deploy/sps-prefixes.jsonl")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    bases = read_bases()
    rows, stats = [], []
    for i, (target, kind, suffix) in enumerate(TARGETS, 1):
        base = bases["art"] if kind == "art" else bases["code"]
        if kind == "mix":
            base = bases["code"]
        n_words = int(target * 1.05)
        body = compose_words(base, n_words, kind)
        content = body + "\n\n" + suffix
        ids = to_ids(tok, content)
        for _ in range(4):  # 按 token 精确修剪(±2% 内收工)
            over = len(ids) - target
            if abs(over) <= target * 0.02:
                break
            if over > 0:
                body_ids = tok(body)["input_ids"]
                keep = max(len(body_ids) - over - 16, target // 2)
                body = tok.decode(body_ids[:keep])
            else:
                n_words += max(int(-over * 1.15), 64)
                body = compose_words(base, n_words, kind)
            content = body + "\n\n" + suffix
            ids = to_ids(tok, content)
        assert len(ids) <= MAX_CTX - GEN - 200, f"row {i} too long: {len(ids)}"
        rows.append({"input_ids": ids, "sampling_params": {"temperature": 0.0}})
        stats.append({"row": i, "kind": kind, "tokens": len(ids), "task": suffix[:40]})
        print(f"row {i}: kind={kind} tokens={len(ids)} task={suffix[:40]!r}", flush=True)
    with open(a.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(s["tokens"] for s in stats)
    print(json.dumps({"output": a.output, "rows": len(rows), "total_tokens": total,
                      "temperatures": sorted({r["sampling_params"]["temperature"] for r in rows})}))
    Path(a.output + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
