"""检索侧评测：规模 vs 检索质量（**零 API 成本**）。

为什么需要这个脚本
------------------
原 W4 的设计是「全量注入 × 三档语料」，观测 lost-in-the-middle。
2026-10-28 查明两条硬约束后该设计不可行：

1. 账户报错原文为 `AllocationQuota.FreeTierOnly` —— 免费额度用尽即 403，
   **不可能产生账单，但也补不了额度**；
2. 346 篇档单题输入 811K token，而 qwen3.7-plus 每次调用还要额外烧
   ~514 个 reasoning token。三档 × 52 题在免费额度下**跑不完**。

故改为测**检索侧**：候选池规模上升时，检索能否把金标篇目排进前 k。
它测的不是「模型在长上下文里走神」，而是「索引被干扰」——机制不同，
但对 RAG 系统更贴近线上真实路径（线上永远是先检索再注入）。

指标
----
篇级（doc）：top-k 篇里是否包含 gold_sources 中的篇目
块级（chunk）：top-k 块的来源篇是否包含 gold —— 更严格，真实 RAG 检的是块
MRR：第一个金标出现的位次倒数，能反映「排第一」与「排第五」的差别

用法
----
    python tools/eval_retrieval.py                    # 三档全跑，排除留出集
    python tools/eval_retrieval.py --tiers 45 346     # 只跑两档
    python tools/eval_retrieval.py --out xxx.md
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from rag.bm25 import BM25                                   # noqa: E402
from rag.chunk import split_chunks, split_sentences         # noqa: E402

CASES = os.path.join(PROJECT, "eval", "cases.jsonl")

# 三档语料。144 档（corpus_clean）是 346 档（external_clean）的子集，
# 保持「严格嵌套」，否则档位之间的差异无法归因到规模。
TIERS = {
    "45": [os.path.join(PROJECT, "data", "docs")],
    # 100 档 = 45 篇真语料 + 55 篇**同域**噪声（提到水果品种的行业文）。
    # 与 144/346 档的区别：那两档是**异构**噪声（食品法规/检出通报），
    # 与水果客服知识的 token 几乎不重叠，因此不构成检索干扰。
    # 要触发衰减，必须用同域噪声 —— 这是本实验的核心设计。
    "100": [os.path.join(PROJECT, "data", "docs"),
            os.path.join(PROJECT, "data", "raw", "fruit_noise")],
    "144": [os.path.join(PROJECT, "data", "docs"),
            os.path.join(PROJECT, "data", "raw", "corpus_clean")],
    "346": [os.path.join(PROJECT, "data", "docs"),
            os.path.join(PROJECT, "data", "raw", "external_clean")],
}


def load_cases(allow_holdout=False):
    cases = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    if not allow_holdout:
        kept = [c for c in cases if not c.get("holdout")]
        dropped = sorted(c["id"] for c in cases if c.get("holdout"))
        print(f"⚠️ 默认排除留出集 {len(dropped)} 题 {dropped}，本次只跑 {len(kept)} 题")
        print("   （调参期间必须排除；带 --allow-holdout 才跑全集）")
        cases = kept
    return cases


def load_docs(dirs):
    """返回 [(doc_id, text)]。doc_id 用文件名（与 gold_sources 口径一致）。"""
    out, seen = [], set()
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.md"))):
            name = os.path.basename(p)
            if name in seen:      # 跨目录重名保护：外部语料已去重，但别赌
                continue
            seen.add(name)
            with open(p, encoding="utf-8") as f:
                out.append((name, f.read()))
    return out


def build_chunks(docs):
    """切块并保留来源篇名。块 id 形如 `文件名#3`。"""
    chunks = []
    for name, text in docs:
        for i, c in enumerate(split_chunks(split_sentences(text), name)):
            chunks.append((f"{name}#{i}", c["text"], name))
    return chunks


def eval_tier(cases, docs, topk=(1, 3, 5, 10)):
    doc_idx = BM25(docs)
    chunks = build_chunks(docs)
    chunk_idx = BM25([(cid, txt) for cid, txt, _ in chunks])
    chunk_src = {cid: src for cid, _, src in chunks}

    max_k = max(topk)
    rows = []
    for c in cases:
        gold = set(c.get("gold_sources") or [])
        if not gold:
            continue          # 库外题（none/constraint）无金标，不参与 recall

        q = c["q"] + (" " + " ".join(h for h in (c.get("history") or [])) if c.get("history") else "")

        d_hits = [d for d, _ in doc_idx.search(q, top_k=max_k)]
        d_rank = next((i + 1 for i, d in enumerate(d_hits) if d in gold), None)

        c_hits = [chunk_src[cid] for cid, _ in chunk_idx.search(q, top_k=max_k)]
        c_rank = next((i + 1 for i, s in enumerate(c_hits) if s in gold), None)

        rows.append({"id": c["id"], "q": c["q"], "category": c.get("category"),
                     "kind": c.get("kind"), "n_gold": len(gold),
                     "doc_rank": d_rank, "chunk_rank": c_rank,
                     "doc_top": d_hits[:5], "chunk_top": c_hits[:5]})

    def agg(key):
        res = {}
        for k in topk:
            hit = sum(1 for r in rows if r[key] and r[key] <= k)
            res[f"R@{k}"] = hit / len(rows) if rows else 0
        res["MRR"] = statistics.mean([1 / r[key] if r[key] else 0 for r in rows]) if rows else 0
        res["miss"] = sum(1 for r in rows if not r[key])
        return res

    return {
        "n_docs": len(docs), "n_chunks": len(chunks), "n_cases": len(rows),
        "doc": agg("doc_rank"), "chunk": agg("chunk_rank"), "rows": rows,
    }


def pct(x):
    return f"{x * 100:.1f}%"


def build_report(results, order):
    L = ["# 检索侧规模评测（零 API 成本）\n"]
    L.append("> 方法：本地 BM25（`rag/bm25.py`，纯 Python + 字符 bigram，无新依赖）。")
    L.append("> 语料三档严格嵌套：45 ⊂ 144 ⊂ 346。")
    L.append("> 只统计有 `gold_sources` 的题；库外题（none / constraint）不参与 recall。")
    L.append("> 默认已排除留出集，符合「调参期不看留出集」的纪律。\n")

    L.append("## 一、总表\n")
    L.append("| 档位 | 篇数 | 块数 | 题数 | 篇级 R@1 | 篇级 R@3 | 篇级 R@5 | 篇级 MRR | 块级 R@1 | 块级 R@3 | 块级 R@5 | 块级 MRR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in order:
        r = results[t]
        d, ch = r["doc"], r["chunk"]
        L.append(f"| {t} 篇 | {r['n_docs']} | {r['n_chunks']} | {r['n_cases']} | "
                 f"{pct(d['R@1'])} | {pct(d['R@3'])} | {pct(d['R@5'])} | {d['MRR']:.3f} | "
                 f"{pct(ch['R@1'])} | {pct(ch['R@3'])} | {pct(ch['R@5'])} | {ch['MRR']:.3f} |")
    L.append("")

    L.append("## 二、跨档位变化（衰减的直接证据）\n")
    base = order[0]
    L.append(f"以 {base} 篇档为基线，看规模上去后指标怎么动：\n")
    L.append("| 档位 | 篇级 R@5 变化 | 篇级 MRR 变化 | 块级 R@5 变化 | 块级 MRR 变化 | 未召回题数（块级） |")
    L.append("|---|---|---|---|---|---|")
    for t in order:
        r = results[t]
        b = results[base]
        L.append(f"| {t} | {pct(r['doc']['R@5'])} ({r['doc']['R@5'] - b['doc']['R@5']:+.1%}) | "
                 f"{r['doc']['MRR']:.3f} ({r['doc']['MRR'] - b['doc']['MRR']:+.3f}) | "
                 f"{pct(r['chunk']['R@5'])} ({r['chunk']['R@5'] - b['chunk']['R@5']:+.1%}) | "
                 f"{r['chunk']['MRR']:.3f} ({r['chunk']['MRR'] - b['chunk']['MRR']:+.3f}) | "
                 f"{r['chunk']['miss']} |")
    L.append("")

    L.append("## 三、按类别（块级 R@5）\n")
    cats = sorted({r["category"] for t in order for r in results[t]["rows"] if r["category"]})
    L.append("| 类别 | 题数 | " + " | ".join(f"{t} 篇" for t in order) + " |")
    L.append("|---|---|" + "---|" * len(order))
    for cat in cats:
        cnt = sum(1 for r in results[order[0]]["rows"] if r["category"] == cat)
        cells = []
        for t in order:
            rs = [r for r in results[t]["rows"] if r["category"] == cat]
            cells.append(pct(sum(1 for r in rs if r["chunk_rank"] and r["chunk_rank"] <= 5) / len(rs))
                         if rs else "—")
        L.append(f"| {cat} | {cnt} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 四、未召回题清单（块级，346 篇档）\n")
    last = order[-1]
    miss = [r for r in results[last]["rows"] if not r["chunk_rank"]]
    if miss:
        L.append("| id | 问题 | 类别 | 金标篇数 | 检索到的前 5 篇 |")
        L.append("|---|---|---|---|---|")
        for r in miss:
            L.append(f"| #{r['id']} | {r['q'][:26]} | {r['category']} | {r['n_gold']} | "
                     f"{'、'.join(sorted(set(r['chunk_top'])))} |")
    else:
        L.append("无 —— 全部题目的金标都在 top-5 块内。")
    L.append("")

    L.append("## 五、口径说明\n")
    L.append("- **块级召回 ≠ 篇级召回**：同一篇可能贡献多个块，块级 top-5 更容易被单篇占满，")
    L.append("  因此块级指标普遍低于篇级。真实 RAG 检的是块，块级才是线上口径。")
    L.append("- **R@k 是「至少召回一篇金标」**，多答案题里的「召回几篇」另见原始 JSON。")
    L.append("- 留出集已排除；本文所有数字都可用于调参，**不可**作为 W13 对外汇报数字。")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="检索侧规模评测（零 API）")
    ap.add_argument("--tiers", nargs="*", default=list(TIERS), choices=list(TIERS))
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--out", default=os.path.join(PROJECT, "eval", "reports", "retrieval_scale.md"))
    ap.add_argument("--json", default=os.path.join(PROJECT, "eval", "reports", "retrieval_scale.json"))
    args = ap.parse_args()

    cases = load_cases(args.allow_holdout)
    results, order = {}, []
    for t in args.tiers:
        docs = load_docs(TIERS[t])
        print(f"\n=== {t} 篇档：{len(docs)} 篇 ===", flush=True)
        r = eval_tier(cases, docs)
        results[t] = r
        order.append(t)
        d, ch = r["doc"], r["chunk"]
        print(f"  块数 {r['n_chunks']}｜题数 {r['n_cases']}")
        print(f"  篇级 R@1 {pct(d['R@1'])}｜R@3 {pct(d['R@3'])}｜R@5 {pct(d['R@5'])}｜MRR {d['MRR']:.3f}")
        print(f"  块级 R@1 {pct(ch['R@1'])}｜R@3 {pct(ch['R@3'])}｜R@5 {pct(ch['R@5'])}｜MRR {ch['MRR']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_report(results, order))
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({t: {k: v for k, v in r.items()} for t, r in results.items()},
                  f, ensure_ascii=False, indent=2)

    print(f"\n报告：{args.out}")
    print(f"原始：{args.json}")


if __name__ == "__main__":
    main()
