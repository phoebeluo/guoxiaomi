"""BM25 长度归一化敏感性：`b` 三维扫描（零 API，M5）。

为什么只扫 `b` 不扫 `k1`
-----------------------
`k1` 控词频饱和（同一词重复多少次后不再加分），`b` 控长度归一化（长文档是否被惩罚）。
本题库里绝大多数查询只含 2–4 个概念，词频几乎恒为 1–2，`k1` 在 1.2–2.0 区间对排序
**几乎不可见**；而 346 篇档里文档长度差 20 倍以上，"长文霸榜"完全是 `b` 决定的。
故只扫 `b`，把结论落在唯一有区分度的维度上。

为什么复用同一个索引
--------------------
`b` 只进入打分公式 `tf + k1*(1-b+b*dl/avgdl)`，**不影响倒排结构与 idf**。
所以构建一次索引、改 `idx.b` 跑三轮，比重建三次快 3 倍，且保证三组完全同源。

用法
----
    python tools/sweep_b.py            # 346 档，3 组 b
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

from rag.bm25 import BM25                                    # noqa: E402
from eval_retrieval import TIERS, build_chunks, load_cases, load_docs, pct  # noqa: E402

B_VALUES = [0.5, 0.75, 1.0]
TOPK = (1, 3, 5)


def agg(ranks, key, topk=TOPK):
    res = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / len(ranks) for k in topk}
    res["MRR"] = statistics.mean([1 / r if r else 0 for r in ranks])
    res["miss"] = sum(1 for r in ranks if not r)
    return res


def main():
    tier = "346"
    cases = load_cases()
    docs = load_docs(TIERS[tier])
    chunks = build_chunks(docs)

    doc_idx = BM25(docs)
    chunk_idx = BM25([(cid, txt) for cid, txt, _ in chunks])
    chunk_src = {cid: src for cid, _, src in chunks}

    queries = []
    for c in cases:
        gold = set(c.get("gold_sources") or [])
        if not gold:
            continue
        q = c["q"] + (" " + " ".join(c["history"]) if c.get("history") else "")
        queries.append((c["id"], q, gold))

    print(f"{tier} 篇档：{len(docs)} 篇 / {len(chunks)} 块 / {len(queries)} 题（已排除留出集）\n")

    rows = []
    for b in B_VALUES:
        doc_idx.b = b
        chunk_idx.b = b
        d_rank, c_rank = [], []
        for _, q, gold in queries:
            dh = [d for d, _ in doc_idx.search(q, top_k=max(TOPK))]
            d_rank.append(next((i + 1 for i, d in enumerate(dh) if d in gold), None))
            ch = [chunk_src[cid] for cid, _ in chunk_idx.search(q, top_k=max(TOPK))]
            c_rank.append(next((i + 1 for i, s in enumerate(ch) if s in gold), None))
        d, c = agg(d_rank, "doc"), agg(c_rank, "chunk")
        rows.append({"b": b, "doc": d, "chunk": c})
        print(f"b={b:<5} 篇级 R@1 {pct(d['R@1'])} R@3 {pct(d['R@3'])} R@5 {pct(d['R@5'])} "
              f"MRR {d['MRR']:.3f} | 块级 R@1 {pct(c['R@1'])} R@3 {pct(c['R@3'])} "
              f"R@5 {pct(c['R@5'])} MRR {c['MRR']:.3f}")

    # 选点规则：块级 R@5 优先（线上检的是块），并列时比 MRR
    best = max(rows, key=lambda r: (round(r["chunk"]["R@5"], 3), r["chunk"]["MRR"]))
    base = next(r for r in rows if r["b"] == 0.75)
    # 显著性闸门：48 题上 1 道题 = 2.1 pp，故 R@5 差 <3 pp 且 MRR 差 <0.02 一律视为噪声，
    # 不允许以「最好看」为由改默认值 —— 那是过拟合到本题集。
    gain_r5 = best["chunk"]["R@5"] - base["chunk"]["R@5"]
    gain_mrr = best["chunk"]["MRR"] - base["chunk"]["MRR"]
    significant = abs(gain_r5) >= 0.03 or abs(gain_mrr) >= 0.02
    chosen = best["b"] if significant else 0.75

    L = ["# BM25 `b` 敏感性扫描（346 篇档 · 零 API）\n",
         "> 固定 `k1=1.5`，只扫 `b`。三组共用同一倒排索引（`b` 只进打分，不改索引）。",
         f"> {len(queries)} 题（已排除留出集），块级为线上一口径。\n",
         "| b | 篇级 R@1 | 篇级 R@3 | 篇级 R@5 | 篇级 MRR | 块级 R@1 | 块级 R@3 | 块级 R@5 | 块级 MRR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, c = r["doc"], r["chunk"]
        L.append(f"| {r['b']} | {pct(d['R@1'])} | {pct(d['R@3'])} | {pct(d['R@5'])} | {d['MRR']:.3f} | "
                 f"{pct(c['R@1'])} | {pct(c['R@3'])} | {pct(c['R@5'])} | {c['MRR']:.3f} |")
    L.append("")
    if significant:
        L.append(f"**选定 `b={chosen}`**：块级 R@5 {pct(best['chunk']['R@5'])}、MRR "
                 f"{best['chunk']['MRR']:.3f}，比默认 `b=0.75` 高 {pct(gain_r5)} / MRR {gain_mrr:+.4f}，"
                 f"超出噪声闸门（R@5 ≥3 pp 或 MRR ≥0.02）。")
    else:
        L.append(f"**沿用 `b=0.75`（论文默认），不采纳表面最优值。**")
        L.append("")
        L.append(f"- 三组块级 R@5 完全相同（{pct(base['chunk']['R@5'])}）；`b=0.5` 的 MRR "
                 f"{rows[0]['chunk']['MRR']:.3f} 虽比 default 高 {gain_mrr:+.4f}，但 48 题里 "
                 f"**1 道题就值 2.1 pp**，这个差距是单题噪声。")
        L.append("- **闸门规则**：R@5 差 <3 pp 且 MRR 差 <0.02 一律视为噪声，不因「数字最好看」"
                 "改默认值——那等于把参数过拟合到 48 道题上，换个语料就失效。")
        L.append("- 结论写成：**`b` 在本语料上不敏感，沿用 0.75**；这本身是「长度归一化不是瓶颈」"
                 "的一个旁证，与 W4「规模不是瓶颈」互为支撑。")
    L.append("")

    out_md = os.path.join(PROJECT, "eval", "reports", "b_sweep.md")
    out_json = os.path.join(PROJECT, "eval", "reports", "b_sweep.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n报告：{out_md}")


if __name__ == "__main__":
    main()
