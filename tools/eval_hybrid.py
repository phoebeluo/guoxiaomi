"""三路检索对比：BM25 / SVD 低秩近似 / RRF 融合（346 篇档，零 API）。

为什么要这个脚本
----------------
W4 只跑了 BM25 单策略，被问「你怎么知道 BM25 够用」时答不上来。
这里把「文字匹配」「语义近似」「两者融合」放在同一批 48 题、同一套指标下比，
并且**只比排名，不比分数** —— 原因见下方「为什么用 RRF」。

三路的定位
----------
A. BM25        数共同的词，稀有词加权，长文打折。快、可解释、只认字面。
B. SVD(128)    把每篇文档压成 128 个数字，看方向接近不接近。能捕捉「词常一起出现」，
               但**不是 embedding**，捕捉不到真正的同义关系（见 rag/vector.py 顶部标注）。
C. RRF 融合    把 A、B 各自的排名合成一份。不看分数，只看名次。

为什么用 RRF 而不是「分数加权相加」
---------------------------------
1. BM25 分数无上界（实测同一批题 top1 从 2.26 到 39.00），余弦在 [-1, 1]；
   两者相加等于把「米」和「斤」加在一起，权重怎么调都是拍脑袋。
2. RRF 只用名次：排第 1 得 1/(60+1)，排第 2 得 1/(60+2)…… 与分数量纲无关。
3. 60 是 RRF 原论文的经验常数，作用是压低头部差异，让第 1 名和第 2 名不至于差太多。

用法
----
    python tools/eval_hybrid.py                # 346 篇档
    python tools/eval_hybrid.py --dim 64        # 换降维维度
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

from rag.bm25 import BM25                                      # noqa: E402
from rag.vector import VectorIndex                             # noqa: E402
from eval_retrieval import TIERS, load_cases, load_docs, pct    # noqa: E402

RRF_K = 60          # RRF 原论文经验常数
POOL = 10           # 每路先取前 10 名再融合（只取 5 会丢掉「一路第 8、另一路第 2」这种互补）


def rrf_fuse(rankings, k=RRF_K):
    """rankings: [[(name, score), ...], ...] → [(name, rrf_score), ...] 降序。"""
    score = defaultdict(float)
    for hits in rankings:
        for rank, (name, _) in enumerate(hits, 1):
            score[name] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def evaluate(queries, getter):
    rows, r1, r3, r5, mrr = {}, 0, 0, 0, []
    for cid, q, gold in queries:
        hits = getter(q)
        rank = next((i + 1 for i, (n, _) in enumerate(hits) if n in gold), None)
        rows[cid] = rank
        r1 += 1 if rank == 1 else 0
        r3 += 1 if rank and rank <= 3 else 0
        r5 += 1 if rank and rank <= 5 else 0
        mrr.append(1 / rank if rank else 0)
    n = len(queries)
    return {"R1": r1 / n, "R3": r3 / n, "R5": r5 / n,
            "MRR": statistics.mean(mrr), "rows": rows, "n": n}


def main():
    ap = argparse.ArgumentParser(description="三路检索对比（零 API）")
    ap.add_argument("--dim", type=int, default=128, help="SVD 降维维度")
    ap.add_argument("--tier", default="346", choices=list(TIERS))
    ap.add_argument("--out", default=os.path.join(PROJECT, "eval", "reports", "hybrid_retrieval.md"))
    args = ap.parse_args()

    cases = load_cases()
    docs = load_docs(TIERS[args.tier])
    bm = BM25(docs)
    vi = VectorIndex(docs, dim=args.dim, verbose=True)

    queries = [(c["id"], c["q"] + (" " + " ".join(c["history"]) if c.get("history") else ""),
                set(c["gold_sources"])) for c in cases if c.get("gold_sources")]
    print(f"{args.tier} 篇档 / {len(queries)} 题（已排除留出集）\n")

    res = {}
    res["BM25"] = evaluate(queries, lambda q: bm.search(q, top_k=POOL))
    res["SVD"] = evaluate(queries, lambda q: vi.search(q, top_k=POOL))
    res["RRF"] = evaluate(
        queries,
        lambda q: rrf_fuse([bm.search(q, top_k=POOL), vi.search(q, top_k=POOL)])[:POOL],
    )

    for name in ["BM25", "SVD", "RRF"]:
        r = res[name]
        print(f"{name:6} R@1 {pct(r['R1'])}  R@3 {pct(r['R3'])}  "
              f"R@5 {pct(r['R5'])}  MRR {r['MRR']:.3f}")

    # ---- 逐题胜负：谁修好了谁 ----
    b, v, f = res["BM25"]["rows"], res["SVD"]["rows"], res["RRF"]["rows"]
    def cmp(x, y):
        return [(i, b[i], v[i] if y is v else 0) for i in b
                if (x[i] or 99) != (y[i] or 99)]
    b_fix = [(i, b[i], f[i]) for i in b if (f[i] or 99) < (b[i] or 99)]
    b_break = [(i, b[i], f[i]) for i in b if (f[i] or 99) > (b[i] or 99)]
    v_only = [(i, b[i], v[i]) for i in b if (v[i] or 99) < (b[i] or 99)]
    b_only = [(i, b[i], v[i]) for i in b if (v[i] or 99) > (b[i] or 99)]

    L = ["# 三路检索对比（346 篇 · 零 API）\n",
         f"> {args.tier} 篇 / {len(queries)} 题（已排除 8 题留出集）/ SVD {args.dim} 维 / "
         f"RRF k={RRF_K} / 每路取前 {POOL} 名融合。",
         "> 三路共用同一批题、同一套指标；**只比排名，不比分数**（量纲不可比）。\n",
         "## 一、总表\n",
         "| 策略 | R@1 | R@3 | R@5 | MRR | 说明 |",
         "|---|---|---|---|---|---|",
         f"| BM25 | {pct(res['BM25']['R1'])} | {pct(res['BM25']['R3'])} | {pct(res['BM25']['R5'])} | {res['BM25']['MRR']:.3f} | 字面匹配，可解释 |",
         f"| SVD({args.dim}) | {pct(res['SVD']['R1'])} | {pct(res['SVD']['R3'])} | {pct(res['SVD']['R5'])} | {res['SVD']['MRR']:.3f} | 低秩语义近似，**不是 embedding** |",
         f"| **RRF 融合** | {pct(res['RRF']['R1'])} | {pct(res['RRF']['R3'])} | {pct(res['RRF']['R5'])} | {res['RRF']['MRR']:.3f} | 只用名次，与分数量纲无关 |",
         "",
         "## 二、逐题胜负（名次，越小越好；None = 前 10 名都没命中）\n",
         f"**向量比 BM25 好的 {len(v_only)} 题：**\n",
         "| 题号 | BM25 | 向量 |", "|---|---|---|"]
    for i, a, c in v_only:
        L.append(f"| #{i} | {a or '未召回'} | {c or '未召回'} |")
    L += ["", f"**向量不如 BM25 的 {len(b_only)} 题：**\n",
          "| 题号 | BM25 | 向量 |", "|---|---|---|"]
    for i, a, c in b_only:
        L.append(f"| #{i} | {a or '未召回'} | {c or '未召回'} |")
    best_r3 = max(res.items(), key=lambda kv: kv[1]["R3"])
    best_r1 = max(res.items(), key=lambda kv: kv[1]["R1"])
    L += ["", "## 四、该用哪个：口径决定结论\n",
          "线上注入 **3 篇**，所以真实指标是 **R@3**，不是 R@1。",
          f"- 按 R@3 排：**{best_r3[0]}** 最高（{pct(best_r3[1]['R3'])}）",
          f"- 按 R@1 排：**{best_r1[0]}** 最高（{pct(best_r1[1]['R1'])}）",
          "",
          "两个口径给出不同答案，这本身是结论：**R@1 高代表「第一篇最准」（影响模型注意力分配），",
          "R@3 高代表「三篇里至少有一篇对」（影响能不能答对）**。线上一次注入 3 篇，模型会通读，",
          "所以 R@3 是更贴近实际的口径。",
          "",
          "## 五、为什么不把分数直接相加\n",
          "1. BM25 分数**无上界**（同一批题 top1 从 2.26 到 39.00），余弦在 [-1, 1]；",
          "2. 两者相加等于把「米」和「斤」加在一起，权重怎么调都是拍脑袋；",
          "3. RRF 只看名次：第 1 名得 1/(60+1)、第 2 名 1/(60+2)，与分数量纲无关。",
          "   60 是原论文经验常数，作用是压低头部差异——让第 1 名和第 2 名不至于差出一倍。",
          "",
          "⚠️ 本表 SVD 列**不是 embedding**。简历只能写「稀疏检索与低秩语义近似的召回对比」。",
          ""]

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res.items()},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n报告：{args.out}")


if __name__ == "__main__":
    main()
