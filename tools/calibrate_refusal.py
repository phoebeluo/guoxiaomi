"""拒答阈值标定：能不能用检索分数判断「库里有没有答案」（零 API）。

为什么这个脚本的结论很可能是「不能」
----------------------------------
主轨计划 W8 的原始设计是：BM25 分数无界 → 改用有界余弦 → 设阈值 → 低于阈值就拒答。
但「有界」不等于「可分」。余弦有界只保证**能定义阈值**，不保证**存在一个能分开两类题的阈值**。

本脚本把 60 题的最高余弦算出来，画一张阈值扫描表：
    误拒率 = 库内题（该回答）里被判成「该拒答」的比例   ← 用户问了能答的问题却被拒
    漏拒率 = 库外题（该拒答）里被判成「该回答问题」的比例 ← 该拒答却硬答
两者此消彼长。如果**不存在**一个阈值让两者都可接受，说明分数判别这条路不通，
必须换机制（规则化的能力边界清单），而不是继续调阈值。

指标口径
--------
- 库内 = 有 `gold_sources` 的题（54 道）
- 库外 = 无 `gold_sources` 的题（6 道：5 规则 + 1 边界）
- 分数 = 检索 top-1 的余弦相似度（45 篇档，与线上一口径）

用法
----
    python tools/calibrate_refusal.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

from rag.vector import VectorIndex                     # noqa: E402
from eval_retrieval import TIERS, load_docs            # noqa: E402

CASES = os.path.join(PROJECT, "eval", "cases.jsonl")
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]   # 0.30 ~ 0.95


def load_all_cases():
    rows = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    return [c for c in rows if not c.get("holdout")]      # 调参期不看留出集


def score_all(idx, cases):
    """返回 [(id, 是否有金标, 最高余弦, 问题)]。"""
    out = []
    for c in cases:
        q = c["q"] + (" " + " ".join(c["history"]) if c.get("history") else "")
        hits = idx.search(q, top_k=1)
        cos = float(hits[0][1]) if hits else 0.0
        out.append((c["id"], bool(c.get("gold_sources")), cos, c["q"]))
    return out


def sweep(rows):
    inside = [r for r in rows if r[1]]
    outside = [r for r in rows if not r[1]]
    table = []
    for t in THRESHOLDS:
        # 阈值 t：低于 t 判为「拒答」
        false_reject = sum(1 for r in inside if r[2] < t) / len(inside)
        missed_reject = sum(1 for r in outside if r[2] >= t) / len(outside)
        table.append({"t": t, "fa": false_reject, "miss": missed_reject})
    return inside, outside, table


def build_report(rows, inside, outside, table, out_md):
    ins = sorted(r[2] for r in inside)
    outs = sorted((r[2] for r in outside), reverse=True)
    L = ["# 拒答阈值标定（45 篇档 · 零 API）\n",
         f"> {len(inside)} 道库内题（有金标）+ {len(outside)} 道库外题（无金标），"
         "已排除 8 题留出集。分数 = 检索 top-1 余弦。\n",
         "## 一、两类题的分数分布\n",
         "| 组 | 最小 | P10 | 中位 | 最大 |", "|---|---|---|---|---|",
         f"| 库内（该回答） | {ins[0]:.3f} | {ins[len(ins) // 10]:.3f} | "
         f"{ins[len(ins) // 2]:.3f} | {ins[-1]:.3f} |",
         f"| 库外（该拒答） | {min(outs):.3f} | — | — | {max(outs):.3f} |",
         "",
         "## 二、阈值扫描（误拒率 vs 漏拒率）\n",
         "| 阈值 | 误拒率（能答的题被拒） | 漏拒率（该拒的题硬答） |",
         "|---|---|---|"]
    for r in table:
        L.append(f"| {r['t']:.2f} | {r['fa'] * 100:.1f}% | {r['miss'] * 100:.1f}% |")

    # 找一个「两边都低」的阈值：误拒 ≤5% 且 漏拒 ≤40%
    ok = [r for r in table if r["fa"] <= 0.05 and r["miss"] <= 0.40]
    L += ["", "## 三、结论\n"]
    if ok:
        best = min(ok, key=lambda r: (r["miss"], r["fa"]))
        L += [f"存在可用阈值 **t={best['t']:.2f}**：误拒 {best['fa'] * 100:.1f}%、"
              f"漏拒 {best['miss'] * 100:.1f}%。可上线做兜底。"]
    else:
        L += ["**不存在可用阈值。** 两个区间的分数完全交叠，无论阈值取在哪，"
              "要么误拒大量能答的问题，要么放过大量该拒的问题。",
              "",
              "### 为什么分数分不开\n",
              "库外题不是「无关话题」，而是「相关话题、但库里没写」。例如"
              "「投诉怎么分级」「能保证一个都不坏吗」——这些题与语料共享大量词汇，"
              "余弦自然很高。分数衡量的是**字面/共现接近度**，不是**库里有没有答案**。",
              "",
              "### 改用机制\n",
              "不调阈值，改为**规则化的能力边界清单**（见 `rag/refusal.py`）：",
              "把「系统原理上不可能做到的事」列成清单（查物流、改订单、登记回访、审批赔付），"
              "命中即给指路话术。这类判断靠业务定义，不靠相似度。",
              "",
              "> 面试价值：先做分数判别 → 实测两类区间交叠 → 换成规则清单。",
              "> **证伪一条路，比硬调一个阈值值钱。**"]
    L.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


def main():
    cases = load_all_cases()
    idx = VectorIndex(load_docs(TIERS["45"]))
    rows = score_all(idx, cases)
    inside, outside, table = sweep(rows)
    out_md = os.path.join(PROJECT, "eval", "reports", "refusal_threshold.md")
    text = build_report(rows, inside, outside, table, out_md)
    print(text)
    print(f"\n报告：{out_md}")


if __name__ == "__main__":
    main()
