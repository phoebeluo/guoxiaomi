"""金标可验证性审计（W2 收尾，2026-10-26）。

思路
----
把「金标是否独立支撑」从人工印象变成机器证据：
**用判据本身去判金标文档的原文**。

- `has` 题：拿该题 `pattern` 去跑每一篇 `gold_sources` 的正文。
  - 命中 → 该篇能独立支撑这条判据所需事实（金标成立）
  - 不命中 → 该篇只是「话题相关」，不是「独立支撑」
    （《金标规则说明》第六节第 4 步要求删掉这种篇目）
- `none` / `constraint` 题：拿 `pattern` 去跑**全量语料**。
  - 命中 → 语料里其实有这条答案，「库外」标注可疑

为什么复用 eval_run.judge()：口径必须与真实评测完全一致（含否定语境剔除），
否则审计结论与评测结论会打架。

零成本，不调 API。用法：
    E:/guoxiaomi/.venv/Scripts/python.exe tools/audit_gold.py
    E:/guoxiaomi/.venv/Scripts/python.exe tools/audit_gold.py --out eval/reports/gold_audit_20261026.md
"""

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

import eval_run as ER                     # noqa: E402
from rag.load_docs import DOCS_DIR        # noqa: E402


def read_doc(name):
    with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
        return f.read()


def audit(cases):
    docs = {n: read_doc(n) for n in ER._all_doc_names()}
    all_text = ER._docs_content()["content"]

    hard, partial, ok_single, ok_multi = [], [], [], []
    leak = []          # none/constraint 题在语料里被命中
    for c in cases:
        gs = c.get("gold_sources") or []
        if c["expect"] == "has":
            detail = []
            for g in gs:
                ok, hits, ref = ER.judge(docs[g], c) if g in docs else (False, [], [])
                detail.append({"doc": g, "hits": hits[:3], "refusals": ref})
            supported = [d for d in detail if d["hits"]]
            rec = {"id": c["id"], "q": c["q"], "detail": detail,
                   "n_gold": len(gs), "n_supported": len(supported)}
            if not supported:
                hard.append(rec)
            elif len(supported) < len(gs):
                partial.append(rec)
            elif len(gs) >= 2:
                ok_multi.append(rec)
            else:
                ok_single.append(rec)
        else:
            ok, hits, ref = ER.judge(all_text, c)
            if hits:
                where = [n for n, t in docs.items() if any(h for h in hits if h in t)]
                leak.append({"id": c["id"], "q": c["q"], "expect": c["expect"],
                             "hits": hits[:5], "docs": where})
    return dict(docs=docs, hard=hard, partial=partial,
                ok_single=ok_single, ok_multi=ok_multi, leak=leak)


def build_md(cases, r):
    L = []
    L.append("# 金标可验证性审计（W2 收尾）\n")
    L.append("> 方法：**用判据本身去判金标文档原文**（复用 `eval_run.judge()`，含否定语境剔除）。")
    L.append("> 零 API 成本。依据《金标规则说明-20261015.md》第六节「独立支撑」口径。\n")

    n_has = sum(1 for c in cases if c["expect"] == "has")
    n_kuwai = len(cases) - n_has
    L.append("## 一、总分\n")
    L.append("| 项 | 数量 |")
    L.append("|---|---|")
    L.append(f"| `has` 题 | {n_has} |")
    L.append(f"| `none`/`constraint` 题 | {n_kuwai} |")
    L.append(f"| 金标全部篇目均可支撑（✅） | {len(r['ok_single']) + len(r['ok_multi'])} |")
    L.append(f"| 部分篇目不可支撑（⚠️ 候选冗余） | {len(r['partial'])} |")
    L.append(f"| 全部篇目均不可支撑（❌ 严重） | {len(r['hard'])} |")
    L.append(f"| 库外题在语料中被命中（❌ 标注可疑） | {len(r['leak'])} |")
    L.append("")

    if r["hard"]:
        L.append("## 二、❌ 严重：金标篇目无一能支撑判据\n")
        L.append("| id | 问题 | 金标 | 各篇命中情况 |")
        L.append("|---|---|---|---|")
        for x in r["hard"]:
            det = "；".join(f"{d['doc']}→{d['hits'] or '无'}" for d in x["detail"])
            L.append(f"| #{x['id']} | {x['q']} | {x['n_gold']} 篇 | {det} |")
        L.append("")

    if r["partial"]:
        L.append("## 三、⚠️ 候选冗余：多答案题里有篇目撑不住判据\n")
        L.append("> 判定口径：该篇正文命中不了 pattern 所需的具体事实 → 只是「话题相关」，")
        L.append("> 按《金标规则说明》第六节第 4 步应删除；删了就要重新算 7:2:1。\n")
        L.append("| id | 问题 | 金标篇数 | 撑得住的 | 撑不住的 |")
        L.append("|---|---|---|---|---|")
        for x in r["partial"]:
            good = [d["doc"] for d in x["detail"] if d["hits"]]
            bad = [d["doc"] for d in x["detail"] if not d["hits"]]
            L.append(f"| #{x['id']} | {x['q']} | {x['n_gold']} | "
                     f"{'、'.join(good)} | {'、'.join(bad)} |")
        L.append("")

    if r["leak"]:
        L.append("## 四、❌ 库外题实际上语料里有答案\n")
        L.append("| id | 问题 | expect | 命中片段 | 命中篇目 |")
        L.append("|---|---|---|---|---|")
        for x in r["leak"]:
            L.append(f"| #{x['id']} | {x['q']} | {x['expect']} | "
                     f"{x['hits']} | {'、'.join(x['docs'][:3])} |")
        L.append("")
    else:
        L.append("## 四、库外题污染检查\n")
        L.append("✅ 6 道库外题的 pattern 在全量语料中**零命中** —— 「语料确实没有这个答案」成立。\n")

    used = set()
    for c in cases:
        used.update(c.get("gold_sources") or [])
    orphans = sorted(set(r["docs"]) - used)
    L.append("## 五、未被任何题引用的篇目（题库盲区）\n")
    if orphans:
        L.append(f"共 {len(orphans)} 篇：")
        for n in orphans:
            L.append(f"- {n}")
    else:
        L.append("无——45 篇全部至少被一题引用。")
    L.append("")

    L.append("## 六、字段统计核对\n")
    L.append("| 项 | 实测 |")
    L.append("|---|---|")
    L.append(f"| 题目总数 | {len(cases)} |")
    L.append(f"| 按类别 | {dict(Counter(c.get('category', '—') for c in cases))} |")
    L.append(f"| 按 expect | {dict(Counter(c['expect'] for c in cases))} |")
    L.append(f"| 按 kind | {dict(Counter(c.get('kind') for c in cases))} |")
    L.append(f"| 多轮题（history 非空） | {sum(1 for c in cases if c.get('history'))} |")
    L.append(f"| 留出集 | {sum(1 for c in cases if c.get('holdout'))} |")
    L.append(f"| 对照组题（counted=false） | {sum(1 for c in cases if not c.get('counted', True))} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="金标可验证性审计")
    ap.add_argument("--out", default=os.path.join(
        PROJECT, "eval", "reports", "gold_audit_latest.md"))
    args = ap.parse_args()

    cases = ER.load_cases()
    errs = ER.check_cases(cases)
    if errs:
        print("⚠️ --check 未通过，先修题库：")
        for e in errs:
            print("  ✗", e)
        raise SystemExit(1)

    r = audit(cases)
    md = build_md(cases, r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"金标可验证性审计：{len(cases)} 题")
    print(f"  ✅ 全部支撑       {len(r['ok_single']) + len(r['ok_multi']):>2} 题")
    print(f"  ⚠️ 部分篇目不支撑 {len(r['partial']):>2} 题")
    print(f"  ❌ 全部篇目均不支撑 {len(r['hard']):>2} 题")
    print(f"  ❌ 库外题被命中   {len(r['leak']):>2} 题")
    print(f"报告：{args.out}")


if __name__ == "__main__":
    main()
