"""幻觉代理指标（零 API）：从**已有存档答案**算「编造率」与「越界率」两栏。

为什么叫「代理」而不叫「幻觉率」
--------------------------------
真正的幻觉要人判：「这句话是否与事实不符」。那需要逐句核对语料，成本高且不可复现。
这里只算两个**可自动化、可复现**的代理指标，它们各自对应一类**真实发生过**的事故：

  · **编造率** —— 库外题 / 约束题里，模型说出了语料没有的东西（#15 承诺代办、#16 编快递公司）。
    数据源：历史报告的 detail（判据已固化这些 pattern），零 API、可反复算。

  · **越界率** —— 回答标注的【来源：X】里，X 不在本轮实际注入的篇名中。
    **这是 W5 上线第一天真实抓到的 badcase**：模型引用了 `售后-坏果赔付政策`，
    而那一轮只注入了 3 篇、其中没有它——它把正文里《》交叉引用当成了自己的出处。
    数据源：`data/logs/requests.jsonl`（只有它记了 sources 字段）。

口径限制（必须写在报告里，否则数字会被误用）
------------------------------------------
1. 越界率只在**有来源标注**的回答上算；没有标注的不进分母（分母为 0 时报「—」）。
2. 越界率样本来自 2026-10-29 那 3 条线上对话，**在 prompt 修补之前**。
   修补后没有再调模型，所以「修补后越界率」当前无法测——只能写「待补」。
3. 编造率依赖 `eval/cases.jsonl` 里的 pattern 覆盖，pattern 覆盖不到的话题不会被算进分子。
   它是**下界**，不是全量。
4. 两者都只用**已存档答案**，因此不会因为「再跑一次模型」而漂移——同一份存档永远得同一个数。

用法
----
    python tools/hallucination_proxy.py
"""

import glob
import json
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "tools"))

from eval_run import judge                      # noqa: E402

REPORTS = os.path.join(PROJECT, "eval", "reports")
LOG = os.path.join(PROJECT, "data", "logs", "requests.jsonl")
OUT = os.path.join(REPORTS, "hallucination_proxy.md")

SRC_RE = re.compile(r"【来源[：:]\s*([^】]+)】")


def load_cases():
    path = os.path.join(PROJECT, "eval", "cases.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def strip_md(name):
    return name[:-3] if name.endswith(".md") else name


# ----------------------------------------------------------------------
# 一、越界率：回答标注的来源，是否在本轮注入范围内
# ----------------------------------------------------------------------
def citation_scope():
    rows = []
    if not os.path.exists(LOG):
        return rows
    for line in open(LOG, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        injected = {strip_md(s) for s in (r.get("sources") or [])}
        if not injected:
            continue                      # 没注入就没有范围可比，不进分母
        cited = []
        for m in SRC_RE.finditer(r.get("answer") or ""):
            for part in re.split(r"[、,，/]+", m.group(1)):
                part = part.strip().strip("《》 ")
                if part:
                    cited.append(part)
        if not cited:
            continue                      # 没有标注的不进分母
        outside = sorted({c for c in cited if c not in injected})
        rows.append({
            "ts": r.get("ts"), "q": (r.get("q") or "")[:24],
            "injected": sorted(injected), "cited": cited, "outside": outside,
        })
    return rows


# ----------------------------------------------------------------------
# 二、编造率：库外 / 约束题里命中「禁止事实」的比例
# ----------------------------------------------------------------------
def fabrication(src_json):
    data = json.load(open(src_json, encoding="utf-8"))
    detail = data.get("detail") or {}
    if not detail:
        return None
    cases = {c["id"]: c for c in load_cases()}
    rows = []
    for mode in sorted(detail):
        stat = {"none": [0, 0], "constraint": [0, 0]}     # [编造数, 总数]
        bad = []
        for cid, r in detail[mode].items():
            c = cases.get(int(cid))
            if not c or c["expect"] not in stat:
                continue
            # 用**当前**判据重算，保证两栏口径一致（不直接采信存档里的 ok）
            _, hits, _ = judge(r["answer"], c)
            stat[c["expect"]][1] += 1
            if hits:
                stat[c["expect"]][0] += 1
                bad.append((cid, c["q"][:22], hits))
        rows.append({"mode": mode, "stat": stat, "bad": bad})
    return {"file": os.path.basename(src_json), "rows": rows}


def main():
    cites = citation_scope()
    fab = []
    for name in ("w4_gold_cases52_rejudged.json", "w2_cases60_gold_rejudged_v2.json"):
        p = os.path.join(REPORTS, name)
        if os.path.exists(p):
            r = fabrication(p)
            if r:
                fab.append(r)

    L = ["# 幻觉代理指标（零 API）", "",
         "> 两个**代理**指标，不是「幻觉率」。由来与口径限制见 `tools/hallucination_proxy.py` 文件头。",
         "> 数据源全部是**已存档答案**，不调用模型，因此可反复复现同一个数。", ""]

    # ---------- 越界率 ----------
    L += ["## 一、越界率（引用范围外文档）", ""]
    tot = len(cites)
    out_n = sum(1 for c in cites if c["outside"])
    L += [f"**{out_n} / {tot}** —— 有来源标注的回答中，引用了本轮未注入篇名的比例。", ""]
    if tot:
        L += ["| 时间 | 问题 | 本轮注入 | 标注的越界篇名 |", "|---|---|---|---|"]
        for c in cites:
            L.append(f"| {c['ts']} | {c['q']} | {'、'.join(c['injected'])} | "
                     f"{'、'.join(c['outside']) or '—'} |")
        L.append("")
    L += ["> ⚠️ 分母只有 %d 条：`requests.jsonl` 里记录了 `sources` 的线上对话就这么多。" % tot,
          "> 且全部来自 **2026-10-29（prompt 修补之前）**。修补后未再调模型，"
          "「修补后越界率」当前**无法测**，登记为待补。", ""]

    # ---------- 编造率 ----------
    L += ["## 二、编造率（库外 / 约束题里说出语料没有的东西）", ""]
    if not fab:
        L.append("（无可用存档报告）")
    for f in fab:
        L.append(f"### 来源：`{f['file']}`\n")
        L += ["| 策略 | 题型 | 编造数 | 题数 | 编造率 |", "|---|---|---|---|---|"]
        for r in f["rows"]:
            for k in ("none", "constraint"):
                n, d = r["stat"][k]
                L.append(f"| {r['mode']} | {k} | {n} | {d} | "
                         f"{('%.1f%%' % (100 * n / d)) if d else '—'} |")
        L.append("")
        for r in f["rows"]:
            if r["bad"]:
                L += [f"`{r['mode']}` 命中的具体题（「编造」判定的证据）：", "",
                      "| 题 | 问题 | 说出的禁止事实 |", "|---|---|---|"]
                for cid, q, hits in r["bad"]:
                    L.append(f"| #{cid} | {q} | {'、'.join(hits)} |")
                L.append("")

    L += ["## 三、怎么用这两个数", "",
          "| 数字 | 能说明什么 | 不能说明什么 |", "|---|---|---|",
          f"| 越界率 {out_n}/{tot} | 出处标注会越过注入范围，且**文件真实存在**，肉眼查不出来 | "
          "不能代表整体幻觉水平；样本仅 %d 条 |" % tot,
          "| 编造率（分题型） | 库外题上模型会不会张嘴就来 | 是下界：pattern 覆盖不到的话题不计入分子 |",
          "",
          "> ⚠️ **编造率全 0 不等于「模型不会编」。** 它的分母只有 "
          f"{sum(r['stat']['none'][1] + r['stat']['constraint'][1] for f in fab for r in f['rows']) // max(len(fab), 1)} 题左右，"
          "且 pattern 只覆盖已知的编造形态（承诺代办、编快递公司、用绝对化用语）。"
          "换成没见过的问法，它测不出来。",
          "",
          "> 面试口径：**「我没有幻觉率这个数，我给的是两个能复现的代理指标，"
          "并写清了各自的样本量和口径限制。」** 说得出限制，比报一个漂亮数字可信。", ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\n报告已写入：{OUT}")


if __name__ == "__main__":
    main()
