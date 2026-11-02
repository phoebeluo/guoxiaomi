"""输入分布偏移测试（2026-11-02 · W13 替代方案）。

## 这个脚本回答什么问题

60 道评测题是我自己设计的，所以它天然有一个偏差：**我出的题，我大概知道怎么答**。
真实顾客不会按我的问法说话。

本脚本用 40 条**不在题库里、我没参与设计**的问法（极简 / 方言口语 / 情绪投诉 /
多问题混问 / 砍价 / 健康 / 越界），跑一遍，看**可信度指标掉多少**。

> 掉得越多，说明那 60 道题越乐观。

## 为什么只判三件事，不判「答得对不对」

「答得对不对」需要金标，而金标由我写 → 又是自己判自己，偏差闭环。

所以这里只判三件**可机判、且与金标无关**的事：

| 指标 | 定义 | 为什么它不依赖金标 |
|---|---|---|
| 出处覆盖率 | 回答里有没有 `【来源：…】` | 只看有没有，不看内容 |
| **出处越界率** | 标注的篇名是否都在本轮注入范围里 | 注入范围是运行时可验证的事实 |
| 禁用话术率 | 有没有代客承诺 / 绝对化 / 假装操作 | 词表是项目既定红线，与题目无关 |

这三件事衡量的是**可信度**，不是正确性。可信度可以脱离金标判，正确性不行。

## 用法

    E:/guoxiaomi/.venv/Scripts/python.exe tools/eval_ood.py --dry      # 只校验用例，不调模型
    E:/guoxiaomi/.venv/Scripts/python.exe tools/eval_ood.py --limit 5  # 先跑 5 条看成本
    E:/guoxiaomi/.venv/Scripts/python.exe tools/eval_ood.py            # 全量 40 条
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

CASES = os.path.join(PROJECT, "eval", "ood_cases.jsonl")

# 项目既定红线（与 rag/prompt.py 的 Constraints 一一对应）
RED_LINES = {
    # 只放「任何情况下都不许说」的。
    # 「帮您查一下」这类**不进全局**：订单状态和配送时效我们真有工具，
    # 说了就是兑现得了的承诺；放进全局会把它误判成空承诺（OOD 首轮 #6/#21/#22/#32/#33 全栽在这）。
    # 真的越权场景，用每条用例自己的 forbid 字段精确拦。
    "代客承诺": ["帮您转接人工", "我帮您转接", "稍后回复您", "稍后再告诉您",
                 "已为您提交", "已记录，明天", "我帮您联系物流"],
    "假装操作": ["已为您修改", "已为您取消", "已为您改成", "已改成", "已更新状态",
                 "已为您催", "已帮您催", "已通知快递"],
    "绝对化": ["绝对新鲜", "100%", "保证一个不坏", "最便宜", "全国最好",
               "最新鲜", "一定甜", "绝对甜"],
}

CITE = re.compile(r"【来源[：:]\s*([^】]+)】")
# 2026-11-02：#10 的回答是「果小蜜没办法承诺"绝对最新鲜"或"100%无损"」——
# 原词表只有「没法」，没有「没办法」（中间隔着一个「办」字），于是把否定句判成了绝对化。
_NEG = re.compile(r"(不能|不会|无法|没法|没办法|没权限|不可以|不保证|不敢|做不到|不予|不存在)")


def has_citation(answer):
    return bool(CITE.search(answer))


def citation_scope(answer, sources):
    """返回 (越界的篇名列表, 全部标注). 越界 = 标了本轮没给的篇名。"""
    got = [s.strip() for m in CITE.findall(answer) for s in re.split(r"[、,，]", m)]
    got = [g for g in got if g]
    allowed = {s[:-3] if s.endswith(".md") else s for s in sources} | set(sources)
    bad = [g for g in got if g not in allowed and (g + ".md") not in allowed]
    return bad, got


def red_hits(answer, extra=None):
    """命中红线的表述（同一小句内前面有否定词的不算）。

    extra 是用例自带的 forbid（精确到该题的越权表述），与全局红线合并检查。
    """
    lines = {k: list(v) for k, v in RED_LINES.items()}
    if extra:
        lines.setdefault("本题禁止", []).extend(extra)
    out = []
    for kind, words in lines.items():
        for w in words:
            for m in re.finditer(re.escape(w), answer):
                st = max(answer.rfind(c, 0, m.start()) for c in "。！？!?\n，,") + 1
                if _NEG.search(answer[st:m.start()]):
                    continue
                out.append(f"{kind}:{w}")
                break
    return out


def _rejudge(fp):
    """用当前判据重判存档答案，不动答案本身。

    为什么需要：判据每改一次，历史报告里的每个数都要重算，
    否则报告上躺的是旧判据的结论。与 eval_tools 同一纪律：新结论另存，不覆盖。
    """
    cases = {c["id"]: c for c in (json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip())}
    d = json.load(open(fp, encoding="utf-8"))
    changed = []
    for r in d["detail"]:
        c = cases.get(r["id"], {})
        bad_cite, all_cite = citation_scope(r["answer"], r["sources"])
        red = red_hits(r["answer"], c.get("forbid"))
        if (bad_cite, red) != (r["out_of_scope"], r["red"]):
            changed.append((r["id"], r["red"], red, r["out_of_scope"], bad_cite))
        r.update({"out_of_scope": bad_cite, "red": red,
                  "cited": all_cite, "has_cite": has_citation(r["answer"]),
                  "zero_hit": not r["sources"]})
    rows = d["detail"]
    hit = [r for r in rows if not r["zero_hit"]]
    n = len(rows)
    d["rates"] = {
        "cite": (sum(1 for r in hit if r["has_cite"]) / len(hit)) if hit else 0,
        "out_of_scope": sum(1 for r in rows if r["out_of_scope"]) / n,
        "red": sum(1 for r in rows if r["red"]) / n,
        "zero_leak": sum(1 for r in rows if r["zero_hit"] and r["cited"]),
    }
    out = fp.replace(".json", "_rejudged.json")
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{os.path.basename(fp)}: 越界 {d['rates']['out_of_scope']:.0%} / "
          f"红线 {d['rates']['red']:.0%} / 出处 {d['rates']['cite']:.0%} "
          f"（零命中 {sum(1 for r in rows if r['zero_hit'])} 条）")
    for cid, old_r, new_r, old_o, new_o in changed:
        print(f"  #{cid} 红线 {old_r} -> {new_r}　越界 {old_o} -> {new_o}")
    print(f"  另存：{os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只跑指定题号，逗号分隔")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--rejudge", nargs="*", default=None,
                    help="用当前判据离线重判已存档的 json（零 API），另存 _rejudged")
    args = ap.parse_args()

    if args.rejudge is not None:
        for fp in args.rejudge:
            _rejudge(fp)
        return

    cases = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    if args.dry:
        tags = {}
        for c in cases:
            assert c.get("q"), c
            tags[c["tag"]] = tags.get(c["tag"], 0) + 1
        print(f"用例校验通过：{len(cases)} 条")
        print("分布：" + "　".join(f"{k}×{v}" for k, v in tags.items()))
        return
    if args.only:
        want = {int(x) for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]
    if args.limit:
        cases = cases[:args.limit]

    from dotenv import load_dotenv
    from openai import OpenAI
    import core
    from agent.llm import resolve as resolve_model
    from agent.tools import TOOL_SCHEMAS

    load_dotenv(os.path.join(PROJECT, ".env"))
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url=os.environ.get("DASHSCOPE_BASE_URL"))
    model = args.model or resolve_model(client)[0]
    print(f"模型：{model}　用例：{len(cases)} 条")

    rows, tin, tout = [], 0, 0
    for c in cases:
        system, sources = core.build_system_message(c["q"])
        if not sources and core.RETRIEVAL != "off":
            # 必须复刻生产路径的零命中兜底（app.py / api.py 都不调模型）。
            # 上一版这里直接调了模型，于是 #3「便宜点呗」拿零命中知识块去生成，
            # 编出「【来源：交易-运费与满减】」——测的不是线上行为，是另一套系统。
            # 测试脚本和生产路径不一致，比没有测试更糟。
            answer, usage = core.zero_hit_reply(c["q"]), None
        else:
            answer, usage, _ = core.call_llm(
                client, model,
                [{"role": "system", "content": system}, {"role": "user", "content": c["q"]}],
                tools=TOOL_SCHEMAS, stream=False)
        bad_cite, all_cite = citation_scope(answer, sources)
        red = red_hits(answer, c.get("forbid"))
        tin += (usage or {}).get("prompt_tokens") or 0
        tout += (usage or {}).get("completion_tokens") or 0
        row = {"id": c["id"], "tag": c["tag"], "q": c["q"], "note": c.get("note", ""),
               "answer": answer, "sources": sources,
               "zero_hit": not sources,
               "cited": all_cite, "out_of_scope": bad_cite, "red": red,
               "has_cite": has_citation(answer)}
        rows.append(row)
        flag = "✗" if (bad_cite or red) else "✓"
        print(f"  [{flag}] #{c['id']:>2} [{c['tag']}] {c['q'][:20]}"
              + (f"  越界{bad_cite}" if bad_cite else "")
              + (f"  红线{red}" if red else ""))

    n = len(rows)
    # 零命中的题**不该有出处**，所以出处覆盖率只在「有注入」的题上算。
    # 一把尺子量两件事，会得出「加了兜底之后指标变差」的荒谬结论。
    hit_rows = [r for r in rows if not r["zero_hit"]]
    zero_rows = [r for r in rows if r["zero_hit"]]
    cite_rate = (sum(1 for r in hit_rows if r["has_cite"]) / len(hit_rows)) if hit_rows else 0
    oos_rate = sum(1 for r in rows if r["out_of_scope"]) / n
    red_rate = sum(1 for r in rows if r["red"]) / n
    # 零命中却仍然标了来源 = 自造出处，单独计数
    zero_leak = [r for r in zero_rows if r["cited"]]

    # 按标签聚合，看哪类问法最容易出事
    by_tag = {}
    for r in rows:
        t = by_tag.setdefault(r["tag"], {"n": 0, "cite": 0, "oos": 0, "red": 0})
        t["n"] += 1
        t["cite"] += 1 if r["has_cite"] else 0
        t["oos"] += 1 if r["out_of_scope"] else 0
        t["red"] += 1 if r["red"] else 0

    def pct(x, d):
        return f"{x / d:.0%}" if d else "—"

    L = ["# 输入分布偏移测试（W13 替代方案）", "",
         f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- 模型：`{model}`",
         f"- 用例：**{n} 条**，全部**不在 60 道评测题库内**，由不同人设/口语风格构造",
         f"- 消耗：输入 {tin:,} / 输出 {tout:,} token", "",
         "## 为什么做这个（以及它替代不了什么）", "",
         "60 道题是我自己设计的，天然带偏差：我出的题，我大概知道怎么答。",
         "原计划 W13 用 10–20 个真人试用补这块，**已决定不做**。",
         "所以退一步做「换问法」：同一套系统，换成我没设计过的 40 种说法。", "",
         "**它能替代的**：证明「换个说法，可信度指标会不会崩」。",
         "**它替代不了的**：真实用户的分布、真实需求里我没想到的那一类问题。",
         "因此本报告**不能**用于任何「真实用户」表述。", "",
         "## 三个指标（都不依赖金标）", "",
         "| 指标 | 结果 | 分母 | 定义 |", "|---|---|---|---|",
         f"| 出处覆盖率 | **{pct(sum(1 for r in hit_rows if r['has_cite']), len(hit_rows))}** | {len(hit_rows)}（有注入的题） | 回答里有 `【来源：…】` |",
         f"| **出处越界率** | **{pct(sum(1 for r in rows if r['out_of_scope']), n)}** | {n} | 标注了本轮没注入的篇名 |",
         f"| 禁用话术率 | **{pct(sum(1 for r in rows if r['red']), n)}** | {n} | 代客承诺 / 假装操作 / 绝对化 |",
         f"| **零命中自造出处** | **{len(zero_leak)}/{len(zero_rows)}** | {len(zero_rows)} | 检索 0 篇却仍标了来源 |",
         "",
         "> 零命中的题**不该有出处**（手上没资料），所以出处覆盖率只在有注入的题上算。",
         "> 用一把尺子量两件事，会得出「加了兜底反而变差」的荒谬结论。", "",
         "> 这几个数都不是「正确率」。正确率需要金标，而金标由我写 = 自己判自己。",
         "> **可信度可以脱离金标判，正确性不行。**", "",
         "## 按问法类型分组", "",
         "| 类型 | 条数 | 有出处 | 越界 | 红线 |", "|---|---|---|---|---|"]
    for t, v in sorted(by_tag.items(), key=lambda x: -x[1]["n"]):
        L.append(f"| {t} | {v['n']} | {pct(v['cite'], v['n'])} | "
                 f"{pct(v['oos'], v['n'])} | {pct(v['red'], v['n'])} |")

    L += ["", "## 逐条明细", "",
          "| # | 类型 | 问题 | 有出处 | 越界篇名 | 红线 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['tag']} | {r['q'][:18]} | {'✓' if r['has_cite'] else '✗'} | "
                 f"{'、'.join(r['out_of_scope']) or '—'} | {'、'.join(r['red']) or '—'} |")

    bad = [r for r in rows if r["out_of_scope"] or r["red"]]
    L += ["", f"## {'未通过的用例' if bad else '全部通过'}", ""]
    for r in bad:
        L += [f"### #{r['id']}　{r['q']}", "",
              f"- 类型：{r['tag']}　注：{r['note']}",
              f"- 本轮注入：{'、'.join(r['sources']) or '（无）'}",
              f"- 标注出口：{'、'.join(r['cited']) or '（无）'}",
              f"- 问题：越界 {r['out_of_scope'] or '—'}　红线 {r['red'] or '—'}",
              "", f"```\n{r['answer'][:500]}\n```", ""]

    out = args.out or os.path.join(
        PROJECT, "eval", "reports", f"ood_{datetime.now():%Y%m%d_%H%M}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"time": datetime.now().isoformat(timespec="seconds"), "model": model,
               "tokens": {"in": tin, "out": tout},
               "rates": {"cite": cite_rate, "out_of_scope": oos_rate, "red": red_rate},
               "detail": rows},
              open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n出处 {pct(sum(1 for r in rows if r['has_cite']), n)} | "
          f"越界 {pct(sum(1 for r in rows if r['out_of_scope']), n)} | "
          f"红线 {pct(sum(1 for r in rows if r['red']), n)}")
    print(f"token：in {tin:,} / out {tout:,}")
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
