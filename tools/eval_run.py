"""评测运行器：读 eval/cases.jsonl，对多种「策略」跑同一批题，输出横向对比。

设计要点
--------
1. **数据与代码分离**：题目在 eval/cases.jsonl，本文件只负责执行与判定。
   后续接 P1（类别路由）/ P3（BM25）时，只需在 STRATEGIES 里加一项，
   题目一行都不用改 —— 这正是「同一批题反复跑」的意义。
2. **路径全部从 __file__ 推导**，不写死绝对路径（换机器不会崩）。
3. `--check` 零成本：只校验题目文件的字段与正则，不调用 API。

用法
----
    # 零成本：只校验题目文件
    python tools/eval_run.py --check

    # 跑全部策略（消耗 API）
    python tools/eval_run.py

    # 只跑某一种策略
    python tools/eval_run.py --mode inject

    # 指定输出
    python tools/eval_run.py --out eval/reports/p1_route.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                      # tools/ 的上一级 = 项目根
sys.path.insert(0, PROJECT)

CASES_PATH = os.path.join(PROJECT, "eval", "cases.jsonl")
DEFAULT_OUT = os.path.join(PROJECT, "eval", "reports", "eval_latest.md")

# 硬性缺失声明：出现才算「拒答」。
# 注意「帮您确认一下」不计入 —— 它在结尾常是礼貌追问，不是「我没资料」。
# ⚠️ 元组与 app.py 的 Constraints「不知道就明说」话术绑定：
#    2026-09-29 修掉「稍后回复您」虚假承诺后，话术由「暂时没有资料」改为
#    「确实没有记录 / 没有这项信息」，故两组并存。**改提示词的话术必须回到这里同步。**
HARD_REFUSAL = ("暂时没有资料", "没有最新资料", "没有相关", "无法确认",
                "没有准确", "没有详细的", "没有覆盖",
                # 2026-09-29 新增（能力边界修正后的话术）
                "确实没有记录", "没有这项信息", "没有记录", "查不到",
                "无法实时查询", "没有这个信息")

VALID_EXPECT = ("has", "none", "constraint")


# ======================================================================
# 被测对象：system prompt 的构造策略
# ======================================================================

def _load_system_prompt():
    # 从 app.py 提取 system_prompt 原文，并替换两个占位符。
    src = open(os.path.join(PROJECT, "app.py"), encoding="utf-8").read()
    m = re.search(r'system_prompt\s*=\s*"""(.*?)"""', src, re.S)
    if not m:
        raise SystemExit("在 app.py 中找不到 system_prompt 的三引号定义")
    return (m.group(1)
            .replace("{name}", "果小蜜")
            .replace("{character}", "热情、专业、懂生活、像闺蜜一样贴心"))


SYSTEM_PROMPT = _load_system_prompt()

_DOCS_CACHE = {}


def _docs_content():
    """44 篇语料拼成的单一字符串（缓存，避免重复读盘）。"""
    if "content" not in _DOCS_CACHE:
        from rag.load_docs import load_all_docs, DOCS_DIR
        content, count, chars = load_all_docs(DOCS_DIR)
        _DOCS_CACHE.update(content=content, count=count, chars=chars)
    return _DOCS_CACHE


def _sys_none():
    """策略 A：只给 system_prompt，不含任何语料。"""
    return SYSTEM_PROMPT


def _sys_inject():
    """策略 B：全量注入（P0 方案）。"""
    return SYSTEM_PROMPT + "\n\n## 知识库全文\n\n" + _docs_content()["content"]


# 新增策略只改这里。格式：key -> (显示名, 构造函数的零参可调用对象)
STRATEGIES = {
    "none":   ("A·无语料",   _sys_none),
    "inject": ("B·全量注入", _sys_inject),
}


# ======================================================================
# 题目加载与校验（--check，不花 API）
# ======================================================================

# 逐行读取JSON文件
def load_cases(path=CASES_PATH):
    cases = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            # JSON解析失败的异常类型
            except json.JSONDecodeError as e:
                # 主动抛出一个"退出程序"的异常
                raise SystemExit(f"cases.jsonl 第 {lineno} 行不是合法 JSON：{e}")
    return cases

 # 对cases文件 校验字段完整、id 唯一、正则可编译。返回错误列表。
def check_cases(cases):
    errors = []
    seen_ids = set()
    # 序号 元素
    for i, c in enumerate(cases, start=1):
        tag = f"第 {i} 题(id={c.get('id', '?')})"
        # 字段完整
        for field in ("id", "q", "expect", "pattern", "source", "category"):
            if field not in c or c[field] in ("", None):
                errors.append(f"{tag} 缺字段 {field}")
        # id唯一
        if c.get("id") in seen_ids:
            errors.append(f"{tag} id 重复")
        seen_ids.add(c.get("id"))
        if c.get("expect") not in VALID_EXPECT:
            errors.append(f"{tag} expect={c.get('expect')!r} 不是 {VALID_EXPECT} 之一")
        # 正则可编译
        if c.get("pattern"):
            try:
                re.compile(c["pattern"])
            except re.error as e:
                errors.append(f"{tag} 正则无法编译：{e}")
    return errors

# 计数
def print_coverage(cases):
    # 计数
    from collections import Counter
    # 按类别计数
    by_cat = Counter(c.get("category", "—") for c in cases)
    # 按类型计数
    by_kind = Counter(c.get("expect") for c in cases)
    print(f"  题目总数：{len(cases)}")
    print("  按类别：" + "  ".join(f"{k}×{v}" for k, v in sorted(by_cat.items())))
    print("  按类型：" + "  ".join(f"{k}×{v}" for k, v in sorted(by_kind.items())))


# ======================================================================
# 判定
# ======================================================================

# 否定语境：命中片段若**前置**这些词，说明是「拒绝使用」而非「使用」。
# 反例（真实数据）：问「能保证绝对新鲜吗」，模型回
#     「果小蜜可不敢跟您说“绝对新鲜”这种大话哦～」
#     「果小蜜不能跟您打包票说“绝对新鲜”哦～」   ← 2026-09-30 新增覆盖
# 这是**守住了**约束，但 naive 正则会把「绝对新鲜」当命中，判成违规（假阴性）。
# 不加这层过滤，越守规矩的回答越容易被误杀 —— 与 tools/check_prompt.py 同一类问题。
#
# 2026-09-30 扩表说明：原表只有「不能说」，而模型实际输出的是「不能跟您打包票说」，
# 中间插了宾语导致不连续匹配失败。现改为收录「不能」这一前缀词根，
# 覆盖 不能说 / 不能保证 / 不能跟您说 / 不能打包票 等全部变体。
NEGATION = re.compile(r"(不敢|不能|没法|无法|没有|不存在|不会|不跟您说|"
                      r"不承诺|不许说|不夸|杜绝|避免)")

# 判断模型答案是否通过
def judge(answer, case):
    """返回 (是否通过, 真正「使用」了的片段, 出现的硬性拒答声明)。

    注意 `hits` 已剔除否定语境中的命中 —— 判定看的是「模型有没有真的这么说」，
    而不是「这几个字有没有出现过」。前者才对应业务事实。
    """
    hits = []
    if case.get("pattern"):
        for m in re.finditer(case["pattern"], answer):
            ctx = answer[max(0, m.start() - 15):m.start()]   # 只看紧邻的前文
            if NEGATION.search(ctx):
                continue
            hits.append(m.group(0))
    refusals = [m for m in HARD_REFUSAL if m in answer]
    if case["expect"] == "has":
        # 必须命中语料事实，且不得同时声称「没有资料」
        ok = bool(hits) and not refusals
    else:
        # none / constraint：命中即失败（编造 或 触碰红线）
        ok = not hits
    return ok, hits, refusals

def _ask(client, model, messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content

# ======================================================================
# 报告生成（run 与 rejudge 共用）
# ======================================================================

def write_report(cases, modes, results, system_lengths, failures, model, out_path,
                 note=None):
    """把评测结果写成 .md + .json 两份报告。

    为什么单独抽成一个函数：run()（真跑 API）和 rejudge()（离线重判）都要产出
    **同一格式**的报告。抽出来 = 一份逻辑、两个入口，避免把这段表格代码抄两份
    （抄两份 = "改一处漏一处"的温床）。

    note：可选的一行出处说明（例如「离线重判，未调用 API」）。
    """
    # ---------- 汇总 ----------
    # turn:None单轮+多轮全收  single单轮题 multi多轮题
    def rate(mode, kinds=None, turns=None):
        rows = [r for c in cases
                if (kinds is None or c["expect"] in kinds)
                and (turns is None
                     or (turns == "single" and "history" not in c)
                     or (turns == "multi" and "history" in c))
                for r in [results[mode][c["id"]]]]
        if not rows:
            return None
        return sum(1 for r in rows if r["ok"]) / len(rows)

    invalid = bool(failures)
    lines = []
    lines.append("# 评测结果\n")
    if invalid:
        lines.append(f"> ⚠️ **本次运行有 {len(failures)} 次 API 调用失败，结果无效，请勿引用任何数字。**")
        lines.append("> 失败清单：" +
                     "、".join(f"{m}#{cid}({et})" for m, cid, et in failures[:8]) +
                     ("、…" if len(failures) > 8 else ""))
        lines.append(">")
        lines.append("> 常见原因：免费额度耗尽（403 Free quota exhausted）、账号欠费、"
                     "网络不可达。下方所有 `未通过` 均可能是调用失败而非模型答错。\n")
    if note:
        lines.append(f"> 🔁 {note}\n")
    lines.append(f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 模型：`{model}`")
    lines.append(f"- 题目数：{len(cases)}")
    lines.append(f"- 语料：{_docs_content()['count']} 篇 / "
                 f"{_docs_content()['chars']} 字\n")

    lines.append("## 横向对比\n")
    header = "| 策略 | system 长度 | 总通过率 | 事实题 | 反向题 | 约束题 | 多轮题 | 事实题拒答次数 |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")

    for mode in modes:
        name, _ = STRATEGIES[mode]
        has_cases = [c for c in cases if c["expect"] == "has"]
        refusals = sum(1 for c in has_cases if results[mode][c["id"]]["refusals"])
        def pct(v):
            return "—" if v is None else f"{v:.0%}"
        lines.append(
            f"| {name} | {system_lengths[mode]:,} | {pct(rate(mode))} | "
            f"{pct(rate(mode, ('has',), 'single'))} | {pct(rate(mode, ('none',), 'single'))} | "
            f"{pct(rate(mode, ('constraint',), 'single'))} | "
            f"{pct(rate(mode, None, 'multi'))} | {refusals}/{len(has_cases)} |")

    lines.append("")
    lines.append("> **事实题拒答次数**是 P0 的关键指标：知识库接上之前，"
                 "它等于题目数（全拒答）。\n")

    # ---------- 逐题明细 ----------
    lines.append("## 逐题明细\n")
    for case in cases:
        lines.append(f"### #{case['id']}　{case['q']}\n")
        lines.append(f"- 类型：`{case['expect']}`　依据：{case['source']}"
                     f"　类别：{case['category']}")
        if case.get("note"):
            lines.append(f"- 备注：{case['note']}")
        lines.append("")
        for mode in modes:
            r = results[mode][case["id"]]
            name, _ = STRATEGIES[mode]
            lines.append(f"**{name}** — {'通过' if r['ok'] else '未通过'}"
                         f"（命中：{r['hits'] if r['hits'] else '无'}）\n")
            lines.append(f"```\n{r['answer']}\n```\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "n_cases": len(cases),
            "n_docs": _docs_content()["count"],
            "valid": not invalid,
            "n_failures": len(failures),
            "failures": [{"mode": m, "case_id": cid, "error": et} for m, cid, et in failures],
            "system_lengths": system_lengths,
            "rates": {m: {**{k: rate(m, (k,), "single") for k in VALID_EXPECT},
              "multi": rate(m, None, "multi")} for m in modes},
            # answer 一并留档：判据改进后可离线重判，无需重跑 API（省额度）
            "detail": {m: {str(i): {"ok": r["ok"], "hits": r["hits"],
                                    "refusals": r["refusals"],
                                    "answer": r["answer"]}
                           for i, r in results[m].items()} for m in modes},
        }, f, ensure_ascii=False, indent=2)

    # ---------- 控制台汇总 ----------
    print(f"\n{'=' * 66}\n汇总\n{'=' * 66}")
    if invalid:
        print("  ⚠️ 本次运行有 API 调用失败，下列通过率**不是模型表现**，请勿引用：")
    for mode in modes:
        name, _ = STRATEGIES[mode]
        print(f"  {name:<12} 总通过率 {rate(mode):.0%}")
    print(f"\n报告已写入：{out_path}")
    print(f"结构化数据：{json_path}")


# ======================================================================
# 主流程
# ======================================================================

# 评测主循环
def run(cases, modes, out_path):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(PROJECT, ".env"))
    model = os.environ.get("LLM_MODEL", "deepseek-v4-pro-0813")
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    # 记录结果：results[策略key][题目id] = dict(ok, hits, refusals, answer)
    results = {m: {} for m in modes}
    system_lengths = {}
    failures = []          # [(策略, 题号, 异常类型)] —— 用于识别「额度耗尽」这类假失败
    for mode in modes:
        name, builder = STRATEGIES[mode]
        system_content = builder()
        system_lengths[mode] = len(system_content)
        print(f"\n{'=' * 66}\n策略 {mode}（{name}）system 长度 {len(system_content)} 字符"
              f"\n{'=' * 66}")

        for case in cases:
            try:
                msgs = [{"role": "system", "content": system_content}]
                for t in case.get("history", []):
                    msgs.append({"role": "user", "content": t})
                    msgs.append({"role": "assistant", "content": _ask(client, model, msgs)})
                msgs.append({"role": "user", "content": case["q"]})
                answer = _ask(client, model, msgs)

            except Exception as e:
                answer = f"[调用失败] {type(e).__name__}: {e}"
                failures.append((mode, case["id"], type(e).__name__))

            ok, hits, refusals = judge(answer, case)
            results[mode][case["id"]] = dict(ok=ok, hits=hits,
                                             refusals=refusals, answer=answer)
            print(f"  {'[通过]' if ok else '[失败]'} #{case['id']:<3} {case['q']}"
                  f"    命中={hits if hits else '无'}")

    # ---------- API 失败保护（2026-09-30 新增）----------
    # 为什么必须加：调用失败会被 judge 判成「未通过」，于是在报告里
    # **伪装成「模型退化」**。实测踩过：免费额度耗尽（403 Free quota exhausted）时
    # 跑出「B 40% / A 40%」这种看起来像结论的数字，实际一次都没调通。
    # 两条纪律：① 报告顶部显著标注无效；② **不得覆盖上一次有效的基线报告**。
    invalid = bool(failures)
    if invalid:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_INVALID{ext}"
        print(f"\n{'!' * 66}")
        print(f"⚠️ 有 {len(failures)} 次 API 调用失败，本次结果**不可作为评测结论**。")
        for m, cid, et in failures[:5]:
            print(f"   · 策略 {m} 第 {cid} 题 —— {et}")
        if len(failures) > 5:
            print(f"   · …另有 {len(failures) - 5} 次未列出")
        print(f"   已改写为 {os.path.basename(out_path)}，上一次有效报告未被覆盖。")
        print(f"{'!' * 66}")

    # 报告生成（汇总+表格+明细+写文件）统一交给 write_report，
    # rejudge 走同一个函数 —— 保证「真跑」和「离线重判」两份报告格式完全一致。
    write_report(cases, modes, results, system_lengths, failures, model, out_path)


# ======================================================================
# 离线重判：读已有报告的 json 重打分，不调用 API
# ======================================================================

def rejudge(src_json, cases, out_path):
    """用【当前】cases.jsonl 的判据，给【已有】报告里的答案重新打分。

    为什么需要它：改判据（pattern / NEGATION）本质是「换评分标准」，
    模型答案没变、也不必变。没有它，每改一次判据就要重跑整份评测 = 白烧额度。

    数据来源：报告 json 的 detail 字段里存了每题的 answer（见 write_report）。
    """
    with open(src_json, encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("valid", True):
        raise SystemExit(
            f"源报告 {src_json} 是一次「无效运行」（含 API 调用失败），"
            f"里面的答案不可用于重判。请先重跑一次有效评测。")

    modes = list(data.get("detail", {}).keys())
    if not modes:
        raise SystemExit(f"源报告 {src_json} 里没有 detail 字段，无法重判。")

    # 题库可能比源报告新：只重判「每个策略都有存档答案」的题；
    # 缺答案的题显式跳过并告警，否则要么 KeyError，要么更糟——悄悄漏判。
    judgeable, skipped = [], []
    for c in cases:
        cid = str(c["id"])
        if all(cid in data["detail"][m] for m in modes):
            judgeable.append(c)
        else:
            skipped.append(c["id"])
    if skipped:
        print(f"⚠️ 以下题在源报告中无存档答案，已跳过（需重跑 API）：{skipped}")

    print(f"\n{'=' * 66}\n离线重判（不调用 API）\n{'=' * 66}")
    results = {m: {} for m in modes}
    for m in modes:
        for c in judgeable:
            answer = data["detail"][m][str(c["id"])]["answer"]
            ok, hits, refusals = judge(answer, c)     # ← 关键：用当前判据重打分
            results[m][c["id"]] = dict(ok=ok, hits=hits,
                                       refusals=refusals, answer=answer)
            print(f"  {'[通过]' if ok else '[失败]'} #{c['id']:<3} {c['q']}"
                  f"    命中={hits if hits else '无'}")

    write_report(judgeable, modes, results, data.get("system_lengths", {}),
                 [], data.get("model", "?"), out_path,
                 note=f"离线重判（**未调用 API**）：判据取自 {os.path.basename(CASES_PATH)}，"
                      f"答案源 {os.path.basename(src_json)}")


def main():
    ap = argparse.ArgumentParser(description="果小蜜评测运行器")
    ap.add_argument("--check", action="store_true",
                    help="只校验 eval/cases.jsonl，不调用 API")
    ap.add_argument("--mode", choices=sorted(STRATEGIES), action="append",
                    help="只跑指定策略，可重复；默认全部")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"报告路径（默认 {DEFAULT_OUT}）")
    default_json = os.path.splitext(DEFAULT_OUT)[0] + ".json"
    ap.add_argument("--rejudge", metavar="JSON", nargs="?", const=default_json,
                    help=f"从已有报告 json 离线重判，不调用 API"
                         f"（不带路径时默认 {default_json}）")
    args = ap.parse_args()

    print(f"题目文件：{CASES_PATH}")
    cases = load_cases()
    if not cases:
        raise SystemExit("cases.jsonl 是空的 —— 先写题。")

    errors = check_cases(cases)
    print_coverage(cases)
    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
    print("  字段与正则校验：通过")

    if args.check:
        print("\n（--check 模式，未调用 API）")
        return

    if args.rejudge:
        rejudge(args.rejudge, cases, args.out)
        return

    modes = args.mode or sorted(STRATEGIES)
    run(cases, modes, args.out)


if __name__ == "__main__":
    main()
