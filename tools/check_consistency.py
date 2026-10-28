"""口径自检（L0-5）：文档、报告、实际三方数字是否一致。

为什么需要
----------
本项目所有结论都建立在"数字可追溯"上。但数字散落在至少四处：
  ① 实际数据（`data/docs/`、`eval/cases.jsonl`、`app.py`）
  ② 文档声明（总纲「事实底座」、`eval/README.md`、计划）
  ③ 评测报告（`eval/reports/*.md` 头部）
  ④ 运行配置（`.env` 的 `LLM_MODEL`）
任何一处漂移，简历上引用的数字就可能是错的 —— 而且**不会报错**，只会静默过期。
本脚本把"人工核对"变成"一条命令"。

检查项（全部只读、零 API、零新依赖）
------------------------------------
1. **语料**：`data/docs/` 实测 篇数/字符  vs  总纲事实底座 vs 最新报告头部
2. **评测集**：`cases.jsonl` 实测 题数/计入分母  vs  `eval/README.md` vs 报告头部
3. **system 长度**：`app.py` 提示词 + 全量语料 实测  vs  报告头部
4. **代码规模**：`app.py` 实测行数  vs  总纲声明
5. **模型口径**：`.env` 的 `LLM_MODEL`  vs  最新报告头部（允不一致，但必须能被解释）

⚠️ 只报警、不修改。发现不一致时，**先确认哪个是错的再动手**。

用法
----
    E:/guoxiaomi/.venv/Scripts/python.exe tools/check_consistency.py
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

DOCS_DIR = os.path.join(PROJECT, "data", "docs")
CASES_PATH = os.path.join(PROJECT, "eval", "cases.jsonl")
APP_PATH = os.path.join(PROJECT, "app.py")
ENV_PATH = os.path.join(PROJECT, ".env")
README_PATH = os.path.join(PROJECT, "eval", "README.md")
MASTER_PATH = os.path.join(PROJECT, "docs", "果小蜜-优化路线与面试总纲.md")

problems, notes = [], []


def note(msg):
    notes.append(msg)
    print(f"  · {msg}")


def problem(msg):
    problems.append(msg)
    print(f"  ❌ {msg}")


# ---------------------------------------------------------------- 实测真值
def truth_docs():
    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    chars = sum(len(open(p, encoding="utf-8").read()) for p in files)
    return len(files), chars


def truth_cases():
    total = 0
    counted = 0
    with open(CASES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            total += 1
            if c.get("counted", True):
                counted += 1
    return total, counted


def truth_system_chars():
    """复现 app.py 的 system 消息长度：提示词（替换占位符）+ 分隔串 + 全量语料。"""
    from rag.load_docs import load_all_docs, DOCS_DIR as D
    src = open(APP_PATH, encoding="utf-8").read()
    m = re.search(r'system_prompt\s*=\s*"""(.*?)"""', src, re.S)
    if not m:
        return None
    prompt = (m.group(1)
              .replace("{name}", "果小蜜")
              .replace("{character}", "热情、专业、懂生活、像闺蜜一样贴心"))
    content, _, _ = load_all_docs(D)
    return len(prompt) + len("\n\n## 知识库全文\n\n") + len(content)


def truth_app_lines():
    return len(open(APP_PATH, encoding="utf-8").read().splitlines())


def env_model():
    if not os.path.exists(ENV_PATH):
        return None
    for line in open(ENV_PATH, encoding="utf-8"):
        line = line.strip()
        if line.startswith("LLM_MODEL="):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------- 文档/报告里的声明值
def _find_int(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def _find_int_comma(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else default


def latest_report():
    """取最新报告 = 按修改时间最新的 eval/reports/*.md（排除 INVALID）。"""
    cands = [p for p in glob.glob(os.path.join(PROJECT, "eval", "reports", "*.md"))
             if "INVALID" not in os.path.basename(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def report_facts(path):
    """从报告头部抽 语料篇数/字数、题目数、计入分母、模型、system 长度。"""
    if not path or not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8").read()
    head = text[:1500]
    facts = {}
    m = re.search(r"语料：\s*(\d+)\s*篇\s*/\s*([\d,]+)\s*字", head)
    if m:
        facts["docs_count"] = int(m.group(1))
        facts["docs_chars"] = int(m.group(2).replace(",", ""))
    facts["cases_total"] = _find_int(r"题目数：\s*(\d+)", head)
    facts["cases_counted"] = _find_int(r"计入通过率：\s*\*\*(\d+)\s*题\*\*", head)
    m = re.search(r"模型：`([^`]+)`", head)
    if m:
        facts["model"] = m.group(1)
    facts["report_path"] = os.path.relpath(path, PROJECT)
    return facts


def master_facts():
    """从总纲「事实底座」抽几个稳定声明的数字。"""
    if not os.path.exists(MASTER_PATH):
        return {}
    text = open(MASTER_PATH, encoding="utf-8").read()
    facts = {}
    m = re.search(r"\*\*45 篇 / ([\d,]+) 字\*\*", text)
    if m:
        facts["docs_chars"] = int(m.group(1).replace(",", ""))
    m = re.search(r"`app\.py` \*\*(\d+) 行\*\*", text)
    if m:
        facts["app_lines"] = int(m.group(1))
    m = re.search(r"\*\*(\d+) 题\*\*（\d+ 单轮", text)
    if m:
        facts["cases_total"] = int(m.group(1))
    return facts


def readme_facts():
    if not os.path.exists(README_PATH):
        return {}
    text = open(README_PATH, encoding="utf-8").read()
    return {
        "cases_total": _find_int(r"### 覆盖率矩阵.*?\*\*按类别（共 (\d+) 题）\*\*", text)
        or _find_int(r"共 (\d+) 题", text),
        "cases_counted": _find_int(r"= \*\*(\d+) 题计入通过率\*\*", text),
    }


def main():
    print("=" * 66)
    print("口径自检：实测 vs 文档 vs 报告")
    print("=" * 66)

    d_count, d_chars = truth_docs()
    c_total, c_counted = truth_cases()
    sys_chars = truth_system_chars()
    app_lines = truth_app_lines()
    model = env_model()
    rep_path = latest_report()
    rep = report_facts(rep_path)
    mas = master_facts()
    rdm = readme_facts()

    print(f"\n[实测真值]\n  data/docs      : {d_count} 篇 / {d_chars:,} 字")
    print(f"  cases.jsonl    : {c_total} 题（计入分母 {c_counted}）")
    print(f"  app.py         : {app_lines} 行")
    print(f"  system 消息    : {sys_chars:,} 字符" if sys_chars else "  system 消息    : 计算失败")
    print(f"  .env LLM_MODEL : {model}")

    print(f"\n[1] 语料篇数/字数")
    if rep.get("docs_count") is not None and rep["docs_count"] != d_count:
        problem(f"报告 {rep['report_path']} 写 {rep['docs_count']} 篇，实测 {d_count} 篇")
    elif rep.get("docs_count") is not None:
        note(f"报告篇数与实测一致（{d_count} 篇）")
    if rep.get("docs_chars") is not None and rep["docs_chars"] != d_chars:
        problem(f"报告 {rep['report_path']} 写 {rep['docs_chars']:,} 字，实测 {d_chars:,} 字")
    elif rep.get("docs_chars") is not None:
        note(f"报告字数与实测一致（{d_chars:,} 字）")
    if mas.get("docs_chars") is not None and mas["docs_chars"] != d_chars:
        problem(f"总纲事实底座写 {mas['docs_chars']:,} 字，实测 {d_chars:,} 字")

    print(f"\n[2] 评测集题数/分母")
    for label, val in (("报告", rep.get("cases_total")), ("eval/README", rdm.get("cases_total"))):
        if val is not None and val != c_total:
            problem(f"{label} 写 {val} 题，实测 {c_total} 题")
    for label, val in (("报告", rep.get("cases_counted")), ("eval/README", rdm.get("cases_counted"))):
        if val is not None and val != c_counted:
            problem(f"{label} 写分母 {val}，实测 {c_counted}")
    if mas.get("cases_total") is not None and mas["cases_total"] != c_total:
        problem(f"总纲写 {mas['cases_total']} 题，实测 {c_total} 题")
    if not problems:
        note(f"题数/分母三处一致（{c_total} / {c_counted}）")

    print(f"\n[3] system 消息长度")
    m = re.search(r"\|\s*B·全量注入\s*\|\s*([\d,]+)\s*\|", open(rep_path, encoding="utf-8").read()
                  if rep_path else "")
    if m and sys_chars:
        rep_sys = int(m.group(1).replace(",", ""))
        if rep_sys != sys_chars:
            problem(f"报告 B·全量注入 system 为 {rep_sys:,}，实测 {sys_chars:,}")
        else:
            note(f"报告 system 长度与实测一致（{sys_chars:,}）")

    print(f"\n[4] 代码规模")
    if mas.get("app_lines") is not None and mas["app_lines"] != app_lines:
        problem(f"总纲写 app.py {mas['app_lines']} 行，实测 {app_lines} 行")

    print(f"\n[5] 模型口径")
    if rep.get("model") and model and rep["model"] != model:
        note(f"⚠️ 最新报告用 `{rep['model']}`，而 .env 现为 `{model}` —— "
             f"**这不是错误**（跑完可以换），但引用该报告时必须写明模型+日期")
    elif rep.get("model"):
        note(f".env 与最新报告模型一致（{model}）")
    if model and "-20" not in model and model.count("-") < 2:
        note(f"⚠️ `{model}` 看起来是浮动别名而非带日期快照 —— 报告将不可复现（见总纲红线）")

    print("\n" + "=" * 66)
    if problems:
        print(f"结果：发现 {len(problems)} 处不一致 —— 需人工确认哪一方是错的")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"结果：无硬性不一致（{len(notes)} 条提示）")
    print("=" * 66)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
