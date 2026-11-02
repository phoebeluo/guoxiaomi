"""零成本检查：验证 system_prompt 中不再含「指向不存在能力」的表述。

背景
----
app.py 的 system_prompt 曾经假设了三套系统不具备的能力：
  1. 检索层      —— "必须先检索到该品种的当季信息"
  2. 工具调用    —— "交由工具获取真实数据"
  3. 人工核实通道 —— "帮你确认一下，稍后回复您"
2026-09-29 修正为新增 `## Capability Boundary` 段。

2026-09-30 二次修正
-------------------
原 Capability Boundary 只声明「模型没有客服能力」，但知识库里**真实存在**
在线客服渠道（`服务-客服渠道与工作时间.md`）。两者并列会让模型把
「我做不到」错说成「我们没有」，是另一个方向的错误。

现改为区分两个概念：
  - 「你做不到」→ 应当**指路**（告知顾客他自己能走的路径）
  - 「业务没有」→ 才该直说没有
因此检查项也相应更新，并新增对「代客承诺话术」的拦截。

用法
----
    python tools/check_prompt.py

不调用 API，只做字符串检查。改提示词后跑一遍即可。
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
# 2026-11-02 起提示词唯一来源是 rag/prompt.py（原先从 app.py 内联字符串抽取）
PROMPT_FILE = os.path.join(PROJECT, "rag", "prompt.py")

# 必须消失的表述：它们指向系统没有的能力，或诱导模型代客承诺
FORBIDDEN = [
    "稍后回复您",            # 无回访机制
    "帮你确认一下",          # 无核实通道
    "帮您确认一下",          # 同上
    "稍后给您答复",          # 无对外联络
    "联系物流核实",          # 无对外联络
    "交由工具获取真实数据",  # 无工具调用
    "必须先检索",            # 无检索层（P0 是全文注入）
    "已为您提交专人跟进",    # 代客承诺：无提交通道
    "已记录，明天跟进",      # 代客承诺：无回访机制
    "转接人工",              # 代客承诺：无法代为转接
]

# 必须存在的表述：能力边界声明与新增规则
REQUIRED = [
    "## Capability Boundary",
    "「你做不到」",              # 与「业务没有」的区分
    "「业务没有」",
    "指路",                      # 允许且鼓励的做法
    "无代办通道",
    # 2026-11-02（W9）：原文是「如实告知你无法实时查询」，工具上线后订单状态可查，
    # 边界表述改为「如实告知无法查询」+ 单独说明订单状态走工具，故锚点同步更新。
    "如实告知无法查询",
    "以商品页当期标价为准",      # 报价必须标注参考价
]

# 否定/示例标记：出现在同一行即认为该表述属于「反例」，不算裸出现
# 注意「无法」必须在列：Capability Boundary 里「你无法代客执行动作（提交工单、转接人工…）」
# 是一句合法的否定陈述，漏掉这个标记会把正确内容误报成残留。
NEG_MARKERS = ("严禁", "不得", "不可", "不能", "无法", "杜绝", "禁止", "绝不",
               "不允许", "没法", "❌")


def main():
    src = open(PROMPT_FILE, encoding="utf-8").read()
    m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', src, re.S)
    if not m:
        print("✗ 在 rag/prompt.py 中找不到 SYSTEM_PROMPT 的三引号定义")
        return 1
    sp = m.group(1)

    print(f"system_prompt 长度：{len(sp)} 字符\n")

    ok = True

    # 关键：提示词里「严禁说『稍后回复您』」中的字符串是**反例**，不是模板。
    # 因此不能只判断「字符串在不在」，必须看它的语义角色。
    # 第一版脚本漏了这层区分，把 4 处正确的反面示例误报成残留。
    #
    # 第二版（当前）改用**按行判定**：只要该表述所在的整行含否定/示例标记，
    # 就认为它是刻意写出的反例。比原来的字符窗口更精确——
    # 字符窗口既可能误判（❌ 示例离得远），也可能漏判（远处的否定词巧合命中）。
    lines = sp.splitlines()

    def naked_hits(s):
        out = []
        for ln in lines:
            if s not in ln:
                continue
            if any(mk in ln for mk in NEG_MARKERS):
                continue
            out.append(ln.strip())
        return out

    print("① 指向不存在能力的表述（须仅作为反例出现）")
    naked_all = []
    for s in FORBIDDEN:
        occurrences = sp.count(s)
        if occurrences == 0:
            print(f"   ✓ {s} —— 未出现")
            continue
        naked = naked_hits(s)
        if naked:
            ok = False
            naked_all.append(s)
            print(f"   ✗ {s} —— {len(naked)} 行裸出现（同行无否定/示例标记）")
            for ln in naked[:2]:
                print(f"       → {ln[:70]}")
        else:
            print(f"   ✓ {s} —— {occurrences} 处，均在否定或示例语境中（反例，正确）")
    if naked_all:
        print("\n   ⚠️ 裸出现的表述会被模型当成模板照搬，必须删掉或改写为反例。")

    print("\n② 应存在的表述（能力边界声明与新增规则）")
    for s in REQUIRED:
        hit = s in sp
        ok = ok and hit
        print(f"   {'✓' if hit else '✗'} {s}")

    print("\n③ 占位符完整性（应各出现 1 次，供 .replace 使用）")
    for ph in ("{name}", "{character}"):
        n = sp.count(ph)
        good = n == 1
        ok = ok and good
        print(f"   {'✓' if good else '✗'} {ph} 出现 {n} 次")

    print("\n" + ("全部通过" if ok else "存在问题，见上方 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
