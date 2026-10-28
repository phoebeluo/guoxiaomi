"""题型分类一致性检查（零成本，不调用 API）

============================================================
这个脚本干什么
============================================================
它给评测集 `eval/cases.jsonl` 里的每一道题做三件事：

  1. 检查有没有标 `kind` 字段、值合不合法（只能是 fact/gap/boundary/rule）
  2. 统计四类题各有几道
  3. 交叉检查：`kind`（失败时该修什么）和 `expect`（怎么判定）搭不搭

它不联网、不花钱、跑一次 1 秒。改完题随手跑。


============================================================
为什么要给每道题标「类型」（kind）
============================================================
原来每道题只有 `expect`（has / none / constraint），它回答的是：
    「这道题**怎么判定**通过还是失败？」

但它没回答一个更本质的问题：
    「这道题**失败了，该去修什么**？」

这两个问题的答案**不一样**。举个例子：
  · 模型答错了价格       → 该去改提示词（它没好好读语料）
  · 语料里压根没有快递信息 → 该去【补语料】（不是模型的错）

如果都当成"改提示词"，第二种就会修错地方——
这正是 `#4`（快递）这道题暴露的问题。

所以补一个字段 `kind`，专门记「失败时该修什么」。


============================================================
kind 的四个取值
============================================================
  fact     A·事实覆盖 ｜ 语料里有答案，模型必须答对   → 失败就改提示词
  gap      B·知识盲区 ｜ 语料里没有，但答案客观存在   → 失败就【补语料】
  boundary C·能力边界 ｜ 答案客观拿不到（要实时接口）  → 必须拒答 + 指路
  rule     D·约束红线 ｜ 行为规范，不得触碰           → 失败就改提示词


============================================================
counted：这道题算不算进「通过率」
============================================================
有些题（如 #7「榴莲热量多少」）的答案属于**模型自带的常识**，
即使一句语料都不给（A 策略）也能答对。

这类题混进分母会产生一个假象：**通过率很高，但高出来的那部分
跟「知识库接上了没有」毫无关系**——等于拿模型本来就有的本事
给你的优化记功。

所以约定：
  counted = true（默认，可不写）→ 计入通过率，分数能反映知识库贡献
  counted = false             → 【不计入】任何通过率，单独列为「对照组」

它们不是废题，是**校准项**：告诉你 A/B 两个策略的差值里，
有多少本来就是模型自己会的东西。


============================================================
怎么运行
============================================================
    E:/guoxiaomi/.venv/Scripts/python.exe tools/check_kinds.py
"""

import json
import os

# ---------------------------------------------------------------
# 路径：从本文件位置反推项目根目录（不写死绝对路径，换机器不会崩）
# 本文件在  <项目根>/tools/check_kinds.py
#   __file__        = .../tools/check_kinds.py
#   dirname 一次    = .../tools
#   dirname 两次    = ...（项目根）
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(BASE_DIR, "eval", "cases.jsonl")


# ---------------------------------------------------------------
# 四个类型的中文名（给人在屏幕上看的）
# 代码内部一律用英文小写比对——这样避免中文全角/空格带来的隐形坑
# （本项目已经被"全角～"坑过一次，见 eval/README.md）
# ---------------------------------------------------------------
KINDS = {
    "fact":     "A·事实覆盖（语料有 → 失败改提示词）",
    "gap":      "B·知识盲区（语料无但客观存在 → 失败补语料）",
    "boundary": "C·能力边界（客观拿不到 → 必须拒答指路）",
    "rule":     "D·约束红线（行为规范 → 失败改提示词）",
}

# ---------------------------------------------------------------
# 常规搭配表：expect（怎么判） ↔ kind（失败归因）
#
#   has        → fact           语料有，必须命中
#   none       → gap / boundary 语料无：要么该补（gap），要么该拒（boundary）
#   constraint → rule           行为红线
#
# 不在这张表里的组合会打印「⚠️ 提示」，但**不会中止**——
# 因为组合本身可能是你有意的设计，脚本只负责提醒，不替你做决定。
# ---------------------------------------------------------------
ALLOWED = {
    "has":        {"fact"},
    "none":       {"gap", "boundary"},
    "constraint": {"rule"},
}

# ---------------------------------------------------------------
# 已备案例外：不在上表里，但**确认是有意为之**，不再每次报警
#
# 为什么要一张白名单：校验器如果对已知的有意设计反复报警，
# 你就会开始无视它 —— 一个被无视的校验器等于没有校验器。
# 所以「真异常」和「已知例外」必须分开输出。
# ---------------------------------------------------------------
KNOWN_EXCEPTIONS = {
    ("none", "rule"):
        "#15 测的是「不得承诺代办」这条**行为红线**，本该记 constraint；"
        "但判据逻辑上 none 与 constraint 完全相同（都是 ok = not hits），"
        "只是统计分栏不同 → 用户确认不改，作为已知例外备案",
}


def load_cases(path):
    """把 cases.jsonl 一行一行读成字典列表（jsonl = 一行一个 JSON）。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue                      # 跳过空行
            cases.append(json.loads(line))    # 每行解析成一个字典
    return cases


def main():
    cases = load_cases(CASES_PATH)

    print(f"题目文件：{CASES_PATH}")
    print(f"题目总数：{len(cases)}")
    print("=" * 66)

    errors = []      # 真错误：字段缺失、值非法
    warns = []       # 提示：组合非常规、可能稀释指标
    noted = []       # 已备案：有意为之的例外，不再是问题

    # ---------- 第一遍：逐题检查字段 ----------
    for c in cases:
        cid = c.get("id", "?")
        kind = c.get("kind")

        # 1) kind 必须在，且值合法
        if kind is None:
            errors.append(f"#{cid} 缺少 kind 字段")
            continue
        if kind not in KINDS:
            errors.append(f"#{cid} 的 kind='{kind}' 不是合法值"
                          f"（只能是 {list(KINDS)}）")
            continue

        # 2) counted 若出现，必须是布尔值
        #    为什么查这个：如果你手滑写成 "counted": "false"（带引号），
        #    Python 里非空字符串是**真值** → 过滤条件 c.get("counted", True)
        #    会得到 "false"（真）→ 这道题**静默地**仍被计入分母，
        #    而你以为已经排除了。这类"看着对、其实没生效"的坑本项目踩过多次。
        counted = c.get("counted", True)
        if not isinstance(counted, bool):
            errors.append(f"#{cid} 的 counted={counted!r} 不是布尔值"
                          f"（应写 true/false 且不带引号）")

        # 3) 交叉检查：kind 和 expect 搭不搭
        expect = c.get("expect")
        allowed = ALLOWED.get(expect, set())
        if allowed and kind not in allowed:
            if (expect, kind) in KNOWN_EXCEPTIONS:
                # 已知且确认过的例外 —— 单独归档，不再当成待解决的问题
                noted.append(f"#{cid} expect='{expect}' kind='{kind}'"
                             f" —— {KNOWN_EXCEPTIONS[(expect, kind)]}")
            else:
                warns.append(
                    f"#{cid} expect='{expect}' 与 kind='{kind}' 非常规搭配"
                    f"（expect='{expect}' 通常配 {sorted(allowed)}）"
                )

    # ---------- 统计 ----------
    from collections import Counter
    counter = Counter(c.get("kind") for c in cases)

    print("\n【四类题数】")
    for k, label in KINDS.items():
        n = counter.get(k, 0)
        ctrl = sum(1 for c in cases
                   if c.get("kind") == k and not c.get("counted", True))
        suffix = f"  ← 其中对照组 {ctrl} 题，计入分母 {n - ctrl} 题" if ctrl else ""
        print(f"  {k:<9} {label:<48} {n} 题{suffix}")

    print("\n【按 expect 交叉】")
    cross = Counter((c.get("expect"), c.get("kind")) for c in cases)
    for (expect, kind), n in sorted(cross.items(), key=lambda x: str(x[0])):
        print(f"  expect={expect:<10} kind={kind:<9} {n} 题")

    # ---------- 分母口径：计入 vs 对照组 ----------
    # 这是本脚本的第二条主线：不仅看「题的分类对不对」，还要看
    # 「哪些题会被算进通过率」。两者混为一谈，是评测里最常见的自欺方式。
    graded = [c for c in cases if c.get("counted", True)]
    control = [c for c in cases if not c.get("counted", True)]

    print("\n【是否计入通过率】")
    print(f"  计入分母（counted=true 或未写）：{len(graded)} 题")
    print(f"  对照组（counted=false）：      {len(control)} 题")
    for c in control:
        print(f"      · #{c['id']} {c['q']}")

    # ---------- 输出结果 ----------
    print("\n" + "=" * 66)
    if errors:
        print(f"❌ 发现 {len(errors)} 个错误：")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✅ 字段与取值校验：通过（kind 合法、counted 均为布尔值）")

    if noted:
        print(f"\n📌 {len(noted)} 条已备案例外（有意为之，无需处理）：")
        for item in noted:
            print(f"   - {item}")

    if warns:
        print(f"\n⚠️ {len(warns)} 条提示（不阻断，供你人工判断）：")
        for w in warns:
            print(f"   - {w}")
    else:
        print("✅ 搭配检查：无异常")

    print("=" * 66)
    print("（--check 类脚本，未调用 API）")


if __name__ == "__main__":
    main()
