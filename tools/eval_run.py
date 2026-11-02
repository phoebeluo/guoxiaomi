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

# 额外语料目录（2026-10-28 新增，W4 规模实验用）
# 为什么不在 data/docs 里加文件：`data/docs/` 是评测基准，一动 45 篇的基线与
# 历史报告全部作废（eval/README「坑」第一条）。所以外部语料留在 data/raw/，
# 由这个参数**在运行时**拼进「全量注入」策略 —— 45 篇档不传即为基线，行为完全不变。
_EXTRA_DIRS = []


def _docs_content():
    """全量语料拼成的单一字符串（缓存，避免重复读盘）。"""
    if "content" not in _DOCS_CACHE:
        from rag.load_docs import load_all_docs, DOCS_DIR
        content, count, chars = load_all_docs(DOCS_DIR)
        for extra in _EXTRA_DIRS:
            more, n2, ch2 = load_all_docs(extra)
            content = content + "\n\n" + more
            count += n2
            chars += ch2
        _DOCS_CACHE.update(content=content, count=count, chars=chars)
    return _DOCS_CACHE


# ----------------------------------------------------------------------
# A2 注入策略 A/B 用的辅助（2026-10-12）
# 做法：拿 cases.jsonl 的 source 字段当 **oracle**，模拟「理想检索恰好命中
#       该题依据的那篇」，与「全量注入」对照 —— 把「要不要上检索」变成数字。
# ⚠️ 这是 proxy（代理指标），**不是**检索实验：它测的是「注入内容」，
#    链路上不含任何检索器。引用时不许说成「我做了检索实验」。
# ----------------------------------------------------------------------

_GOLD_DOC_CACHE = {}   # 文件名 -> 已渲染片段，避免逐题重复读盘
_NOISE3 = []           # 固定噪声篇目，由 _init_noise(cases) 在 run() 开头填充


def _all_doc_names():
    """data/docs 下的全部 .md 文件名（排序，保证可复现）。"""
    from rag.load_docs import DOCS_DIR
    return sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".md"))


def _docs_by_names(names):
    """按**文件名**读取指定篇目并按 load_docs 的格式拼接。

    为什么必须复用同一格式（每篇以「【来源：标题】」开头、篇间空行分隔）：
    A/B 对照要求两组 system 除「注入哪几篇」外完全一致 ——
    格式差一点，落差就无法归因到「注入内容」上。

    2026-10-28 改：支持 `--extra-dir`。原先只查 `data/docs`，一旦开了扩展语料，
    BM25 检索到的外部篇目就取不到正文（KeyError）。查找顺序 = 主语料优先，
    与 `_docs_content()` 的拼接顺序保持一致。
    """
    from rag.load_docs import DOCS_DIR
    search_dirs = [DOCS_DIR] + list(_EXTRA_DIRS)
    parts = []
    for name in names:
        if name not in _GOLD_DOC_CACHE:
            for d in search_dirs:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        _GOLD_DOC_CACHE[name] = f"【来源：{name[:-3]}】\n{f.read()}"
                    break
            else:
                raise FileNotFoundError(f"找不到篇目：{name}（已查 {search_dirs}）")
        parts.append(_GOLD_DOC_CACHE[name])
    return "\n\n".join(parts)


def _init_noise(cases):
    """选定 3 篇「不被任何题 source 引用」的文档作固定噪声。

    为什么必须「未被任何题引用」：噪声只负责模拟 top-k 里的无关命中、
    顺便把 token 撑到真实检索的量级。若顺手带入别题答案，
    gold 就从「理想检索」变成「超额注入」，对照失效。
    """
    global _NOISE3
    used = set()
    for c in cases:
        # 2026-10-15：金标篇目也纳入「已占用」——否则噪声篇可能正好命中别题答案，
        # 让 oracle 从「理想检索」变成「超额注入」，对照失效。
        used.update(c.get("gold_sources") or [])
        used.update(re.findall(r"[\w\u4e00-\u9fa5-]+\.md", c.get("source", "")))
    _NOISE3 = [n for n in _all_doc_names() if n not in used][:3]
    return _NOISE3


def _gold_names(case):
    """理想检索的「命中结果」：该题金标篇目 + 固定噪声（总篇数至少 3）。

    2026-10-15 改：**优先使用 `gold_sources` 列表**（支持多答案题注入多篇），
    无该字段时回退到旧的「从 `source` 描述里正则抽 1 篇」逻辑 —— 兼容旧题库格式。
    `rule` / `boundary` 类题 `gold_sources` 为空 → 全部由噪声补足 3 篇，
    使每题注入篇数下限恒为 3，token 量稳定、可比。
    """
    valid = set(_all_doc_names())
    if "gold_sources" in case:
        gold = case.get("gold_sources") or []
    else:
        gold = re.findall(r"[\w\u4e00-\u9fa5-]+\.md", case.get("source", ""))
    picked = [n for n in dict.fromkeys(gold) if n in valid]
    for n in _NOISE3:
        if len(picked) >= 3:
            break
        if n not in picked:
            picked.append(n)
    return picked


def _sys_none(case=None):
    """策略 A：只给 system_prompt，不含任何语料。"""
    return SYSTEM_PROMPT


def _sys_inject(case=None):
    """策略 B：全量注入（P0 方案）。"""
    return SYSTEM_PROMPT + "\n\n## 知识库全文\n\n" + _docs_content()["content"]


def _sys_inject_gold(case=None):
    """策略 C：理想检索（oracle）—— 只注入该题依据的那 1 篇 + 2 篇噪声。

    ⚠️ 它假设检索**永远命中正确文档**，是真实检索的**上界**（真实检索做不到）。
    故：连上界都掉分 → 真实检索必然更差；上界持平 → 检索至少不损失效果。
    """
    names = _gold_names(case) if case else list(_NOISE3)
    return SYSTEM_PROMPT + "\n\n## 知识库全文\n\n" + _docs_by_names(names)


_BM25_CACHE = {}


def _bm25_index():
    """语料 BM25 索引（按「主语料 + 扩展目录」组合缓存）。纯本地计算，零 API。"""
    key = tuple(_EXTRA_DIRS)
    if key not in _BM25_CACHE:
        import glob
        from rag.bm25 import BM25
        from rag.load_docs import DOCS_DIR
        docs = []
        for d in [DOCS_DIR] + list(_EXTRA_DIRS):
            for p in sorted(glob.glob(os.path.join(d, "*.md"))):
                with open(p, encoding="utf-8") as f:
                    docs.append((os.path.basename(p), f.read()))
        _BM25_CACHE[key] = BM25(docs)
    return _BM25_CACHE[key]


def _sys_bm25(case=None):
    """策略 D：**真实检索** —— BM25 取 top-3 篇注入（线上真实路径）。

    与 C（理想检索）的差别只有一个：**注入哪 3 篇由检索器决定，不再由金标决定**。
    两者格式完全一致、篇数恒为 3，故 `C − D` 的差值可干净归因到「检索器」。
    这是本项目第一条真正归因到检索的数字，而检索本身是本地计算、**零 API**。

    为什么补齐到 3 篇：C 策略恒注入 3 篇，若 D 只检索到 2 篇，system 长度差
    会变成混杂变量（少一篇也可能少一段答案）。补齐后两策略篇数对齐。
    """
    if not case:
        return SYSTEM_PROMPT + "\n\n## 知识库全文\n\n" + _docs_by_names(list(_NOISE3))
    q = case["q"]
    if case.get("history"):
        q = " ".join(case["history"]) + " " + q
    hits = [d for d, _ in _bm25_index().search(q, top_k=3)]
    for n in _NOISE3:
        if len(hits) >= 3:
            break
        if n not in hits:
            hits.append(n)
    return SYSTEM_PROMPT + "\n\n## 知识库全文\n\n" + _docs_by_names(hits)


# 新增策略只改这里。格式：key -> (显示名, 构造函数：接收当前题 case，可忽略)
STRATEGIES = {
    "none":        ("A·无语料",        _sys_none),
    "inject":      ("B·全量注入",      _sys_inject),
    "inject-gold": ("C·理想检索(3篇)", _sys_inject_gold),
    "bm25":        ("D·BM25检索(3篇)", _sys_bm25),
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

        # gold_sources（2026-10-15 新增）：金标篇目列表
        #   规则（见 docs/金标规则说明-20261015.md）：
        #   · 必须存在、必须是列表；
        #   · 元素必须是 data/docs 下真实存在的 .md 文件名（不写路径）；
        #   · 与 expect 自洽：has 必须非空；none / constraint 必须为空。
        if "gold_sources" not in c:
            errors.append(f"{tag} 缺字段 gold_sources")
        else:
            gs = c["gold_sources"]
            if not isinstance(gs, list):
                errors.append(f"{tag} gold_sources 不是列表：{gs!r}")
            else:
                valid_docs = set(_all_doc_names())
                for n in gs:
                    if not isinstance(n, str):
                        errors.append(f"{tag} gold_sources 元素不是字符串：{n!r}")
                    elif n not in valid_docs:
                        errors.append(f"{tag} gold_sources 指向不存在的文档：{n}")
                if c.get("expect") == "has" and not gs:
                    errors.append(f"{tag} expect=has 但 gold_sources 为空")
                if c.get("expect") in ("none", "constraint") and gs:
                    errors.append(
                        f"{tag} expect={c.get('expect')} 却带了 gold_sources：{gs}")
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
    # 金标分布（2026-10-15 新增）：口径见 docs/金标规则说明-20261015.md
    n_single = sum(1 for c in cases if len(c.get("gold_sources") or []) == 1)
    n_multi = sum(1 for c in cases if len(c.get("gold_sources") or []) >= 2)
    n_none = sum(1 for c in cases if not c.get("gold_sources"))
    print(f"  金标：可回答(1篇)×{n_single}  多答案(≥2篇)×{n_multi}  库外(0篇)×{n_none}")
    # 留出集（2026-10-19 划入）：只用于最终汇报，调参期间须用 --no-holdout 排除。
    n_hold = sum(1 for c in cases if c.get("holdout"))
    if n_hold:
        print(f"  留出集：{n_hold} 题（默认计入全集；调参请加 --no-holdout）")


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
                      r"不承诺|不许说|不夸|"
                      # 2026-11-02 修 J2 新增：**能力否定**。
                      # 实证（C 策略 #15「帮我登记让专人联系我」）：模型答
                      # 「我的系统能力确实**不包含**“代客提交申请”或“**转接人工**回访”的功能」，
                      # 「不包含」是老词表外的否定词 → 命中未被剔除 → 守住边界被判成承诺代办。
                      # 只补这三个「能力否定」短语，不加裸「不」——裸「不」会把
                      # 「不贵」「不少」这类正向词也当否定，误杀范围不可控。
                      r"不包含|不提供|不具备)")
# 实证 #38：「为了避免物流信息混乱，**同一订单最多支持修改 2 次**」
# —— 「避免」在讲规定的原因，却被当成对「2 次」的否认，命中被整条剔除。
# 这两词在真实语料里更多出现在「为了避免踩坑，建议…」这类正向句，保留只会制造假阴性。

# 命中词之前、**同一句内**的上下文（不跨句末标点/换行）。
# 2026-10-19 修正：原实现固定回看 15 字符，两个方向都踩坑 ——
#   · 窗口太短（漏判）：#5「没有任何人敢拍着胸脯保证“100% 绝对新鲜”」，
#     否定词「没有」距命中 20 字外，被当成「使用」→ 守住约束的回答被判违规。
#   · 跨句污染（误杀）：#48「…**绝对不会！**\n\n我们的客服**绝不会**索要…」，
#     回看 15 字跨到上一句的「不会」，把正确答案自己剔掉。
# 改用句子边界：语义上「否定」的辖域本来就止于句末，比固定字数可靠。
_SENT_END = "。！？!?\n"


def _ctx_in_sentence(text, pos):
    """取 text[pos] 所在句子中、pos 之前的部分。"""
    start = max(text.rfind(ch, 0, pos) for ch in _SENT_END) + 1
    return text[start:pos]


def _ctx_after_in_sentence(text, pos, width=20):
    """取 text[pos] 所在句子中、pos 之后的至多 width 字（不跨句末标点/换行）。"""
    end = len(text)
    for ch in _SENT_END:
        p = text.find(ch, pos)
        if p != -1:
            end = min(end, p)
    return text[pos:min(end, pos + width)]


# ======================================================================
# 拒答声明的**作用域**判断（2026-11-02 · 修 J1）
# ======================================================================
# 「查不到」是**歧义**短语：
#   · 知识库没有  →「这边查不到相关资料」        → 是拒答，has 题应判失败
#   · 接口查不到  →「这边查不到实时数据」        → 是能力边界的**正确**表述，不是拒答
# 旧实现对 HARD_REFUSAL 做**子串直判**，两种一律算拒答。
# 实证（C 策略 #45「积分会过期吗」）：答案正文正确写出「次年 12 月 31 日 / 年底清零」，
# 只在结尾说「果小蜜这边查不到实时数据，您可以登录账号查看」——这是**加分**的诚实表述，
# 却被判成拒答，整题由通过翻成失败。**判据把模型的优点算成了缺点。**
#
# 修法：只对歧义词「查不到」看其后文作用域；其余短语（暂时没有资料 / 没有这项信息 /
# 确实没有记录 / 无法确认 …）语义单一，仍直判，避免过度设计。
_DATA_SCOPE = re.compile(r"实时|数据|余额|明细|订单|物流|进度|账户|账号|积分|记录")
_SCOPED_REFUSAL = "查不到"


def _hard_refusals(answer):
    """返回答案里**真正构成「知识库缺失」**的声明（去歧义后的列表）。"""
    out = []
    for phrase in HARD_REFUSAL:
        if phrase not in answer:
            continue
        if phrase != _SCOPED_REFUSAL:
            out.append(phrase)
            continue
        # 歧义词逐处判断：只要有一处不是「查数据」，就算真的说了没资料
        for m in re.finditer(re.escape(phrase), answer):
            if _DATA_SCOPE.search(_ctx_after_in_sentence(answer, m.end())):
                continue                      # 查不到实时数据 → 能力边界，不计拒答
            out.append(phrase)
            break
    return out

# 判断模型答案是否通过
def judge(answer, case):
    """返回 (是否通过, 真正「使用」了的片段, 出现的硬性拒答声明)。

    注意 `hits` 已剔除否定语境中的命中 —— 判定看的是「模型有没有真的这么说」，
    而不是「这几个字有没有出现过」。前者才对应业务事实。
    """
    hits = []
    if case.get("pattern"):
        for m in re.finditer(case["pattern"], answer):
            if NEGATION.search(_ctx_in_sentence(answer, m.start())):
                continue
            hits.append(m.group(0))
    refusals = _hard_refusals(answer)
    if case["expect"] == "has":
        # 必须命中语料事实，且不得同时声称「没有资料」
        ok = bool(hits) and not refusals
    else:
        # none / constraint：命中即失败（编造 或 触碰红线）
        ok = not hits
    return ok, hits, refusals

def _ask(client, model, messages):
    # max_tokens 兜底，防止模型自由发挥把输出 token 拉高（output 是 36 元/百万，
    # 比 input 12 元/百万 贵 3 倍，是主要成本项）。
    # enable_thinking=False：果小蜜的判分只看「有没有命中关键词 / 有没有拒答」，
    # 不需要思维链。实测关掉后 reasoning token 从 146 → 0，输出 token 从 172 → 14。
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        extra_body={"enable_thinking": False},
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
    # counted:False 的题（对照组，如普适常识题）一律不计入分母 —— 它们
    #   答对与否不能归功于知识库接入，混进来会虚高通过率、掩盖真实差距。
    def rate(mode, kinds=None, turns=None):
        rows = [r for c in cases
                if (kinds is None or c["expect"] in kinds)
                and c.get("counted", True)
                and (turns is None
                     or (turns == "single" and "history" not in c)
                     or (turns == "multi" and "history" in c))
                for r in [results[mode][c["id"]]]]
        if not rows:
            return None
        return sum(1 for r in rows if r["ok"]) / len(rows)

    invalid = bool(failures)

    # ---------- 可疑样本：极短回答（2026-10-13 新增）----------
    # 为什么需要：A2 实测撞到一次 —— inject 侧 #19 的存档回答只有 10 个字符
    # （『哈喽呀!" />』），明显是被截断或写坏的响应。它**不抛异常**（所以不被
    # failures 捕获），judge 却把它判成「未通过」→ **冒充成模型失败**，
    # 差点让整轮 A/B 得出错误结论（单独重跑 3 次，全部正常通过）。
    # 阈值取 12 字符：本项目合法回答都是整段话术，最短的也在 300 字以上。
    # ⚠️ 它只**告警**、不改判定 —— 极短回答也可能是一条合法的简短拒答，
    #    所以它进的是「可疑清单」而不是 failures（后者会把整份报告标为无效）。
    suspects = []
    for m in modes:
        for c in cases:
            r = results[m].get(c["id"])
            if not r:
                continue
            ans = (r.get("answer") or "").strip()
            if len(ans) < 12:
                suspects.append((m, c["id"], ans))

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
    if suspects:
        lines.append(f"> ⚠️ **可疑样本 {len(suspects)} 个**（回答不足 12 字符，疑为被截断或写坏的响应）："
                     + "、".join(f"{m}#{cid}" for m, cid, _ in suspects)
                     + "。其「未通过」**不可当作模型失败**，须单独重跑确认。\n")
    lines.append(f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 模型：`{model}`")
    lines.append(f"- 题目数：{len(cases)}")
    n_control = sum(1 for c in cases if not c.get("counted", True))
    if n_control:
        # 分母口径必须写进报告 —— 否则读报告的人无法判断 "94%" 是几分之几，
        # 也无法和旧报告对比（旧报告分母含对照组）。
        lines.append(f"- 计入通过率：**{len(cases) - n_control} 题**"
                     f"（另 {n_control} 题对照组不计入，见下方说明）")
    lines.append(f"- 语料：{_docs_content()['count']} 篇 / "
                 f"{_docs_content()['chars']} 字\n")

    lines.append("## 横向对比\n")
    header = "| 策略 | system 长度 | 总通过率 | 事实题 | 反向题 | 约束题 | 多轮题 | 事实题拒答次数 |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")

    for mode in modes:
        name, _ = STRATEGIES[mode]
        # 与「事实题」列**同口径**：单轮 + 计入分母。
        # 否则分子分母会打架（原来分母把多轮 has 题也算进来，和那列的题数对不上）。
        has_cases = [c for c in cases
                     if c["expect"] == "has" and "history" not in c
                     and c.get("counted", True)]
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

    # ---------- 对照组题（counted:false）单独列出 ----------
    # 为什么要单列：这类题（如「榴莲热量」）是模型自带常识，无语料也能答对。
    # 若混进分母，会让「事实题通过率」虚高，掩盖知识库接入的真实贡献。
    # 它们不是废题——它们是**校准项**：告诉你 A/B 差值里有多少是模型本来就有的。
    control = [c for c in cases if not c.get("counted", True)]
    if control:
        lines.append("> **对照组题（不计入任何通过率）**：" +
                     "、".join(f"#{c['id']}『{c['q']}』" for c in control) +
                     "。\n> 理由：答案属模型自带常识，无语料策略也会答对，"
                     "因此**不能用来证明知识库接入的收益**，"
                     "故排除在分母外、单独列出作为 A/B 差值的校准参考。\n")

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
            # 对照组题：不计入通过率（模型自带常识，无法证明知识库贡献）
            "control_cases": [c["id"] for c in cases if not c.get("counted", True)],
            "valid": not invalid,
            "n_failures": len(failures),
            "failures": [{"mode": m, "case_id": cid, "error": et} for m, cid, et in failures],
            # 可疑样本：极短回答（疑被截断）—— 与 failures 分开，不使报告失效
            "suspects": [{"mode": m, "case_id": cid, "answer": a} for m, cid, a in suspects],
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
    if suspects:
        print(f"\n  ⚠️ 可疑样本 {len(suspects)} 个（回答 < 12 字符，疑被截断/写坏）："
              + "、".join(f"{m}#{cid}" for m, cid, _ in suspects))
        for m, cid, a in suspects:
            print(f"     {m}#{cid} 原文={a!r}")
        print("     其「未通过」不可当模型失败 —— 请单独重跑该题确认。")
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
    # 模型身份的唯一来源是 .env 的 LLM_MODEL。
    # 兜底：未配置时回退到 qwen3.7-plus，但**必须显式告警** ——
    # 静默回退会跑出「看起来正常、其实用错模型」的报告（指标管道骗人）。
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    # 2026-11-02 改：原兜底模型已 403 下线，单值写死 = 一个模型挂则整条评测停摆。
    # 改为降级链（agent/llm.py），报告里的 model 字段一律是**实际**用上的那个。
    from agent.llm import resolve as resolve_model
    model, dropped = resolve_model(client)
    if dropped:
        print(f"⚠️ 降级：{'、'.join(dropped)} 不可用，本次使用 {model}")

    # 记录结果：results[策略key][题目id] = dict(ok, hits, refusals, answer)
    results = {m: {} for m in modes}
    system_lengths = {}
    failures = []          # [(策略, 题号, 异常类型)] —— 用于识别「额度耗尽」这类假失败
    _init_noise(cases)      # gold 策略的固定噪声篇目（依赖题目集）

    for mode in modes:
        name, builder = STRATEGIES[mode]
        # gold 策略逐题不同 → 该列取「逐题长度均值」作代表值；
        # 其余策略每题相同，均值即常量，不受影响。
        lens = [len(builder(c)) for c in cases]
        system_lengths[mode] = sum(lens) // len(lens)
        varied = "（逐题不同，此为均值）" if len(set(lens)) > 1 else ""
        print(f"\n{'=' * 66}\n策略 {mode}（{name}）system 长度 "
              f"{system_lengths[mode]:,} 字符{varied}\n{'=' * 66}")

        for case in cases:
            try:
                system_content = builder(case)
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
    ap.add_argument("--no-holdout", action="store_true",
                    help="排除留出集题目（W3–W8 调参期间用；最终汇报才跑全集）")
    ap.add_argument("--extra-dir", action="append", default=[], metavar="DIR",
                    help="额外语料目录（可重复）。只影响「全量注入」策略；不传 = 45 篇基线")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="只跑前 N 题（额度探底用）。0 = 不限")
    args = ap.parse_args()

    print(f"题目文件：{CASES_PATH}")
    if args.extra_dir:
        # 必须在任何 _docs_content() 调用之前设置，否则缓存里是旧内容
        _EXTRA_DIRS.extend(args.extra_dir)
        _DOCS_CACHE.clear()
        print(f"⚠️ --extra-dir：全量注入将额外并入 {len(args.extra_dir)} 个目录 —— "
              f"这与 45 篇基线**不可直接比较**，报告里必须写明语料规模")
    cases = load_cases()
    if not cases:
        raise SystemExit("cases.jsonl 是空的 —— 先写题。")

    # 留出集隔离：调参期间不看这 8 题，避免「用考题调参」导致的乐观偏差。
    if args.no_holdout:
        held = sorted(c["id"] for c in cases if c.get("holdout"))
        cases = [c for c in cases if not c.get("holdout")]
        print(f"⚠️ --no-holdout：已排除留出集 {len(held)} 题 {held}"
              f"，本次只跑 {len(cases)} 题")

    errors = check_cases(cases)
    print_coverage(cases)
    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
    print("  字段与正则校验：通过")

    # 额度探底：先跑 N 题确认链路与免费额度，再决定要不要跑全量。
    # 动机是「免费额度模式下 403 只会出现在中途」——52 题跑到第 30 题断掉，
    # 会留下半份报告并被改名 *_INVALID，白烧前面 29 题的额度。
    # 故探底必须在开始前显式收缩题量，而不是边跑边撞。
    if args.limit:
        before = len(cases)
        cases = cases[:args.limit]
        print(f"⚠️ --limit：只跑前 {len(cases)} 题（共 {before} 题），用于额度探底 —— "
              f"**这份报告不能当验收数字**")

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
