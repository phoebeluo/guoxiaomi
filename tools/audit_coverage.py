# -*- coding: utf-8 -*-
"""知识库覆盖度审计（零成本，不调用 API）。

回答一个问题：**客户可能问的东西，语料到底盖住了多少？**

四段输出：
  A. 断链检测    —— 文档里写了《某某文档》，但那个文件并不存在
  B. 分类覆盖    —— 每类文档的篇数与字数（看结构是否偏科）
  C. 栏目一致性  —— 同类文档的栏目是否齐全（如 19 篇百科是否都有 5 栏）
  D. 问题域扫描  —— 预置的「生鲜电商客服问题域」词表，语料里零命中 = 候选盲区

用法：
    python tools/audit_coverage.py
"""
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "docs")

# ---- D 段用的词表：每个域 = (域名, 代表关键词正则, 处置状态) ----
# why 用词表而不是枚举问题：任何一条具体问句都可能恰好没被问到，
# 但「域」是稳定集合 —— 域缺失才是真盲区。
#
# 处置状态（用户 2026-10-12 裁定；只有「应覆盖」才可能报警）：
#   "应覆盖" —— 语料应当命中；零命中 = ❌ 回归告警（文档被删/被改坏，与业务无关）
#   "不补"   —— 业务上不存在该功能，**保持拒答才是正确答案** → 📌 已封口，永不再报警
#
# ★ 为什么必须留「不补」这一档：对**已知的有意设计**反复报警，会让人开始无视报警，
#   而被无视的校验器等于没有校验器（本项目原则第 10 条；同 check_kinds.py 的 KNOWN_EXCEPTIONS）。
DOMAINS = [
    ("快递承运商", r"快递|承运|物流商|顺丰|京东物流|中通|圆通|韵达|申通|邮政|EMS", "应覆盖"),
    ("物流轨迹查询", r"运单号|物流更新|实时轨迹|查物流", "不补"),
    ("发货出库时效", r"发货时间|出库|下单后.{0,4}小时|预计发货", "应覆盖"),
    ("自提与门店", r"自提|门店|到店取", "应覆盖"),
    ("预约/指定送达", r"预约配送|指定时间|指定送达", "不补"),
    ("起送价", r"起送|最低消费|最低下单", "应覆盖"),
    ("企业采购/团购", r"团购|批量采购|企业客户|公司采购", "不补"),
    ("价格保护", r"价格保护|保价|降价补差|买贵", "不补"),
    ("溯源与检测", r"溯源|检测报告|质检报告|农残", "应覆盖"),
    ("礼品卡/储值", r"礼品卡|储值卡|购物卡|充值", "不补"),
    ("缺货与补货", r"缺货|补货|到货通知|售罄", "不补"),
    ("特殊人群食用", r"孕妇|糖尿病|婴幼儿|过敏", "应覆盖"),
    ("催熟与保鲜剂", r"催熟|乙烯|保鲜剂|防腐剂", "应覆盖"),
    ("有机/认证", r"有机|绿色食品|无公害|认证", "不补"),
    ("进口与产地直发", r"进口|产地直发|产地直采|原产地", "应覆盖"),
    ("下单渠道", r"小程序|公众号|APP|微信下单", "应覆盖"),
    ("优惠券有效期", r"有效期|过期|失效", "应覆盖"),
    ("售后回访/工单", r"回访|工单|处理进度查询", "不补"),
]


def load_all():
    out = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        text = open(path, encoding="utf-8").read()
        out.append((os.path.basename(path), text))
    return out


def main():
    docs = load_all()
    if not docs:
        print(f"❌ 没找到语料：{DOCS_DIR}")
        return

    print("=" * 68)
    print("A. 断链检测：写了《某某文档》但文件不存在")
    print("=" * 68)
    names = {n[:-3] for n, _ in docs}
    broken = []
    for name, text in docs:
        for ref in set(re.findall(r"《([^》]+)》", text)):
            if ref not in names:
                broken.append((name, ref))
    if broken:
        for src, ref in broken:
            print(f"  ❌ {src}  →  引用《{ref}》（缺失）")
    else:
        print("  ✅ 无断链，所有交叉引用都有对应文件")

    print()
    print("=" * 68)
    print("B. 分类覆盖：篇数 / 字数（去空白）")
    print("=" * 68)
    buckets = {}
    for name, text in docs:
        cat = name.split("-")[0]
        chars = len(re.sub(r"\s", "", text))
        c, w = buckets.get(cat, (0, 0))
        buckets[cat] = (c + 1, w + chars)
    total_c, total_w = 0, 0
    for cat, (c, w) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
        total_c += c
        total_w += w
        print(f"  {cat:<6} {c:>3} 篇   {w:>6} 字")
    print(f"  {'合计':<6} {total_c:>3} 篇   {total_w:>6} 字")

    print()
    print("=" * 68)
    print("C. 栏目一致性：同名前缀的文档，二级标题是否一致")
    print("=" * 68)
    groups = {}
    for name, text in docs:
        prefix = name.split("-")[0]
        heads = tuple(re.findall(r"^##\s+(.+)$", text, re.M))
        groups.setdefault(prefix, []).append((name, heads))
    for prefix, items in sorted(groups.items()):
        if len(items) < 3:
            continue  # 1~2 篇无统计意义：主题本就不同，栏目自然不同
        sigs = {}
        for name, heads in items:
            sigs.setdefault(heads, []).append(name)
        if len(sigs) == 1:
            print(f"  ✅ {prefix}：{len(items)} 篇，栏目完全一致（{len(next(iter(sigs)))} 栏）")
        else:
            ranked = sorted(sigs.items(), key=lambda kv: -len(kv[1]))
            print(f"  ℹ️ {prefix}：{len(items)} 篇 → {len(sigs)} 种结构"
                  "（各篇主题不同，无统一模板属正常；只有『多数篇共享同一模板』才叫异常）")
            for heads, ns in ranked:
                print(f"       {len(ns)} 篇 · {len(heads)} 栏：{' / '.join(heads) or '(无二级标题)'}")

    print()
    print("=" * 68)
    print("D. 问题域扫描：语料里零命中 = 候选盲区")
    print("=" * 68)
    def samples(pat, limit=3):
        """返回 [(文件名, 命中词, 上下文)]。上下文用于人工核对语义角色 —— 见下方提示一。"""
        out = []
        for name, text in docs:
            for m in re.finditer(pat, text):
                s, e = max(0, m.start() - 12), min(len(text), m.end() + 12)
                out.append((name, m.group(), re.sub(r"\s+", " ", text[s:e])))
                if len(out) >= limit:
                    return out
        return out

    hit, regress, sealed = [], [], []
    for domain, pat, status in DOMAINS:
        found = samples(pat)
        if found:
            hit.append((domain, status, found))
        elif status == "应覆盖":
            regress.append(f"  ❌ {domain:<12} 应覆盖却零命中 —— 该域文档可能被删/改坏，请核查")
        else:
            sealed.append(f"  📌 {domain:<12} 已裁定不补（业务无此功能 → 保持拒答才是正确答案）")

    for domain, status, found in hit:
        print(f"  {'✅' if status == '应覆盖' else '📌'} {domain}")
        for name, word, ctx in found:
            print(f"       「{word}」@{name}：…{ctx}…")
    print()
    if regress:
        print(f"  —— ❌ 回归告警 {len(regress)} 个域 ——")
        for line in regress:
            print(line)
        print()
    print(f"  —— 📌 已裁定不补 {len(sealed)} 个域（已封口，不再提示补语料）——")
    for line in sealed:
        print(line)

    print()
    print("提示一：命中 ≠ 语料真有这项。**子串撞车极常见** —— 本项目实测：")
    print("        「快递承运商」看似命中『中通』，实际来自『小程序中通过…』；")
    print("        所以必须看上下文确认语义角色（这是本项目第 6 次踩同一条）。")
    print("提示二：本段只回答「语料有没有提」，不回答「该不该补」。")
    print("        该补的已补完 → ✅；业务本就没有的已裁定封口 → 📌，不再打扰。")
    print("        **只有 ❌ 才需要你介入**，且那是「已有文档被改坏」，不是新缺口。")


if __name__ == "__main__":
    main()
