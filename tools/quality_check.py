"""语料质检 + 冲突扫描（W3 第 5 步，2026-10-28）。

为什么必须有这一步
------------------
抓回来的是「网页正文」，不是「语料」。至少四类东西要挡掉：

  1. **太短**：正文 < 300 字，切完一块都不够，只会拉低平均块长
  2. **英文/编号为主**：中文占比过低，切句会产出 1,700 字超长块（10-28 实测）
  3. **重复**：同一篇被不同栏目转载三五次，会人为放大某条信息的权重
  4. **业务冲突**（最要命）：外部文章里写了别家的运费门槛、会员等级、赔付比例——
     一旦并进知识库，`配送/售后/会员/优惠` 那 30 道题的 `gold_sources` 就会
     指向互相矛盾的篇目。W4 测出来的「准确率随规模下降」，就变成「我自己把语料写矛盾了」。

第 4 条是 A 方案（噪声路线）的硬规则，不靠记性，靠脚本扫描。

用法
----
    E:/guoxiaomi/.venv/Scripts/python.exe tools/quality_check.py --dir data/raw/corpus
    E:/guoxiaomi/.venv/Scripts/python.exe tools/quality_check.py --dir data/raw/corpus --apply
"""

import argparse
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

MIN_CHARS = 300          # 正文字数下限
MIN_CJK_RATIO = 0.50     # 中文占比下限
NEAR_DUP_JACCARD = 0.85  # 近似重复阈值

# 业务冲突词：这些是「果小蜜自己的规则」，外部语料出现即为冲突。
# 判定用「至少命中 2 类」——单个「24 小时」在食品新闻里太常见，单命中会误杀。
CONFLICT_RULES = {
    "运费门槛": r"(满\s*\d+\s*元?\s*免运|免运费门槛|包邮门槛|起送价|不设起送)",
    "会员体系": r"(会员等级|金卡|银卡|V3\s*金卡|会员权益|积分抵)",
    "赔付标准": r"(坏果赔付|赔付比例|按.{0,4}比例赔|七天无理由|无理由退货)",
    "配送承诺": r"(当日达|次日达|15\s*[:：]\s*00\s*前|截单时间)",
    "承运商": r"(顺丰|中通|圆通|韵达|京东物流).{0,12}(发货|承运|配送)",
}


def body_of(path):
    """去掉我们自己加的三行头（【来源】/引用块），只留正文。"""
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = [ln for ln in text.splitlines() if not ln.startswith("> 抓取自")]
    text = "\n".join(lines)
    text = re.sub(r"^【来源：[^】]*】\s*", "", text)
    return text.strip()


def cjk_ratio(s):
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    return cjk / max(1, len(s))


def shingles(s, n=3, k=400):
    """取正文前 k 字的 3-gram 集合，用于近似重复判定。"""
    head = re.sub(r"\s+", "", s)[:k]
    return {head[i:i + n] for i in range(max(0, len(head) - n))}


def main():
    ap = argparse.ArgumentParser(description="语料质检 + 业务冲突扫描")
    ap.add_argument("--dir", required=True, help="待检目录")
    ap.add_argument("--apply", action="store_true", help="把通过的文件复制到 <dir>_clean/")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".md"))
    if not files:
        raise SystemExit(f"目录里没有 .md：{args.dir}")

    rows, seen_hash, seen_sh = [], {}, []
    for name in files:
        path = os.path.join(args.dir, name)
        body = body_of(path)
        n = len(body)
        ratio = cjk_ratio(body)
        h = hashlib.md5(re.sub(r"\s+", "", body).encode("utf-8")).hexdigest()

        verdict, reason = "OK", ""
        conflict_hits = [k for k, pat in CONFLICT_RULES.items() if re.search(pat, body)]

        if n < MIN_CHARS:
            verdict, reason = "DROP", f"正文仅 {n} 字"
        elif ratio < MIN_CJK_RATIO:
            verdict, reason = "DROP", f"中文占比 {ratio:.0%} 过低"
        elif h in seen_hash:
            verdict, reason = "DROP", f"与 {seen_hash[h]} 完全重复"
        else:
            sh = shingles(body)
            for prev_name, prev_sh in seen_sh:
                inter = len(sh & prev_sh)
                union = len(sh | prev_sh) or 1
                if inter / union >= NEAR_DUP_JACCARD:
                    verdict, reason = "DROP", f"与 {prev_name} 近似重复（{inter/union:.0%}）"
                    break
            if verdict == "OK":
                seen_hash[h] = name
                seen_sh.append((name, sh))

        if verdict == "OK" and len(conflict_hits) >= 2:
            verdict, reason = "CONFLICT", "命中业务冲突：" + "、".join(conflict_hits)

        rows.append({"name": name, "chars": n, "ratio": ratio,
                     "verdict": verdict, "reason": reason, "url": ""})

    ok = [r for r in rows if r["verdict"] == "OK"]
    drop = [r for r in rows if r["verdict"] == "DROP"]
    conflict = [r for r in rows if r["verdict"] == "CONFLICT"]

    print(f"待检 {len(rows)} 篇 → 通过 {len(ok)}｜剔除 {len(drop)}｜冲突 {len(conflict)}\n")
    for label, group in [("剔除", drop), ("冲突", conflict)]:
        if group:
            print(f"--- {label}（{len(group)}）---")
            for r in group[:15]:
                print(f"  {r['name'][:34]:<36} {r['chars']:>5} 字  {r['reason']}")
            if len(group) > 15:
                print(f"  …另有 {len(group) - 15} 篇")

    lens = sorted(r["chars"] for r in ok)
    if lens:
        print(f"\n通过篇目字数：中位 {lens[len(lens)//2]}｜最小 {lens[0]}｜最大 {lens[-1]}"
              f"｜合计 {sum(lens):,} 字")

    if args.apply:
        out = args.dir.rstrip("\\/") + "_clean"
        os.makedirs(out, exist_ok=True)
        for r in ok:
            shutil.copy2(os.path.join(args.dir, r["name"]), os.path.join(out, r["name"]))
        print(f"\n✅ 已复制 {len(ok)} 篇到 {out}")
    else:
        print("\n（未复制。确认清单后加 --apply）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
