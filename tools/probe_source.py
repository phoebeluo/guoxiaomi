"""来源可行性探针（W3 第 1 步配套工具，2026-10-27）。

为什么需要它
------------
W3 的 go/no-go 是「100 篇抓取成本 > 6 h 就换源」。
但不试抓就写来源清单 = 纸面清单：写的时候觉得能抓，动手时才发现要登录 / 要 JS / 被滑块拦。

本工具把「这个源能不能用」变成 30 秒可判的一件事：
    HTTP 状态 → 字节数 → 去标签后的正文中文字数 → 中文占比 → 反爬信号

为什么不用 BeautifulSoup
------------------------
项目 venv 里只有 requests，没有 bs4。为了不为一个探针装依赖，这里用正则去标签。
它不追求完美解析，只回答一个问题：**静态 HTML 里到底有没有中文正文**。

用法
----
    # 跑内置候选清单（覆盖 4 类来源）
    E:/guoxiaomi/.venv/Scripts/python.exe tools/probe_source.py --list

    # 测自己找到的网址（可多个，含中文直接贴）
    E:/guoxiaomi/.venv/Scripts/python.exe tools/probe_source.py https://xxx/a https://xxx/b

    # 存下 HTML 供人工检查（存到 data/raw/_probe/）
    E:/guoxiaomi/.venv/Scripts/python.exe tools/probe_source.py --save
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
PROBE_DIR = os.path.join(PROJECT, "data", "raw", "_probe")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# 未经验证的候选：脚本自己会告诉你哪个能用，别提前相信任何一条
CANDIDATES = [
    ("百科", "百度百科·柚子", "https://baike.baidu.com/item/柚子"),
    ("百科", "百度百科·山竹", "https://baike.baidu.com/item/山竹"),
    ("百科", "搜狗百科·车厘子", "https://baike.sogou.com/v64551339.htm"),
    ("国标", "全国标准信息公共服务平台", "https://openstd.samr.gov.cn/bzgk/gb/index"),
    ("客服", "京东帮助中心", "https://help.jd.com/"),
    ("商品页", "惠农网首页", "https://www.cnhnb.com/"),
]

# 反爬 / 需要登录 / 需要 JS 的信号词
BLOCK_SIGNALS = ["请开启JavaScript", "验证码", "滑动验证", "安全验证", "请登录",
                 "访问受限", "人机验证", "captcha", "Access Denied"]


def strip_tags(html):
    """粗去标签，只留可见文本。够用于判断「有没有正文」。"""
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", html).strip()


def probe(url, save=False, timeout=12):
    import requests

    row = {"url": url, "status": "-", "bytes": 0, "cjk": 0, "ratio": 0.0,
           "verdict": "NG", "reason": ""}
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    except Exception as e:                       # 网络不可达 / 超时 / TLS
        row["reason"] = f"{type(e).__name__}（网络不可达或被拦）"
        return row, ""

    r.encoding = r.apparent_encoding or r.encoding
    html = r.text
    text = strip_tags(html)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ratio = cjk / max(1, len(text))

    row.update(status=r.status_code, bytes=len(r.content), cjk=cjk, ratio=ratio)

    blocked = next((s for s in BLOCK_SIGNALS if s.lower() in html.lower()), None)
    if r.status_code != 200:
        row["reason"] = f"HTTP {r.status_code}"
    elif blocked:
        row["reason"] = f"反爬信号「{blocked}」"
    elif cjk < 300:
        row["reason"] = f"正文仅 {cjk} 个中文字（疑似 JS 渲染或需登录）"
    elif ratio < 0.35:
        row["reason"] = f"中文占比仅 {ratio:.0%}（正文混在大量脚本/样式里）"
    else:
        row["verdict"] = "OK"
        row["reason"] = f"正文 {cjk} 中文字，占比 {ratio:.0%}"

    if save and html:
        os.makedirs(PROBE_DIR, exist_ok=True)
        safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", url)[-60:] + ".html"
        with open(os.path.join(PROBE_DIR, safe), "w", encoding="utf-8") as f:
            f.write(html)
        row["saved"] = safe
    return row, text


def main():
    ap = argparse.ArgumentParser(description="来源可行性探针")
    ap.add_argument("urls", nargs="*", help="要测的网址（留空则跑内置候选）")
    ap.add_argument("--list", action="store_true", help="跑内置候选清单")
    ap.add_argument("--save", action="store_true", help="把 HTML 存到 data/raw/_probe/")
    args = ap.parse_args()

    if args.urls:
        targets = [("自定义", "", u) for u in args.urls]
    else:
        targets = CANDIDATES

    print(f"探测 {len(targets)} 个来源（超时 12s/个）\n")
    print(f"{'判定':<4} {'类别':<6} {'名称':<22} {'状态':<4} {'字节':>8} {'中文字':>6}  说明")
    print("-" * 96)

    ok = 0
    for cat, name, url in targets:
        row, _ = probe(url, save=args.save)
        ok += row["verdict"] == "OK"
        label = name or url[:22]
        print(f"{row['verdict']:<4} {cat:<6} {label:<22} {str(row['status']):<4} "
              f"{row['bytes']:>8} {row['cjk']:>6}  {row['reason']}")

    print("-" * 96)
    print(f"可用 {ok} / {len(targets)}")
    print("\n判定口径：200 + 正文 ≥300 中文字 + 中文占比 ≥35% → OK")
    print("⚠️ OK 只代表「静态 HTML 里有正文」，不代表可批量抓（频率限制要另测）。")
    if args.save:
        print(f"HTML 已存到：{PROBE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
