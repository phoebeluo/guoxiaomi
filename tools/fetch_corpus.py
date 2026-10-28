"""语料抓取骨架（两级：列表页 → 详情页），W3 第 2 步。

设计要点
--------
1. **两级抓取**（沿用 TMDB 那套）：列表页只负责「拿候选链接」，详情页才拿正文。
   分开的好处：换源时只改列表页解析，正文提取不用重写。
2. **先落 raw，不碰 data/docs**：`data/docs/` 是评测基准，一动全部报告作废。
   抓下来的东西先进 `data/raw/sample/`，人工看过、质检过，再决定要不要并入。
3. **每篇带溯源头**：正文前写 `【来源：标题】` 与原始 URL。
   后面任何一条语料都能回到它的出处——这是「不清洗就不知道自己在测什么」的前提。
4. **礼貌抓取**：默认每篇间隔 1 秒。不是为了躲封，是因为 400 篇打完就跑会真的被封。

用法
----
    # 从已保存的列表页 HTML 抓 20 篇（推荐：先存列表页，避免重复请求）
    E:/guoxiaomi/.venv/Scripts/python.exe tools/fetch_corpus.py --limit 20

    # 直接给列表页 URL
    E:/guoxiaomi/.venv/Scripts/python.exe tools/fetch_corpus.py --list-url https://example.com/news/ --limit 20

    # 只抓标题含指定关键词的（提高内容对口率）
    E:/guoxiaomi/.venv/Scripts/python.exe tools/fetch_corpus.py --keep 选购,保存,挑选,怎么,如何 --limit 20
"""

import argparse
import html as htmllib
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
PROBE_DIR = os.path.join(PROJECT, "data", "raw", "_probe")
SAMPLE_DIR = os.path.join(PROJECT, "data", "raw", "sample")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# 明显不是知识型的标题词：会议、通报、招标、项目公示……
DROP_WORDS = ["会议", "论坛", "公示", "公布", "名单", "招标", "中标", "采购公告",
              "培训", "大赛", "评选", "通知", "函", "实施细则", "征求意见"]


def clean_text(frag):
    frag = re.sub(r"<script[\s\S]*?</script>", " ", frag, flags=re.I)
    frag = re.sub(r"<style[\s\S]*?</style>", " ", frag, flags=re.I)
    frag = re.sub(r"<br\s*/?>", "\n", frag, flags=re.I)
    frag = re.sub(r"</p>", "\n", frag, flags=re.I)
    frag = re.sub(r"<[^>]+>", "", frag)
    frag = htmllib.unescape(frag)
    frag = re.sub(r"[ \t\u3000]+", " ", frag)
    frag = re.sub(r"\n\s*\n+", "\n", frag)
    return frag.strip()


def extract_article(html):
    """取 id="article" 容器的内容，用 div 开合平衡来确定结束位置。

    为什么不用 `([\\s\\S]+?)</div>`：正文里有嵌套 div，非贪婪会在第一个 </div> 就截断
    （实测只拿到 55 字）。平衡计数虽然笨，但不会截断。
    """
    i = html.find('id="article"')
    if i < 0:
        return ""
    start = html.find(">", i) + 1
    depth = 1
    for m in re.finditer(r"<div\b|</div>", html[start:]):
        if m.group(0).lower().startswith("<div"):
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return clean_text(html[start:start + m.start()])
    return clean_text(html[start:start + 20000])


def get_title(html):
    m = re.search(r"<title>([^<]+)</title>", html)
    if not m:
        return ""
    t = htmllib.unescape(m.group(1))
    return re.split(r"[_|｜]", t)[0].strip()


def collect_links(list_html, base):
    """从列表页 HTML 抓「详情页链接 + 标题」。"""
    links = []
    for m in re.finditer(r'href="([^"]+\.html)"[^>]*>([^<]{6,80})<', list_html):
        url, title = m.group(1), m.group(2).strip()
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = base + url
        elif not url.startswith("http"):
            url = base + "/" + url.lstrip("./")
        links.append((url, htmllib.unescape(title)))
    # 去重保序
    return list(dict.fromkeys(links))


def safe_name(title, idx, out_dir=None):
    """生成文件名；同名已存在时自动加后缀（支持多批次抓进同一目录）。"""
    t = re.sub(r'[\\/:*?"<>|\s]+', "-", title)[:40].strip("-")
    base = f"{idx:03d}-{t}"
    if out_dir:
        cand, k = f"{base}.md", 1
        while os.path.exists(os.path.join(out_dir, cand)):
            k += 1
            cand = f"{base}-{k}.md"
        return cand
    return f"{base}.md"


def main():
    ap = argparse.ArgumentParser(description="两级抓取：列表页 → 详情页")
    ap.add_argument("--list-html", default=os.path.join(PROBE_DIR, "http_www_foodmate_net_.html"),
                    help="已保存的列表页 HTML（默认用探针存下的食品伙伴网首页）")
    ap.add_argument("--list-url", help="列表页 URL（给了就不用 --list-html）")
    ap.add_argument("--limit", type=int, default=20, help="抓几篇")
    ap.add_argument("--keep", default="", help="标题必须含其中之一（逗号分隔）")
    ap.add_argument("--delay", type=float, default=1.0, help="每篇间隔秒数")
    ap.add_argument("--out", default=SAMPLE_DIR, help="落盘目录")
    args = ap.parse_args()

    import requests

    if args.list_url:
        r = requests.get(args.list_url, headers={"User-Agent": UA}, timeout=15)
        r.encoding = r.apparent_encoding or r.encoding
        list_html, base = r.text, re.match(r"https?://[^/]+", args.list_url).group(0)
    else:
        if not os.path.exists(args.list_html):
            raise SystemExit(f"列表页 HTML 不存在：{args.list_html}\n先跑 tools/probe_source.py --save <url>")
        list_html = open(args.list_html, encoding="utf-8", errors="replace").read()
        base = "https://www.foodmate.net"

    links = collect_links(list_html, base)
    keep = [k for k in args.keep.split(",") if k]
    picked, dropped = [], 0
    for url, title in links:
        if any(d in title for d in DROP_WORDS):
            dropped += 1
            continue
        if keep and not any(k in title for k in keep):
            dropped += 1
            continue
        picked.append((url, title))

    print(f"列表页解析到 {len(links)} 条；过滤掉 {dropped} 条；待抓 {min(len(picked), args.limit)} 条\n")
    os.makedirs(args.out, exist_ok=True)

    ok = fail = 0
    for i, (url, title) in enumerate(picked[:args.limit], 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
            r.encoding = r.apparent_encoding or r.encoding
            body = extract_article(r.text)
            real_title = get_title(r.text) or title
        except Exception as e:
            print(f"  [失败] {title[:30]} —— {type(e).__name__}")
            fail += 1
            continue

        if len(body) < 200:
            print(f"  [太短] {title[:30]} —— 正文 {len(body)} 字，跳过")
            fail += 1
            continue

        path = os.path.join(args.out, safe_name(real_title, i, args.out))
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"【来源：{real_title}】\n\n> 抓取自 {url}\n\n{body}\n")
        print(f"  [OK] {len(body):>5} 字  {real_title[:34]}")
        ok += 1
        time.sleep(args.delay)

    print(f"\n成功 {ok} 篇｜失败 {fail} 篇｜目录 {args.out}")
    if ok:
        print(f"下一步：python rag/chunk.py --dir {os.path.relpath(args.out, PROJECT)} --quiet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
