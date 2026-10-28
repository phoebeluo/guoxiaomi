import re
import os
import glob

CHUNK_SIZE = 400          # 单个块的最大字符数
OVERLAP=50               # 相邻块重叠的目标字符数；按整句回退，实际重叠会略大于此值

# 单句长度上限：超过就二次切分（2026-10-28 新增，W3 小样验证触发）
# 依据：现有 45 篇最长句 112 字，留约 60% 余量取 180 →
#   · 对现有语料**完全不触发**（所以 W1 基线 96 块 / 330.1 不受影响，可回归验证）
#   · 对新语料能拦住 1725 字的超长句（无标点的编号串 / 英文段落）
# 不设这个上限的后果：超长句会直接撑爆块（实测出现 1817 字的块，P95 从 398 飙到 889）。
MAX_SENT = 180


def _hard_split(sentence, limit):
    """超长句二次切分：先按逗号/分号切，仍超长则硬切。

    为什么先按逗号：中文长句的语义最小单元是逗号分句，按它切至少不会把
    一个词组劈开；只有连逗号都没有的串（编号、英文、代码）才需要硬切。
    """
    if len(sentence) <= limit:
        return [sentence]
    out = []
    for piece in re.split(r'(?<=[，；、,;])', sentence):
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            out.append(piece)
    return out

# 句子分割
def split_sentences(content):
    parts = re.split(r'(?<=[。！？\n])', content)
    cleaned = [s.strip() for s in parts]
    cleaned = [s for s in cleaned if s]
    return [x for s in cleaned for x in _hard_split(s, MAX_SENT)]

# 打包成“块（chunk）”
def split_chunks(sentences, source):
    chunks = []
    buf = []
    for s in sentences:
        if len("".join(buf)) + len(s) > CHUNK_SIZE:
            chunks.append({"source": source, "text": "".join(buf)})
            tail = get_tail(buf, OVERLAP)  # ← 从刚交出的块里取尾巴
            buf = tail  # ← 自己填：新袋子该怎么起步？
            buf.append(s)
        else:
            buf.append(s)
    if buf:
        chunks.append({"source": source, "text": "".join(buf)})

    return chunks

# [
#   {"source": "xxx.txt", "text": "句子1句子2"},
#   {"source": "xxx.txt", "text": "句子2句子3句子4"},
#   ...
# ]
# 典型的 RAG chunk 数据结构


def get_tail(sentences, overlap):
    tail = []
    total = 0
    for s in reversed(sentences):
        if total>=overlap:
            break
        else:
            tail.insert(0, s)
            total+=len(s)
    return tail


if __name__ == "__main__":
    # 2026-10-27 改：原版把目录硬编码成 data/docs，且只打印「每篇几块」。
    # W3 的小样验证需要「规模无关」的 4 个数（平均块长 / P95 块长 / 中位句长 / P90 句长），
    # 所以这里加 --dir 指向小样目录，并把统计打出来。
    import argparse
    import statistics

    ap = argparse.ArgumentParser(description="切分统计：给语料目录出验收数")
    ap.add_argument("--dir", default=r"E:\guoxiaomi\data\docs", help="语料目录（小样验证时指向 data/raw/...）")
    ap.add_argument("--quiet", action="store_true", help="不逐篇打印，只出汇总")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.md")))
    if not files:
        raise SystemExit(f"目录里没有 .md：{args.dir}")

    sent_lens, chunk_lens = [], []
    empty_chunks = 0
    total_chars = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        total_chars += len(text)
        sentences = split_sentences(text)
        chunks = split_chunks(sentences, os.path.basename(path))
        sent_lens += [len(s) for s in sentences]
        chunk_lens += [len(c["text"]) for c in chunks]
        empty_chunks += sum(1 for c in chunks if not c["text"].strip())
        if not args.quiet:
            print(f"{os.path.basename(path)}     共 {len(chunks)} 块")

    def pct(vals, p):
        """分位数（最近秩法）。P95 用它能避免小样本被插值抹平。"""
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    print(f"\n篇数 {len(files)}｜总字数 {total_chars}｜块数 {len(chunk_lens)}｜句数 {len(sent_lens)}")
    print(f"中位句长 {statistics.median(sent_lens):.0f}｜P90 句长 {pct(sent_lens, .9)}")
    print(f"平均块长 {statistics.mean(chunk_lens):.1f}｜P95 块长 {pct(chunk_lens, .95)}｜空块 {empty_chunks}")
    # 66 = W1 实测的相邻块重叠字数（公式预测 65.5）。此处只做跨来源粗比，不用于对外引用。
    print(f"冗余率 ≈ {(len(chunk_lens) - len(files)) * 66 / total_chars:.0%}（粗估）")


