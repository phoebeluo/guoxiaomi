import re
import os
import glob

CHUNK_SIZE = 400          # 单个块的最大字符数


def split_sentences(content):
    parts = re.split(r'(?<=[。！？\n])', content)
    cleaned = [s.strip() for s in parts]
    return [s for s in cleaned if s]


def split_chunks(sentences, source):
    """把句子列表切成块，返回 [{"source": 文件名, "text": 块内容}, ...]"""
    chunks = []
    buf = ""
    for s in sentences:
        if len(buf)+len(s)>CHUNK_SIZE:
            chunks.append({"source": source, "text": buf})
            buf = s
        else:
            buf += s
    if buf:
        chunks.append({"source": source, "text": buf})
    return chunks


if __name__ == "__main__":
    FOLDER = r"E:\guoxiaomi\data\docs"
    files = sorted(glob.glob(os.path.join(FOLDER, "*.md")))

    total_chunks = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = split_sentences(text)
        chunks = split_chunks(sentences, os.path.basename(path))
        total_chunks += len(chunks)
        print(f"{os.path.basename(path)}     共 {len(chunks)} 块")
    print(f"共 {len(files)} 篇，合计 {total_chunks} 块")

