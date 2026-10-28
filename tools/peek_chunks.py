"""临时脚本：打印 百科-柚子.md 的切分结果，验证 overlap 是否生效。
运行：.venv\\Scripts\\python.exe tools\\peek_chunks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunk import split_sentences, split_chunks

PATH = r"E:\guoxiaomi\data\docs\百科-柚子.md"

with open(PATH, encoding="utf-8") as f:
    text = f.read()

chunks = split_chunks(split_sentences(text), os.path.basename(PATH))
print(f"块数：{len(chunks)}")

for i, c in enumerate(chunks, 1):
    t = c["text"]
    print(f"\n===== 第 {i} 块 | source={c['source']} | 长度={len(t)} =====")
    print("【前 80】", t[:80])
    print("【后 80】", t[-80:])

if len(chunks) >= 2:
    a, b = chunks[0]["text"], chunks[1]["text"]
    n = 0
    for k in range(1, min(len(a), len(b)) + 1):
        if a[-k:] == b[:k]:
            n = k
    print(f"\n>>> 相邻块重叠长度：{n} 字（期望 60~70）")
else:
    print("\n>>> 只有 1 块，与预判不符")
