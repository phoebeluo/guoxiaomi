import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")

# print(__file__)   E:\guoxiaomi\rag\load_docs.py
# print(os.path.abspath(__file__))  E:\guoxiaomi\rag\load_docs.py
# print(os.path.dirname(os.path.abspath(__file__)))  E:\guoxiaomi\rag
# print(BASE_DIR)  E:\guoxiaomi
# print(DOCS_DIR)  E:\guoxiaomi\data\docs
# print(os.path.join(DOCS_DIR, "*.md"))  E:\guoxiaomi\data\docs\*.md

def load_all_docs(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.md")))
    parts = []
    total_chars = 0

    for path in files:
        name = os.path.basename(path)
        title = name[:-3] #没有.md
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        parts.append(f"【来源：{title}】\n{text}")
        total_chars += len(text)

    return "\n\n".join(parts), len(files), total_chars


if __name__ == "__main__":
    content, count, chars = load_all_docs(DOCS_DIR)
    print(f"文件数：{count}")
    print(f"总字数：{chars}")
    print("含琯溪蜜柚：", "琯溪蜜柚" in content)
    print("来源标注数：", content.count("【来源："))
