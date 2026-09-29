import glob
import os

FOLDER = "E:/guoxiaomi/data/docs/"

pattern=os.path.join(FOLDER,"*.md")
files=glob.glob(pattern)

total_files = 0
total_chars = 0

for path in files:
    with open(path,"r",encoding="utf-8") as f:
        text=f.read()

    name=os.path.basename(path)
    char_count=len(text)

    print(f"{name}  {char_count}字")

    total_files+=1
    total_chars += char_count
print(f"共 {total_files} 个文件，总计 {total_chars} 字")