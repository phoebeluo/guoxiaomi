# Step 1 · 语料加载器（逐块讲解版）

> 配套：`果小蜜-P0分步实施手册.md`　本文件把 Step 1 拆到"每一块做什么 + 为什么"的粒度

---

## 先定路径（最容易写错的一块）

`load_docs.py` 放在 `rag/` 下，而语料在 `data/docs/`。所以要先从 `rag/` **往上退一级**回到项目根，再往下进 `data/docs`。

```
E:/guoxiaomi/
├── app.py
├── rag/
│   └── load_docs.py     ← 你在这里
└── data/
    └── docs/            ← 要去这里
```

`os.path.dirname()` 的作用是"去掉路径最后一段"：

```python
os.path.abspath(__file__)          # E:/guoxiaomi/rag/load_docs.py
os.path.dirname(上面)               # E:/guoxiaomi/rag
os.path.dirname(再套一次)            # E:/guoxiaomi      ← 项目根
```

套两次 `os.path.dirname` 就回到根目录了。然后：

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
```

`app.py:10` 就是这个写法，可以对照着看。

**为什么要这么麻烦**：写死 `E:/guoxiaomi/...` 的话，换台电脑、换个盘符、把文件夹挪个位置，全部崩掉。用 `__file__` 推导出来的路径，文件夹整体搬到哪都能跑。

---

## 第 1 块 · 拿到文件列表

```python
import glob
import os

files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
files = sorted(files)
```

- `glob.glob("*.md")` = 找出这个目录下所有 `.md` 结尾的文件，返回一个**列表**
- 第二行的 `sorted()` **不能省**

### 为什么 sorted() 不能省（本步最值钱的知识点）

`glob` 返回的顺序取决于文件系统返回目录项的次序，**不保证每次都一样**。

后果：你拼出来的字符串顺序一旦变化，21k token 的 prefix cache 就全部失效——每轮对话都在重算全量输入，成本上去了，延迟上去了，而且**你在界面上看不出任何异常**，只会觉得"好像变慢了"。这种 bug 极难排查。

`sorted()` 让顺序固定下来，前缀就能被复用。

> 面试可以这么讲：全量注入方案里我最关心的是前缀可缓存，所以语料顺序必须固定、必须放在消息列表最前、且不能混入时间戳之类的动态字段。这三件事决定成本是线性可控还是失控。

---

## 第 2 块 · 准备一个"片段仓库"

```python
parts = []
```

先建一个空列表，后面每读一篇就往里放一个片段。**不要在循环里用字符串 `+=` 拼接**。

原因：Python 的字符串是不可变的，`s = s + x` 并不是"在原字符串后面追加"，而是"新建一个更长的字符串，把旧内容抄一遍"。循环 44 次就要抄 44 遍，越到后面越慢。

正确做法是**先收集到列表，最后一次 join**。

---

## 第 3 块 · 循环处理每一篇

```python
for path in files:
    name = os.path.basename(path)      # 百科-柚子.md
    title = name[:-3]                  # 百科-柚子（去掉 .md）
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts.append(f"【来源：{title}】\n{text}")
```

逐行解释：

| 代码 | 作用 | 备注 |
|---|---|---|
| `os.path.basename(path)` | 从完整路径里取出文件名 | `E:/.../百科-柚子.md` → `百科-柚子.md` |
| `name[:-3]` | 切片，砍掉末尾 3 个字符 | `-3` 表示"倒数第 3 个到结尾"，正好是 `.md` |
| `encoding="utf-8"` | 显式指定编码 | Windows 默认编码不一定是 utf-8，不写会乱码或直接抛异常 |
| `f"【来源：{title}】\n{text}"` | f-string 拼接 | 你在 `app.py` 里写过 `f"{session_name}.json"`，同一个东西 |

`\n` 是换行符，让来源标注单独占一行。

---

## 第 4 块 · 拼成一个字符串

```python
return "\n\n".join(parts)
```

`join` 的作用是：**用左边的字符串当胶水，把列表里的每一项粘成一个大字符串**。

```python
"\n\n".join(["A", "B", "C"])   →   "A\n\nB\n\nC"
```

用 `\n\n`（两个换行 = 一个空行）是因为语料是 Markdown。Markdown 里空行表示分段，模型读到空行就知道"上一篇结束了，下一篇开始了"。只用一个 `\n` 的话，两篇文档会糊在一起。

---

## 第 5 块 · 计数（为了验收）

你需要两个数字来验收：文件数、总字数。

- 文件数：`len(files)`
- 总字数：可以在循环里用一个变量累加 `len(text)`，也可以最后 `sum(...)`

让函数返回三个值（`拼接结果, 文件数, 总字数`），Python 支持一次返回多个：

```python
return content, len(files), total_chars
```

调用时：

```python
content, count, chars = load_all_docs(DOCS_DIR)
```

---

## 常见错误预判（写之前先看一遍）

这五个坑，几乎每个人第一次写都会踩到至少一个：

| # | 错误写法 | 实际后果 |
|---|---|---|
| 1 | 忘了 `sorted()` | 顺序不稳定 → prefix cache 随机失效 → 成本延迟上升，但界面上看不出任何异常 |
| 2 | `total_chars = 0` 写进了 for 循环**里面** | 每读一篇就重置一次，最终结果只等于最后一篇的字数（约 500），而不是 25,432 |
| 3 | `name[:3]` 写成了正切片 | `[:3]` 是"取前三个字符"（得到 `百科-`），砍掉 `.md` 要用 `[:-3]`（负号表示从末尾数） |
| 4 | `open(path)` 不写 `encoding="utf-8"` | Windows 默认编码可能是 GBK，中文直接抛 `UnicodeDecodeError` |
| 5 | `parts.append` 写成了 `parts = f"..."` | 循环里只有 append 才会累积，赋值会把上一篇覆盖掉，最后只剩 1 篇 |

第 2 条和第 5 条最阴——**不报错，只是结果不对**。看到"文件数 44 但总字数只有几百"，先怀疑第 2 条。

---

## 验收：跑两遍比对

在文件末尾加：

```python
if __name__ == "__main__":
    content, count, chars = load_all_docs(DOCS_DIR)
    print(f"文件数：{count}")
    print(f"总字数：{chars}")
    print("含琯溪蜜柚：", "琯溪蜜柚" in content)
    print("来源标注数：", content.count("【来源："))
```

期望输出：

| 检查项 | 期望值 |
|---|---|
| 文件数 | 44 |
| 总字数 | 25,432 |
| 含琯溪蜜柚 | True |
| 来源标注数 | 44 |

> 注意：`chars` 应该是 25,432（纯正文），而 `len(content)` 会更大，因为多了 44 条来源标注。**两个数不一样是正常的**，别以为算错了。

再加一步验证顺序稳定性：

```bash
python rag/load_docs.py > a.txt
python rag/load_docs.py > b.txt
fc a.txt b.txt
```

`fc` 应当提示"找不到差异"。这一条直接验证了 cache 能不能吃上。

---

## 完成标志

以上四项全部对上，且两次运行输出完全一致 → Step 1 完成，可以进 Step 2（注入）。

**现在先不要碰 app.py。** Step 1 只验证加载器本身能工作，注入是下一步的事。
