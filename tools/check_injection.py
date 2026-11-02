"""
Step 2 注入验证脚本
------------------------------------------------------------
不启动 Streamlit，直接从 app.py 里把 system_prompt 抠出来，
离线复现 system_content 的构造过程，验证三件事：

  一、改动是否到位（静态检查）
  二、构造结果是否正确（占位符、语料、来源标注）
  三、为什么不能用 .format()（现场演示崩溃）

运行：python check_injection.py
"""

import os
import re
import sys

# 2026-11-02 修 CI：原来写死 r"E:/guoxiaomi"，在 Linux runner 上 sys.path 里是条
# 不存在的路径 → `from rag...` 直接 ModuleNotFoundError。改为按文件位置推导。
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from rag.load_docs import load_all_docs, DOCS_DIR

# 2026-11-02（W12）：提示词唯一来源改为 rag/prompt.py
APP_PATH = PROJECT + "/app.py"
PROMPT_PATH = PROJECT + "/rag/prompt.py"
src = open(APP_PATH, encoding="utf-8").read()
prompt_src = open(PROMPT_PATH, encoding="utf-8").read()


def line(title):
    print()
    print("=" * 52)
    print(title)
    print("=" * 52)


def show(desc, ok):
    print(f"  [{'OK' if ok else '!!'}] {desc}")


line("一、静态检查：改动是否到位")
for desc, ok in [
    ("已移除 system_prompt.format() 调用", "system_prompt.format(" not in src),
    ("使用了 .replace 填 name", '.replace("{name}"' in src),
    ("使用了 .replace 填 character", '.replace("{character}"' in src),
    ("存在 @st.cache_data 装饰器", "@st.cache_data" in src),
    ("messages 用的是 system_content", '"role": "system", "content": system_content' in src),
    ("已 import load_all_docs", "from rag.load_docs import" in src),
]:
    show(desc, ok)


line("二、离线复现 system_content 构造")
m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', prompt_src, re.S)
system_prompt = m.group(1)
content, count, chars = load_all_docs(DOCS_DIR)

prompt_replaced = (
    system_prompt
    .replace("{name}", "果小蜜")
    .replace("{character}", "热情、专业、懂生活、像闺蜜一样贴心")
)
head = prompt_replaced + "\n\n## 知识库全文\n\n"
built = head + content

print(f"  语料篇数            : {count}")
print(f"  语料正文总字数      : {chars}")
print(f"  system_content 长度 : {len(built)} 字符")
print()

# messages 里「历史展开」用的锚点会随重构变化（Step 3 把裸切片换成了 recent_messages()）。
# 原脚本写死 '*st.session_state.messages'，重构后直接抛 ValueError —— 属"检查脚本比实现先过期"。
# 现改为按优先级探测，命中哪个用哪个。
# 2026-10-13：线上护栏+遥测改造把历史先取成局部变量 hist（为了记 n_history），
# 于是 create() 里变成 `*hist,` —— 故补一个锚点。**这不是放宽检查**：
# 仍要求「system 消息出现位置 < 历史锚点位置」，即 system 必须在最前。
HISTORY_ANCHORS = ("*hist,", "*recent_messages(", "*st.session_state.messages")
hist_anchor = next((a for a in HISTORY_ANCHORS if a in src), None)
system_first = (
    hist_anchor is not None
    and src.index('"role": "system"') < src.index(hist_anchor)
)

for desc, ok in [
    ("占位符 {name} 已全部替换", "{name}" not in built),
    ("占位符 {character} 已全部替换", "{character}" not in built),
    ("语料正文已拼入", "琯溪蜜柚" in built),
    (f"来源标注 {count} 个齐全", built.count("【来源：") == count),
    ("知识库标题存在", "## 知识库全文" in built),
    ("语料紧贴提示词末尾，中间无插入内容", built.index(content) == len(head)),
    (f"system 消息位于 messages 首位（锚点 {hist_anchor or '未找到'}）", system_first),
    # 内部 SOP 隔离的**内容级**验证：即使文件还在磁盘上，也必须进不了 system
    ("内部 SOP 未被注入（投诉分级/上报链路/台账）",
     all(k not in built for k in ("客服组长", "客服主管", "店铺负责人", "录入台账", "不得自行处理"))),
    ("对客投诉依据已注入", "投诉受理说明" in built or "投诉受理" in built),
]:
    show(desc, ok)


line("三、为什么不能用 .format()：现场演示")
fake_prompt = "你是{name}。参考资料如下："
fake_doc = "配置示例：{timeout: 30, retry: 3}"

print(f"  模拟提示词 : {fake_prompt}")
print(f"  模拟语料   : {fake_doc}")
print()

print("  方案 A  .replace ：", end="")
r = fake_prompt.replace("{name}", "果小蜜") + "\n" + fake_doc
print(f"通过，语料原样保留 → {r.splitlines()[-1]}")

print("  方案 B  .format  ：", end="")
try:
    (fake_prompt + "\n" + fake_doc).format(name="果小蜜")
    print("意外通过（不该发生）")
except Exception as e:
    print(f"崩溃 → {type(e).__name__}: {e}")

line("结论")
print("  当前语料里花括号数为 0，所以两种写法现在都能跑。")
print("  但只要往库里加一篇带 JSON 示例的文档，方案 B 立刻崩。")
print("  方案 A 不受任何影响。")
