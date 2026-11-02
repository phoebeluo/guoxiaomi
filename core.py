"""果小蜜公共核心层（2026-11-02 · W12）。

## 为什么抽这一层

在此之前，所有业务逻辑长在 `app.py` 里，而 `app.py` 的第一行是 `import streamlit`。
后果很具体：想给同一套能力加一个 HTTP 接口，只能**复制一遍**检索 + 拼 prompt + 调模型 + 存会话，
然后两份代码各自漂移。

所以这里只抽四件事（计划里 W12 只允许抽 1 个文件、4 个函数）：

| 函数 | 干什么 |
|---|---|
| `retrieve` | 按 `RETRIEVAL` 档取 top-k 篇 |
| `build_system_message` | 拼 system（人设 + 知识库全文），返回来源名单 |
| `call_llm` | 调模型，流式 / 非流式两种入口 |
| `save_session` / `load_session` | 会话落盘（Streamlit 与 Flask 共用同一份文件格式） |

**不抽**：UI、状态管理、页面控件。那些是 Streamlit 强绑定的，抽出来只会造出一层假抽象。

## 一条纪律

核心层**不 import streamlit**。只要这一条守住了，Flask 那边就能直接复用；
一旦有人在这里为了省事 `import streamlit`，这层就退化成第二个 app.py。
`tests/test_core.py` 里有一条断言专门锁这件事。
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

from agent.loop import stream_chat
from agent.tools import TOOL_SCHEMAS
from rag.bm25 import BM25
from rag.prompt import SYSTEM_PROMPT
from rag.vector import VectorIndex

PROJECT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(PROJECT, "data", "docs")
SESSIONS_DIR = os.path.join(PROJECT, "data", "sessions")
LOGS_DIR = os.path.join(PROJECT, "data", "logs")

# 与 app.py 同口径的历史截断条数（10 条消息 = 5 轮）
MAX_HISTORY = 10

# RETRIEVAL: vector（默认，语义近似 top-k 篇）/ bm25（可回退）/ off（全量注入，A/B 对照）
# TOOLS:    on（默认）/ off（退回纯 RAG）
#
# 为什么是环境变量而不是请求参数：HTTP 接口一旦允许调用方切换，
# 同一批压测请求就会落在不同的注入口径上，QPS 与回答质量都无法归因。
RETRIEVAL = os.environ.get("RETRIEVAL", "vector").strip().lower()
TOOLS_ON = os.environ.get("TOOLS", "on").strip().lower() != "off"
TOPK = 3

_docs = None
_index = {}


def load_docs(extra_dirs=None):
    """[(篇名, 正文)]。线上只用冻结的 45 篇；外部语料靠 RETRIEVAL_EXTRA 显式开启。"""
    global _docs
    if _docs is None:
        import glob
        docs = []
        for d in [DOCS_DIR] + list(extra_dirs or []):
            for p in sorted(glob.glob(os.path.join(d, "*.md"))):
                with open(p, encoding="utf-8") as f:
                    docs.append((os.path.basename(p), f.read()))
        _docs = docs
    return _docs


def _get_index(kind, extra_dirs=None):
    docs = load_docs(extra_dirs)
    key = (kind, len(docs))
    if key not in _index:
        _index[key] = VectorIndex(docs) if kind == "vector" else BM25(docs)
    return _index[key]


def retrieve(query, k=TOPK, mode=None, extra_dirs=None):
    """返回 [(篇名, 正文)]，按分数降序。

    检索的是**整篇**不是块，与 `eval_run` 的 D 策略同口径 ——
    线上和评测必须用同一个检索单元，否则报告里的召回率无法支撑线上行为。
    """
    mode = mode or RETRIEVAL
    idx = _get_index("vector" if mode == "vector" else "bm25", extra_dirs)
    # 两个索引都实现了 text(doc_id)，所以这里不需要分支——
    # 线上只有一条取文路径，测试才能锁住「换了索引实现但取文口径不变」
    return [(n, idx.text(n)) for n, _ in idx.search(query, top_k=k)]


def build_knowledge_block(query, mode=None, extra_dirs=None):
    """拼「知识库全文」段。返回 (文本, 来源篇名列表)。"""
    mode = mode or RETRIEVAL
    if mode == "off":
        return "".join(t for _, t in load_docs(extra_dirs)), []
    hits = retrieve(query, mode=mode, extra_dirs=extra_dirs)
    if not hits:
        # 零命中不静默退回全量：那等于把 30K 字符又塞回去，检索白做，
        # 而且用一堆不相关的资料回答库里没有的问题 = 鼓励编造。
        return ("（本次检索未命中任何文档，请按能力边界如实告知没有这项信息）", [])
    return "\n\n".join(f"### {n}\n{t}" for n, t in hits), [n for n, _ in hits]


def build_system_message(query, nickname="果小蜜", character="热情、专业、懂生活、像闺蜜一样贴心",
                         mode=None, extra_dirs=None):
    """返回 (system_content, sources)。线上与离线评测共用同一套拼接方式。"""
    knowledge, sources = build_knowledge_block(query, mode=mode, extra_dirs=extra_dirs)
    system = (SYSTEM_PROMPT
              .replace("{name}", nickname)
              .replace("{character}", character)
              + "\n\n## 知识库全文\n\n" + knowledge)
    return system, sources


def recent_messages(messages, limit=MAX_HISTORY):
    """只保留最近 limit 条，且**从 user 起头**。

    不截断的话多轮对话会一路膨胀；不从 user 起头则可能把 assistant 的半句话
    排在 messages 最前，部分模型会直接报 400。
    """
    recent = messages[-limit:] if limit and len(messages) > limit else list(messages)
    while recent and recent[0].get("role") != "user":
        recent.pop(0)
    return recent


def call_llm(client, model, messages, tools=None, stream=True, extra_body=None):
    """调模型。stream=True 返回事件生成器；False 返回 (answer, usage, tool_calls)。

    事件类型由 `agent.loop.stream_chat` 定义：delta / tools / usage。
    """
    kw = dict(tools=TOOL_SCHEMAS if (tools is None and TOOLS_ON) else tools,
              extra_body=extra_body if extra_body is not None else {"enable_thinking": False})
    gen = stream_chat(client, model, messages, **kw)
    if stream:
        return gen

    answer, usage, calls = "", None, []
    for kind, payload in gen:
        if kind == "delta":
            answer += payload
        elif kind == "tools":
            calls += payload
        elif kind == "usage":
            usage = payload
    return answer, usage, calls


# ---------------------------------------------------------------- 会话落盘
def generate_session_name():
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def session_path(name):
    return os.path.join(SESSIONS_DIR, f"{name}.json")


def save_session(name, messages, nickname="果小蜜",
                 character="热情、专业、懂生活、像闺蜜一样贴心"):
    """落盘一个会话。返回文件路径。

    只做**整份覆盖**，不做增量追加：会话文件是「当前状态的快照」，
    增量追加一旦中途失败会留下半条消息，读回来直接 400。
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = session_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nickname": nickname, "character": character,
                   "current_session": name, "messages": messages},
                  f, ensure_ascii=False, indent=2)
    return path


def list_sessions():
    if not os.path.exists(SESSIONS_DIR):
        return []
    names = [f[:-5] for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    return sorted(names, reverse=True)


def load_session(name):
    path = session_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delete_session(name):
    path = session_path(name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------- 遥测
def append_request_log(record):
    """追加一行请求遥测到 `data/logs/requests.jsonl`。

    HTTP 接口与 Streamlit 写同一个文件：压测数据要和线上数据放在一张表里，
    否则「P99 是多少」和「首字延迟是多少」会变成两套互不可比的数字。
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, "requests.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_env():
    """按项目根目录加载 .env（不依赖调用方 cwd）。"""
    load_dotenv(os.path.join(PROJECT, ".env"))
