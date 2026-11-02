import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import time
from rag.load_docs import load_all_docs, DOCS_DIR
from rag.prompt import SYSTEM_PROMPT
import core
from rag.bm25 import BM25
from rag.vector import VectorIndex
from rag.refusal import check as check_refusal
from agent.loop import stream_chat
from agent.llm import resolve as resolve_model
from agent.tools import TOOL_SCHEMAS

load_dotenv()

# 会话目录：基于本文件位置解析，避免从其它目录启动 streamlit 时路径错乱
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sessions")
MAX_HISTORY = 10

# 函数
# 生成会话名称或时间
# 会话与历史：文件读写全部委托 core.py（2026-11-02 · W12）
# 之前 save/load/delete 各写一遍，Flask 那边就要再写一遍；现在两边共用同一份落盘格式。
def generate_session_name():
    return core.generate_session_name()


def save_messages():
    if st.session_state.current_session:
        core.save_session(st.session_state.current_session,
                          st.session_state.messages,
                          st.session_state.nickname,
                          st.session_state.character)


def load_sessions():
    return core.list_sessions()


def load_session(session_name):
    try:
        data = core.load_session(session_name)
        st.session_state.messages = data["messages"]
        st.session_state.nickname = data["nickname"]
        st.session_state.character = data["character"]
        st.session_state.current_session = data["current_session"]
    except Exception as e:
        st.error(f"加载会话失败：{e}")


def delete_session(session_name):
    try:
        if core.delete_session(session_name):
            # 删的是当前会话就清空内存状态并换新名，避免界面继续指向已删文件
            if session_name == st.session_state.current_session:
                st.session_state.messages = []
                st.session_state.current_session = core.generate_session_name()
            # 无论删的是不是当前会话都要 rerun：侧边栏列表是本次运行开头读的，
            # 不刷新的话被删条目会一直留着（st.rerun 抛 BaseException，不会被上面的 except 吞掉）
            st.rerun()
    except Exception as e:
        st.error(f"删除会话失败：{e}")


def recent_messages(messages, max_count):
    return core.recent_messages(messages, max_count)


# 请求记录目录：与 data/sessions 同级，同样基于本文件位置解析
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs")

# 错误分类表：按本项目**实际撞到过**的异常特征归类，不是凭空猜的。
# 为什么要分类：分类后才能给不同降级话术，也才能统计「哪类错误最常见」。
# 顺序有意义：quota 放最前，避免 403 携带的其它关键字被后面的分支抢走。
ERROR_KINDS = (
    ("quota",      ("403", "quota", "Free quota exhausted", "Arrearage", "欠费", "额度")),
    ("timeout",    ("Timeout", "timeout", "APITimeoutError")),
    ("connection", ("APIConnectionError", "ConnectionError", "Connection error")),
    ("content",    ("data_inspection", "DataInspection", "400", "invalid")),
)

# 降级话术：出错时给用户一句**人话**，而不是原始异常。
# ⚠️ 不暴露内部原因（尤其不暴露「额度」）—— 这也是本项目的对客约束。
FALLBACK_TEXT = {
    "quota":      "果小蜜这会儿有点忙不过来，稍等一下再来问我呀～🍈",
    "timeout":    "刚才网络有点慢，麻烦您再发一次哦～",
    "connection": "刚才网络有点慢，麻烦您再发一次哦～",
    "content":    "这个果小蜜不太方便答，我们聊聊水果吧～🍈",
    "other":      "果小蜜这边出了点小状况，麻烦您再试一次哦～",
}


def classify_error(exc):
    """把异常归类成 quota / timeout / connection / content / other 五类之一。"""
    text = f"{type(exc).__name__}: {exc}"
    for kind, keys in ERROR_KINDS:
        if any(k in text for k in keys):
            return kind
    return "other"


def log_request(record):
    """把一轮请求追加写入 data/logs/requests.jsonl。

    为什么用 JSONL：一行一条、追加写入，崩溃不会毁掉前面的记录，
    也能直接 grep / jq，不需要额外依赖。
    为什么整段包 try：遥测是**旁路**，它自己失败绝不能拖垮主流程。
    """
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, "requests.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"⚠️ 请求记录写入失败：{e}")
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# 页面配置
st.set_page_config(
    page_title="果小蜜",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

# 初始化聊天信息
if "messages" not in st.session_state:
    st.session_state.messages = []
# 昵称
if "nickname" not in st.session_state:
    st.session_state.nickname = "果小蜜"
# 性格
if "character" not in st.session_state:
    st.session_state.character = "热情、专业、懂生活、像闺蜜一样贴心"
# 会话标识
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()

# 知识库注入方式（W5 · 2026-10-29）
# RETRIEVAL=on  → BM25 取 top-3 篇注入（线上真实路径，与 W4「D·BM25检索」同口径）
# RETRIEVAL=off → 全量注入 45 篇（W1–W4 基线口径，保留作 A/B 对照组）
#
# 为什么把开关做成环境变量而不是界面控件：界面控件会让用户在会话中途切换注入方式，
# 同一段对话前后的 system 口径就变了，回答变化无法归因。环境变量是「启动即定」。
# RETRIEVAL=vector（默认）→ 语义近似 top-3 篇（SVD 低秩，45 篇档实测最优）
# RETRIEVAL=bm25          → BM25 top-3 篇（W5 上线版，保留可回退）
# RETRIEVAL=off           → 全量注入 45 篇（W1–W4 基线，A/B 对照组）
#
# 为什么默认从 bm25 改成 vector（2026-11-01 实测）：
#   线上只有 45 篇，同一批 48 题的 R@3：vector 95.8% > RRF 93.8% > bm25 89.6%。
#   这与 346 篇档的结论相反（那里 RRF 最好）—— 因为降维维度受文档数限制：
#   45 篇只能降到 45 维（几乎无损），此时语义近似单路最强，融合反而被较弱的 BM25 拖累。
#   **在 346 篇上调出来的结论不能搬到 45 篇用。**
_MODE = os.environ.get("RETRIEVAL", "vector").strip().lower()
if _MODE in ("", "on", "1", "true", "yes"):
    _MODE = "vector"                     # 兼容 W5 时期的 RETRIEVAL=on
if _MODE not in ("vector", "bm25", "off"):
    _MODE = "vector"
TOPK = 3

# 工具开关（2026-11-02 · W9）
# TOOLS=on（默认）→ 带 3 个工具的 schema 调模型；off → 退回纯 RAG（W8 行为），A/B 对照用
_TOOLS = os.environ.get("TOOLS", "on").strip().lower() != "off"

# 外部语料**默认不进线上索引**（2026-10-29 实测决定）：
#   加上 301 篇后，命中一篇法规长文就能把 system 从 5.6K 撑到 18.7K 字符，
#   且口语提问（如「这单能赔多少」）会被「如何处理」之类的虚词拉去命中《能力验证不合格怎么办》
#   这类完全无关的检定文书 —— 异构噪声在 R@5 指标上不衰减，但在**实际注入内容**上明显跑偏。
#   故线上只用冻结的 45 篇；301 篇的价值在评测侧（证明规模不衰减），不在线上。
#   需要复现 346 档线上路径时：RETRIEVAL_EXTRA=data/raw/external_clean
_extra = os.environ.get("RETRIEVAL_EXTRA", "").strip()
EXTRA_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), _extra)] if _extra else []


def build_knowledge_block(query):
    """拼「知识库全文」段。返回 (文本, 来源篇名列表)。

    2026-11-02（W12）起改为调用 `core.build_knowledge_block` —— 同一份逻辑
    Streamlit 与 Flask 共用，不再各写一份。
    索引缓存也随之搬到 core 的模块级变量：模块只 import 一次，
    所以 Streamlit 每次交互重跑脚本时不会重建索引，不需要 @st.cache_resource。
    """
    return core.build_knowledge_block(query, mode=_MODE, extra_dirs=EXTRA_DIRS)

# 系统提示词
# 系统提示词：唯一来源是 rag/prompt.py（2026-11-02 抽离，评测与线上同源）
system_prompt = SYSTEM_PROMPT

# 标题
st.title("果小蜜")
# 会话名称
st.text(f"会话名称：{st.session_state.current_session}")
# 加载历史会话
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


# 创建与AI大模型交互的客户或对象
# 阿里云百炼（DashScope）OpenAI 兼容模式
# 文档：https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
client = OpenAI(
    api_key=os.environ.get('DASHSCOPE_API_KEY'),
    base_url=os.environ.get('DASHSCOPE_BASE_URL',
                            'https://dashscope.aliyuncs.com/compatible-mode/v1'))

# ________________________________________________________________________________________________________
# 侧边栏
with st.sidebar:
    st.subheader("AI控制面板")
    # 新建会话
    if st.button("新建会话",width="stretch",icon="🍇"):
        if st.session_state.messages:
            save_messages()
        st.session_state.messages=[]
        st.session_state.current_session=generate_session_name()
        st.rerun()

    # 会话历史
    st.text("会话历史")
    session_list=load_sessions()
    for session in session_list:
        col1,col2=st.columns([4,1])
        with col1:
            if st.button(session,width="stretch",icon="🍋‍🟩",key=f"load_{session}"):
                load_session(session)
                st.rerun()
        with col2:
            if st.button(session,width="stretch",icon="❌️",key=f"delete_{session}"):
                delete_session(session)

    st.divider()

    # 输入框
    st.subheader("信息")
    # 昵称
    nickname=st.text_input("昵称",placeholder="请输入昵称",value=st.session_state.nickname)
    if nickname:
        st.session_state.nickname=nickname
    # 性格
    character=st.text_input("性格",placeholder="请输入性格",value=st.session_state.character)
    if character:
        st.session_state.character=character


# 消息输入框
prompt=st.chat_input("请输入您的问题：")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    # ---- 能力边界拒答（W8）------------------------------------------------
    # 规则命中就直接给指路话术，**不调模型**。
    # 为什么放在调模型之前：这类问题的正确答案是一句固定的指路话术，
    # 调模型既浪费 token，又给了它「编一个处理流程」的机会。
    # 为什么用规则不用检索分数：实测两类题的余弦区间完全交叠，没有可用阈值
    # （见 tools/calibrate_refusal.py 与 eval/reports/refusal_threshold.md）。
    _deny = check_refusal(prompt)
    if _deny:
        _rule, _reply = _deny
        st.chat_message("assistant").write(_reply)
        st.session_state.messages.append({"role":"assistant","content":_reply})
        log_request({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session": st.session_state.current_session,
            "model": None,
            "turn": len(st.session_state.messages),
            "n_history": len(recent_messages(st.session_state.messages, MAX_HISTORY)),
            "system_chars": 0,
            "q": prompt,
            "sources": [],
            "refused": _rule,          # 命中的规则名，便于统计哪类拒答最多
            "ok": True,
            "error": None,
            "answer": _reply,
            "ttft_s": 0.0,
            "total_s": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
        })
        st.caption(f"📌 能力边界：{_rule}（未调用模型）")
        st.stop()

    # 与AI大模型进行交互
    # 用 replace 而不是 format：str.format 会把语料里的 {} 当作占位符去解析，
    # 一旦语料中出现 JSON 示例或模板语法就会抛 KeyError。
    # replace 只精确替换这两个字符串，语料内容不会被误伤。
    # 语料拼在系统提示词之后、位于 messages 最前，保证前缀固定可命中 cache
    knowledge, sources = build_knowledge_block(prompt)
    system_content = (
        system_prompt
        .replace("{name}", st.session_state.nickname)
        .replace("{character}", st.session_state.character)
        + "\n\n## 知识库全文\n\n"
        + knowledge
    )
    # 模型身份的唯一来源是 .env 的 LLM_MODEL。
    # 兜底：未配置时回退到 qwen3.5-plus-2026-02-15，但**必须显式告警** ——
    # 静默回退会导致「以为在跑 A、其实在跑 B」，指标全部失效。
    # 降级链：LLM_MODEL → LLM_FALLBACKS → 内置默认（agent/llm.py）。
    # 2026-11-02 起：额度耗尽不再让整个应用停摆，但**降级会在遥测里留痕**，
    # 记的是实际回答的模型，不是配置里写的那个。
    model, dropped = resolve_model(client)
    if dropped:
        st.warning(f"⚠️ 模型降级：{'、'.join(dropped)} 不可用，本次使用 {model}")
    # ---- 护栏 + 遥测（2026-10-13 新增）----------------------------------
    # 为什么加：这段调用原本是**裸调** —— 额度耗尽 / 网络抖动会把原始异常
    # 直接抛给用户；而离线评测（tools/eval_run.py）早就有一套失败保护。
    # 同一件事，离线严丝合缝、线上零防护 —— 这个不对称就是要补的东西。
    hist = recent_messages(st.session_state.messages, MAX_HISTORY)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session": st.session_state.current_session,
        "model": model,
        "turn": len(st.session_state.messages),   # 含本轮提问，即「第几问」
        "n_history": len(hist),
        "system_chars": len(system_content),
        "q": prompt,
        "sources": sources,          # 本次注入的篇名，用于事后核对「回答依据 vs 实际注入」
        "ok": True,
        "error": None,
    }
    t0 = time.perf_counter()
    first_chunk_at = None
    usage = None
    tool_traces = []
    try:
        # 2026-11-02（W9）：模型调用统一走 agent/loop.py 的流式对话循环。
        # 它比裸调多两件事：① 本轮带 function calling schema ② 收到 tool_calls 后
        # 执行工具、把结构化结果回灌、再跑一轮生成。不需要工具的提问行为完全不变
        # （仍是逐字流式），所以这不影响绝大多数问答的体验与延迟。
        with st.chat_message("assistant"):
            response_message = st.empty()
            full_response = ""
            for kind, payload in stream_chat(
                client,
                model,
                [{"role": "system", "content": system_content}, *hist],
                tools=TOOL_SCHEMAS if _TOOLS else None,
                extra_body={"enable_thinking": False},
            ):
                if kind == "delta":
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()   # 首个正文 token 的时刻
                    full_response += payload
                    response_message.write(full_response)
                elif kind == "tools":
                    tool_traces.extend(payload)
                elif kind == "usage":
                    usage = payload          # dict：prompt_tokens / completion_tokens
            if tool_traces:
                names = "、".join(t["name"] for t in tool_traces)
                ok = all(t["result"].get("ok") for t in tool_traces)
                st.caption(f"🔧 已调用工具：{names}" + ("" if ok else "（返回失败，已如实告知）"))

    except Exception as e:
        # 出错不把异常抛给用户：换成一句降级话术，并把错误分类落进记录。
        # （原先这里没有 try —— 额度耗尽时用户看到的就是一屏原始异常。）
        kind = classify_error(e)
        full_response = FALLBACK_TEXT[kind]
        with st.chat_message("assistant"):
            st.write(full_response)
        # 界面提示**不含内部原因**（不暴露「额度」等），细节只写进日志
        st.warning("⚠️ 本轮回复失败，已降级处理（详见 data/logs/requests.jsonl）")
        record["ok"] = False
        record["error"] = {"type": type(e).__name__, "kind": kind}

    # ---- 落盘：无论成功还是降级，都记一条 ----
    t_end = time.perf_counter()
    record["answer"] = full_response
    record["ttft_s"] = round(first_chunk_at - t0, 3) if first_chunk_at else None
    record["total_s"] = round(t_end - t0, 3)
    record["in_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
    record["out_tokens"] = getattr(usage, "completion_tokens", None) if usage else None
    log_request(record)

    # 保存
    st.session_state.messages.append({"role":"assistant","content":full_response})

    # 出处展示：放在回答之后，用 caption 弱化视觉权重（不是正文，是依据）
    if sources:
        st.caption("📚 本轮依据：" + " · ".join(s[:-3] if s.endswith(".md") else s for s in sources))

    save_messages()