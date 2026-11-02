"""果小蜜 HTTP 接口（2026-11-02 · W12）。

## 为什么是 Flask 而不是 FastAPI

17 条 JD 里 Flask/Django 出现 3 次、FastAPI 0 次。同一个 `/chat` 接口两边成本一样，
用 Flask 可以**一次做两件事**：果小蜜服务化 + 补后端关键词。

## 三个接口

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/chat` | 一次问答，返回 JSON（含来源、工具调用、用量） |
| POST | `/chat/stream` | 同上，SSE 流式 |
| GET/DELETE | `/sessions[/<id>]` | 会话列表 / 详情 / 删除 |
| GET | `/health` | 探活（压测时用来区分「服务挂了」和「上游慢」） |

## 并发保护做在入口

上游 DashScope 有 RPM 配额，压到某个并发必然 429。所以这里用信号量限制**同时在途**的请求数，
超出直接 503 + `Retry-After`，而不是让请求堆在上游超时。
「我的吞吐上限不在我的服务，在上游 RPM」——这个判断决定了限流放在哪一层。

## 压测怎么不被模型拖死

设 `MOCK_LLM=1` 时用假模型（固定延迟），这样测到的是**服务层**的 QPS/P99，
不含上游波动。两件事必须分开测，否则「接口慢」到底是我的问题还是阿里的问题说不清。
"""

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, request, stream_with_context
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core
from agent.llm import resolve as resolve_model

core.load_env()

app = Flask(__name__)

# 同时在途请求上限。设小了排队、设大了撞上游 429，所以做成环境变量可调，
# 压测时用它找出「开始出错的那个并发」——那个点才是真实的容量上限。
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "8"))
_slots = threading.BoundedSemaphore(MAX_INFLIGHT)

MOCK = os.environ.get("MOCK_LLM", "").strip().lower() in ("1", "true", "yes", "on")
MOCK_DELAY = float(os.environ.get("MOCK_DELAY", "0.05"))

_client = None
_model = None
_model_lock = threading.Lock()

# 每会话一把写锁。压测实测：40 个并发里有 4 个报 400，
# 根因是默认会话名只到「秒」，同一秒进来的请求拿到同一个名字，
# 于是并发写同一个 json —— 一方读到另一方写了一半的文件，JSONDecodeError。
# JSONDecodeError 是 ValueError 的子类，还被当成「参数错误」返回 400，
# 把「我的并发 bug」伪装成了「客户端乱传」，查起来会绕很远。
_session_locks = {}
_session_locks_guard = threading.Lock()


def _new_session_id():
    """时间戳 + 6 位随机后缀。可读性保留，唯一性补上。"""
    return f"{datetime.now():%Y-%m-%d %H-%M-%S}-{uuid.uuid4().hex[:6]}"


def _lock_for(sid):
    with _session_locks_guard:
        if sid not in _session_locks:
            _session_locks[sid] = threading.Lock()
        return _session_locks[sid]


def get_client():
    """懒初始化：压测跑 500 个请求也只探活一次。"""
    global _client, _model
    with _model_lock:
        if _client is None:
            _client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                             base_url=os.environ.get("DASHSCOPE_BASE_URL"))
            _model, _ = resolve_model(_client, verbose=False)
        return _client, _model


def _mock_stream():
    """假模型：固定延迟 + 固定答案。用来把「服务层开销」和「上游延迟」分开。"""
    time.sleep(MOCK_DELAY)
    yield ("delta", "满 99 元免运费【来源：配送-运费与包邮门槛】。")
    yield ("usage", {"prompt_tokens": 100, "completion_tokens": 20})


def _parse(body):
    msg = (body or {}).get("message") or (body or {}).get("q") or ""
    msg = str(msg).strip()
    if not msg:
        raise ValueError("message 不能为空")
    if len(msg) > 500:
        # 限长不是洁癖：超长问题会把 system 里的历史也拖长，
        # 一个请求就能吃掉几万 token，等于给了一个免费 DoS 入口
        raise ValueError("message 超过 500 字")
    return msg


def _handle(session_id, message, nickname, character, stream):
    """一次完整问答。返回 (事件生成器, sources, model_used)。

    这里**不吞异常**：压测要能区分 5xx（我的锅）和 503（限流的锅），
    全包成 200 会让错误率永远是 0，那种压测没有意义。
    """
    system, sources = core.build_system_message(message, nickname, character)
    # 零命中兜底（同 app.py）：检索不到资料就不调模型，给定稿话术。
    # 这不是「省 token」的优化，是断掉编造路径——实测模型会无视知识块里的禁令。
    if core.RETRIEVAL != "off" and not sources:
        def _zero():
            yield ("delta", core.zero_hit_reply(message))
        return _zero(), [], "none(zero-hit)"
    history = []
    if session_id:
        try:
            history = core.recent_messages(core.load_session(session_id)["messages"])
        except FileNotFoundError:
            history = []

    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": message}]

    if MOCK:
        gen = _mock_stream()
        model_used = "mock"
    else:
        client, model_used = get_client()
        gen = core.call_llm(client, model_used, messages, stream=True)
    return gen, sources, model_used


@app.get("/health")
def health():
    return jsonify(ok=True, mock=MOCK, max_inflight=MAX_INFLIGHT,
                   model="mock" if MOCK else (_model or "unresolved"),
                   retrieval=core.RETRIEVAL)


@app.post("/chat")
def chat():
    acquired = _slots.acquire(blocking=False)
    if not acquired:
        # 主动拒绝比让请求在上游超时好：用户 200 ms 就知道要重试，
        # 而不是等 30 秒拿到一个 504
        return jsonify(error="busy", max_inflight=MAX_INFLIGHT), 503, {"Retry-After": "1"}

    t0 = time.perf_counter()
    sid = None
    try:
        body = request.get_json(silent=True) or {}
        message = _parse(body)
        sid = body.get("session_id") or _new_session_id()
        nickname = body.get("nickname") or "果小蜜"
        character = body.get("character") or "热情、专业、懂生活、像闺蜜一样贴心"

        gen, sources, model_used = _handle(sid, message, nickname, character, stream=False)
        answer, usage, calls = "", None, []
        for kind, payload in gen:
            if kind == "delta":
                answer += payload
            elif kind == "tools":
                calls += payload
            elif kind == "usage":
                usage = payload

        # 落盘：读旧的再追加，避免覆盖掉历史里已有的轮次。
        # 读-改-写必须整段互斥，否则并发下会丢轮次（后写的覆盖先写的）。
        with _lock_for(sid):
            try:
                msgs = core.load_session(sid)["messages"]
            except FileNotFoundError:
                msgs = []
            msgs += [{"role": "user", "content": message},
                     {"role": "assistant", "content": answer}]
            core.save_session(sid, msgs, nickname, character)

        elapsed = time.perf_counter() - t0
        core.append_request_log({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": "http", "session": sid, "q": message,
            "answer": answer, "sources": sources,
            "tools": [c["name"] for c in calls],
            "model": model_used, "in_tokens": (usage or {}).get("prompt_tokens"),
            "out_tokens": (usage or {}).get("completion_tokens"),
            "elapsed_s": round(elapsed, 3), "ok": True,
        })
        return jsonify(answer=answer, sources=sources, session_id=sid,
                       tools=[c["name"] for c in calls], model=model_used,
                       usage=usage, elapsed_s=round(elapsed, 3))
    except json.JSONDecodeError as e:
        # 必须排在 ValueError 之前：JSONDecodeError 是它的子类，
        # 顺序反了会把「会话文件损坏」这类服务端问题报成 400 客户端错误
        return jsonify(error="session_corrupt", message=str(e)), 500
    except ValueError as e:
        return jsonify(error="bad_request", message=str(e)), 400
    except Exception as e:
        core.append_request_log({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": "http", "session": sid, "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })
        return jsonify(error="internal", message=f"{type(e).__name__}: {e}"), 500
    finally:
        if acquired:
            _slots.release()


@app.post("/chat/stream")
def chat_stream():
    if not _slots.acquire(blocking=False):
        return jsonify(error="busy"), 503, {"Retry-After": "1"}
    try:
        body = request.get_json(silent=True) or {}
        message = _parse(body)
        sid = body.get("session_id") or _new_session_id()
        gen, sources, model_used = _handle(sid, message,
                                           body.get("nickname") or "果小蜜",
                                           body.get("character") or "热情、专业、懂生活、像闺蜜一样贴心",
                                           stream=True)
    except ValueError as e:
        _slots.release()
        return jsonify(error="bad_request", message=str(e)), 400
    except Exception:
        _slots.release()
        raise

    def sse():
        answer = ""
        try:
            yield f"event: meta\ndata: {json.dumps({'session_id': sid, 'sources': sources, 'model': model_used}, ensure_ascii=False)}\n\n"
            for kind, payload in gen:
                if kind == "delta":
                    answer += payload
                    yield f"data: {json.dumps({'delta': payload}, ensure_ascii=False)}\n\n"
                elif kind == "tools":
                    yield f"event: tools\ndata: {json.dumps([c['name'] for c in payload], ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        finally:
            _slots.release()
            try:
                core.save_session(sid, [{"role": "user", "content": message},
                                        {"role": "assistant", "content": answer}])
            except Exception:
                pass

    return Response(stream_with_context(sse()), mimetype="text/event-stream")


@app.get("/sessions")
def sessions():
    return jsonify(sessions=core.list_sessions())


@app.get("/sessions/<sid>")
def session_detail(sid):
    try:
        return jsonify(core.load_session(sid))
    except FileNotFoundError:
        return jsonify(error="not_found"), 404


@app.delete("/sessions/<sid>")
def session_delete(sid):
    return jsonify(deleted=core.delete_session(sid))


if __name__ == "__main__":
    # 默认单进程：压测要测的是「一个 worker 能扛多少」，
    # 多进程会把队列和信号量复制成 N 份，MAX_INFLIGHT 就失去意义
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), threaded=True)
