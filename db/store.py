"""会话库读写（2026-11-02 · M14-A/B）。

对外两套语义，刻意分开：

| 函数 | 语义 | 谁在用 |
|---|---|---|
| `upsert_session` | **整份覆盖**，与原来的 JSON 落盘等价 | `core.save_session`（Streamlit / Flask 现有调用） |
| `append_turn` | **增量追加**，一个事务写一轮问答 + 工具调用 + 用量 | 新的写入路径，数据库真正的收益 |

为什么保留「整份覆盖」：`app.py` 每轮把内存里的 `st.session_state.messages` 交出来，
它并不知道「这一轮新增了哪几条」。要让上层改成增量，得先改 Streamlit 的状态管理，
属于另一件事。所以先用覆盖语义保证**行为不变**，再单独提供增量入口——
迁移的第一原则是先对齐行为，再谈优化。
"""

import json
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import select

from db import get_engine, init_db, session_scope
from db.models import (
    DEFAULT_CHARACTER,
    DEFAULT_NICKNAME,
    LlmUsage,
    Message,
    Session,
    ToolCall,
)


@contextmanager
def _tx():
    """一个写事务：正常提交，异常整体回滚，最后一定关连接。"""
    get_engine()
    s = session_scope()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _readonly():
    get_engine()
    return session_scope()


# ------------------------------------------------------------------ 读
def to_dict(session):
    """转成与 `data/sessions/*.json` **完全一致**的字典。

    这是迁移的关键约束：`api.py` 的 `/sessions/<id>` 直接把返回值 jsonify 出去，
    `app.py` 读 `data["messages"]`。结构差一个键，上层就静默坏掉。
    """
    return {
        "nickname": session.nickname,
        "character": session.character,
        "current_session": session.name,
        "messages": [{"role": m.role, "content": m.content} for m in session.messages],
    }


def list_session_names():
    """按名字倒序（名字就是时间戳，倒序即最新在前），与 json 版排序一致。"""
    s = _readonly()
    try:
        rows = s.execute(select(Session.name).order_by(Session.name.desc())).scalars().all()
        return list(rows)
    finally:
        s.close()


def get_session_dict(name):
    """取一个会话；不存在抛 `FileNotFoundError`，与 json 版保持同一异常类型。"""
    s = _readonly()
    try:
        obj = s.execute(select(Session).where(Session.name == name)).scalar_one_or_none()
        if obj is None:
            raise FileNotFoundError(f"session not found: {name}")
        return to_dict(obj)
    finally:
        s.close()


def stats():
    """给「库里有几张表、多少行」这类健康检查用。"""
    s = _readonly()
    try:
        return {
            "sessions": s.query(Session).count(),
            "messages": s.query(Message).count(),
            "tool_calls": s.query(ToolCall).count(),
            "llm_usage": s.query(LlmUsage).count(),
        }
    finally:
        s.close()


# ------------------------------------------------------------------ 写
def upsert_session(name, messages, nickname=DEFAULT_NICKNAME,
                   character=DEFAULT_CHARACTER):
    """整份覆盖一个会话（与 `core.save_session` 等价），返回会话名。

    覆盖的实现是「删旧消息 + 插新消息」而不是逐条 diff：
    会话只有 5~10 轮，diff 的复杂度换不来收益，而且 diff 写错会留下错位的轮次。
    """
    init_db()
    now = datetime.now()
    with _tx() as s:
        obj = s.execute(select(Session).where(Session.name == name)).scalar_one_or_none()
        if obj is None:
            obj = Session(name=name, created_at=now)
            s.add(obj)
        obj.nickname = nickname
        obj.character = character
        obj.updated_at = now
        # 先落主表拿到 id，才能给子表填外键
        s.flush()

        s.execute(
            Message.__table__.delete().where(Message.session_id == obj.id)
        )
        for i, m in enumerate(messages, start=1):
            s.add(Message(
                session_id=obj.id,
                turn_no=i,
                role=str(m.get("role", ""))[:16],
                content=str(m.get("content", "")),
                created_at=now,
            ))
        obj.message_count = len(messages)
    return name


def append_turn(name, user_text, assistant_text, tool_calls=None, usage=None,
                nickname=DEFAULT_NICKNAME, character=DEFAULT_CHARACTER):
    """增量追加一轮问答。**一个事务**写完 5 类数据。

    参数：
        tool_calls: [{"tool_name","arguments","result","ok","latency_ms"}]，挂到本轮 assistant 消息
        usage:      {"model","prompt_tokens","completion_tokens","latency_ms"}

    返回本轮两条消息的 id `(user_id, assistant_id)`。

    为什么必须一个事务：中途失败若已提交 user 消息，会话就变成「有提问没回答」。
    原来 JSON 方案靠整份重写规避，代价是第 n 轮要重写前 n-1 轮（O(n) 写放大）。
    """
    init_db()
    now = datetime.now()
    with _tx() as s:
        obj = s.execute(select(Session).where(Session.name == name)).scalar_one_or_none()
        if obj is None:
            obj = Session(name=name, nickname=nickname, character=character,
                          created_at=now, updated_at=now)
            s.add(obj)
            s.flush()

        base = obj.message_count  # 已有条数即下一个 turn_no 的起点
        u = Message(session_id=obj.id, turn_no=base + 1, role="user",
                    content=user_text, created_at=now)
        a = Message(session_id=obj.id, turn_no=base + 2, role="assistant",
                    content=assistant_text, created_at=now)
        s.add_all([u, a])
        s.flush()  # 拿到 a.id

        for tc in (tool_calls or []):
            s.add(ToolCall(
                message_id=a.id,
                tool_name=tc.get("tool_name", ""),
                arguments=json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                result=str(tc.get("result", "")),
                ok=bool(tc.get("ok", True)),
                latency_ms=int(tc.get("latency_ms", 0)),
                created_at=now,
            ))

        if usage:
            s.add(LlmUsage(
                session_id=obj.id,
                model=usage.get("model", ""),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=int(usage.get("latency_ms", 0)),
                created_at=now,
            ))

        obj.message_count = base + 2
        obj.updated_at = now
        return u.id, a.id


def delete_session(name):
    """删会话，子表靠 `ON DELETE CASCADE` 连带删除。返回是否真的删掉了。"""
    with _tx() as s:
        obj = s.execute(select(Session).where(Session.name == name)).scalar_one_or_none()
        if obj is None:
            return False
        s.delete(obj)
        return True
