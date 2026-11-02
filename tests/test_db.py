"""会话库测试（2026-11-02 · M14-A）。

测的是**数据库带来的东西**，不是「能不能读写」：
外键真的生效了吗、事务真的原子吗、两层后端行为真的一致吗。
一个只有 CRUD 的测试集，换成 JSON 文件也能全绿，那它就没测到迁移的意义。
"""

import json

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

import core
import db
from db import store


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """每个用例一个全新库文件；用完销毁并清掉模块级引擎缓存。"""
    monkeypatch.setenv("GUOXIAOMI_DB", str(tmp_path / "t.db"))
    db._engine = None
    db._SessionFactory = None
    eng = db.init_db()
    yield eng
    eng.dispose()
    db._engine = None
    db._SessionFactory = None


# ------------------------------------------------------------------ 结构
def test_four_tables_created(fresh_db):
    names = set(inspect(fresh_db).get_table_names())
    assert {"sessions", "messages", "tool_calls", "llm_usage"} <= names


def test_indexes_exist(fresh_db):
    """索引不是装饰：会话详情页的查询就是 `session_id + turn_no`，没索引就是全表扫。"""
    idx = {i["name"] for i in inspect(fresh_db).get_indexes("messages")}
    assert "ix_message_session_turn" in idx


def test_foreign_key_pragma_on(fresh_db):
    """SQLite 的 FK 默认关闭，靠 connect 事件打开——断了这个开关，级联就静默失效。"""
    with fresh_db.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# ------------------------------------------------------------------ 外键与级联
def test_delete_session_cascades(fresh_db):
    store.upsert_session("s1", [{"role": "user", "content": "种子"}])
    store.append_turn(
        "s1", "满多少免运费", "满 99 免运费",
        tool_calls=[{"tool_name": "query_order", "arguments": {}, "result": "{}"}],
        usage={"model": "qwen", "prompt_tokens": 10, "completion_tokens": 5},
    )
    before = store.stats()
    assert before == {"sessions": 1, "messages": 3, "tool_calls": 1, "llm_usage": 1}

    assert store.delete_session("s1") is True
    assert store.stats() == {"sessions": 0, "messages": 0, "tool_calls": 0, "llm_usage": 0}


def test_delete_missing_returns_false(fresh_db):
    assert store.delete_session("不存在") is False


# ------------------------------------------------------------------ 事务
def test_append_turn_is_atomic(fresh_db):
    """工具调用写失败时，本轮的两条消息也不能留下——否则会话变成「有提问没回答」。"""
    store.upsert_session("s1", [{"role": "user", "content": "种子"}])
    before = store.stats()
    with pytest.raises(IntegrityError):
        # tool_name 非空约束被违反，异常发生在 commit 之前的 flush
        store.append_turn("s1", "q", "a", tool_calls=[{"tool_name": None}])
    assert store.stats() == before


def test_turn_no_unique(fresh_db):
    """同一会话同一轮次不能写两遍：重试要报错，而不是悄悄多出一条。"""
    from sqlalchemy import select

    from db.models import Message, Session

    store.upsert_session("s1", [{"role": "user", "content": "a"}])
    s = db.session_scope()
    try:
        obj = s.execute(select(Session).where(Session.name == "s1")).scalar_one()
        s.add(Message(session_id=obj.id, turn_no=1, role="user", content="dup"))
        with pytest.raises(IntegrityError):
            s.commit()
        s.rollback()
    finally:
        s.close()


# ------------------------------------------------------------------ 行为等价
def test_roundtrip_matches_json_shape(fresh_db):
    """返回结构必须与 `data/sessions/*.json` 一模一样，否则 api.py 静默坏掉。"""
    msgs = [
        {"role": "user", "content": "坏果怎么赔？"},
        {"role": "assistant", "content": "24 小时内拍照联系客服"},
    ]
    store.upsert_session("2026-11-02 10-00-00", msgs, "果小蜜", "贴心")
    got = store.get_session_dict("2026-11-02 10-00-00")
    assert got == {
        "nickname": "果小蜜",
        "character": "贴心",
        "current_session": "2026-11-02 10-00-00",
        "messages": msgs,
    }
    # 与 json 后端的键集合完全一致
    assert set(got) == {"nickname", "character", "current_session", "messages"}


def test_upsert_overwrites_not_appends(fresh_db):
    """覆盖语义：第二次写 1 条要得到 1 条，不是 2 条。"""
    store.upsert_session("s1", [{"role": "user", "content": "1"}])
    store.upsert_session("s1", [{"role": "user", "content": "2"}])
    data = store.get_session_dict("s1")
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "2"


def test_list_is_newest_first(fresh_db):
    for name in ["2026-11-01 09-00-00", "2026-11-02 09-00-00", "2026-10-31 09-00-00"]:
        store.upsert_session(name, [{"role": "user", "content": "x"}])
    assert store.list_session_names() == [
        "2026-11-02 09-00-00", "2026-11-01 09-00-00", "2026-10-31 09-00-00",
    ]


def test_load_missing_raises_filenotfound(fresh_db):
    """异常类型必须与 json 后端一致：api.py 靠 `except FileNotFoundError` 返回 404。"""
    with pytest.raises(FileNotFoundError):
        store.get_session_dict("没有这个会话")


# ------------------------------------------------------------------ core 分发
def test_core_dispatch_to_db(fresh_db, monkeypatch):
    """把 core 切到 db 后端，跑一遍原来 json 后端的不变式。"""
    monkeypatch.setattr(core, "SESSION_BACKEND", "db")
    name = "2026-11-02 12-00-00"
    core.save_session(name, [{"role": "user", "content": "满多少免运费"}], "果小蜜", "贴心")
    assert name in core.list_sessions()
    assert core.load_session(name)["messages"][0]["content"] == "满多少免运费"
    assert core.delete_session(name) is True
    assert core.delete_session(name) is False
    with pytest.raises(FileNotFoundError):
        core.load_session(name)
