"""数据库连接与建表（2026-11-02 · M14-A）。

## 入口只有三个

| 名字 | 用途 |
|---|---|
| `get_engine()` | 取（并缓存）连接池；测试通过 `GUOXIAOMI_DB` 换库文件 |
| `init_db()` | 建表；可重复调用 |
| `session_scope()` | 事务上下文：正常提交、异常回滚、结束关连接 |

## 为什么每次开连接都设 PRAGMA

SQLite 的 `PRAGMA foreign_keys` 是**连接级**开关，不是库级。
连接池复用的每一条新连接都得重新打开，否则「删会话不连带删消息」的 bug
只会在池子换连接之后才出现——本地测不出来，线上跑几天才暴露。
同理 WAL 也在这里设，保证任何入口拿到的连接行为一致。
"""

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT, "data", "guoxiaomi.db")

_engine = None
_SessionFactory = None


def db_path():
    """库文件路径。测试用 `GUOXIAOMI_DB=:memory:` 或临时文件覆盖。"""
    return os.environ.get("GUOXIAOMI_DB") or DEFAULT_DB_PATH


def _connect_url():
    path = db_path()
    if path == ":memory:":
        return "sqlite://"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return f"sqlite:///{path}"


def get_engine():
    global _engine, _SessionFactory
    if _engine is None:
        _engine = create_engine(
            _connect_url(),
            future=True,
            # Streamlit 与 Flask 可能在不同线程里用同一个引擎，
            # sqlite3 默认禁止跨线程复用连接
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(_engine, "connect")
        def _set_pragma(dbapi_conn, _rec):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # 默认是 OFF，不设等于没建 FK
            cur.execute("PRAGMA journal_mode=WAL")  # 读写不互斥
            cur.close()

        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def init_db():
    """建表。已经存在就跳过，所以可以每次启动都调。"""
    from db.models import Base

    Base.metadata.create_all(get_engine())
    return get_engine()


def session_scope():
    """事务上下文管理器：`with session_scope() as s:`，异常自动回滚。"""
    get_engine()
    return _SessionFactory()
