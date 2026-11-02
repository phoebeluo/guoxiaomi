"""会话库的 4 张表（2026-11-02 · M14-A）。

## 为什么不把会话存成一个大 JSON 列

最省事的做法是一张表两列：`name` + `payload TEXT`。但这样就白用了数据库——
唯一能用的操作还是「整份读出来 / 整份写回去」，跟现在的 JSON 文件没有区别。
拆成 4 张表的理由是**每个实体的生命周期和访问模式不同**：

| 表 | 谁写 | 访问模式 | 为什么单独一张 |
|---|---|---|---|
| `sessions` | 每轮更新 | 列表页频繁扫 | 列表页只读 `name/updated_at/message_count`，不能把整段正文读出来 |
| `messages` | 每轮追加 | 按会话顺序读 | 是「会话体」，1:N，需要 `(session_id, turn_no)` 唯一约束防重复 |
| `tool_calls` | 工具触发时 | 按消息回查 | 一条 assistant 消息可能调 **多次** 工具，塞不进固定列 |
| `llm_usage` | 每次调模型 | 按月聚合算成本 | 与消息不同步：模型可能重试，一次回答对应 1~N 条用量 |

## 两个 SQLite 的关键设置

1. **外键默认是关的**。SQLite 的历史包袱：`PRAGMA foreign_keys` 默认 OFF，
   建了 FK 也不会生效，删会话不会连带删消息，数据库会慢慢长出一堆孤儿行。
   所以 `db/__init__.py` 在每次建连接时显式打开。
2. **WAL 模式**。默认的 rollback journal 下，写事务会阻塞读事务；
   列表页和正在写的问答撞上就会等锁。WAL 让读写不互斥。

## 事务边界

一次完整问答 = 写 1 条 user 消息 + 1 条 assistant 消息 + N 条工具调用 + 1 条用量
+ 更新会话 `updated_at` / `message_count`。这 5 类写必须**同一个事务**。
分开提交就会出现「有提问没回答」的会话——原来 JSON 方案靠「整份重写」掩盖了这个问题，
代价是 O(n) 写放大：第 20 轮要把前 19 轮全部重写一遍。
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DEFAULT_NICKNAME = "果小蜜"
DEFAULT_CHARACTER = "热情、专业、懂生活、像闺蜜一样贴心"


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    # name 才是业务主键（沿用 JSON 文件名 `2026-10-29 22-55-37`），
    # 数字 id 只给外键用。不把时间戳当主键，是因为同一秒可能开两个会话。
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    nickname: Mapped[str] = mapped_column(String(32), default=DEFAULT_NICKNAME)
    character: Mapped[str] = mapped_column(String(200), default=DEFAULT_CHARACTER)
    # 冗余计数：列表页要显示条数，但不该为此 COUNT(*) 一遍 messages
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, index=True
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.turn_no",
    )
    usages: Mapped[list["LlmUsage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        # 同一会话里轮次不能重复：重试写同一条要报错，而不是悄悄写两遍
        UniqueConstraint("session_id", "turn_no", name="uq_message_turn"),
        # 读会话时的主查询：WHERE session_id=? ORDER BY turn_no
        Index("ix_message_session_turn", "session_id", "turn_no"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    turn_no: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant / system
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    session: Mapped["Session"] = relationship(back_populates="messages")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_toolcall_message", "message_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    tool_name: Mapped[str] = mapped_column(String(64))
    # 参数和结果都按 JSON 字符串存：3 个工具的参数结构互不相同，
    # 拆成列等于给每个工具建一张表，而工具是会增删的
    arguments: Mapped[str] = mapped_column(Text, default="{}")
    result: Mapped[str] = mapped_column(Text, default="")
    ok: Mapped[bool] = mapped_column(default=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    message: Mapped["Message"] = relationship(back_populates="tool_calls")


class LlmUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (Index("ix_usage_session_created", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    model: Mapped[str] = mapped_column(String(64))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    session: Mapped["Session"] = relationship(back_populates="usages")
