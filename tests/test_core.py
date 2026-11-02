"""core.py 的契约测试（2026-11-02 · W12）。

核心层存在的唯一理由是「Streamlit 和 Flask 共用一份逻辑」，
所以最要紧的一条测试不是功能，而是**依赖边界**：core 不许 import streamlit。
一旦有人为了省事破了这条，这层就退化成第二个 app.py，而测试不会变红——
所以必须有断言盯着。
"""

import ast
import os
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import core


def test_core_does_not_import_streamlit():
    """核心层不能依赖 Streamlit。

    用 AST 扫 import，而不是 `import core; assert 'streamlit' not in sys.modules`——
    后者会被「别的测试先 import 了 streamlit」污染，永远为真。
    """
    src = open(os.path.join(PROJECT, "core.py"), encoding="utf-8").read()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "streamlit" not in names


def test_system_message_contains_prompt_and_knowledge():
    system, sources = core.build_system_message("满多少免运费？")
    assert "果小蜜" in system
    assert "## 知识库全文" in system
    assert sources and all(s.endswith(".md") for s in sources)
    # 检索档下 system 必须远小于全量注入，否则等于没做检索
    full = core.build_knowledge_block("满多少免运费？", mode="off")[0]
    assert len(system) < len(full) / 3


def test_retrieval_modes_agree_on_top1_roughly():
    """vector 与 bm25 是两条不同的路，不要求排序一致，
    但都必须把「运费」相关篇目带进前三——这是业务上的硬要求。"""
    for mode in ("vector", "bm25"):
        _, sources = core.build_knowledge_block("满多少免运费", mode=mode)
        assert any("运费" in s or "包邮" in s for s in sources), (mode, sources)


def test_recent_messages_starts_with_user_and_truncates():
    msgs = [{"role": "assistant", "content": f"a{i}"} for i in range(3)]
    msgs += [{"role": "user", "content": f"u{i}"} for i in range(3)]
    out = core.recent_messages(msgs, 4)
    assert out[0]["role"] == "user"
    assert len(out) <= 4
    # 全部是 assistant 时必须返回空，而不是硬塞一条 assistant 给模型（会 400）
    assert core.recent_messages([{"role": "assistant", "content": "x"}], 4) == []


def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "SESSIONS_DIR", str(tmp_path))
    name = "2026-11-02 12-00-00-ab12cd"
    msgs = [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]
    core.save_session(name, msgs, "果小蜜", "贴心")
    assert name in core.list_sessions()
    data = core.load_session(name)
    assert data["messages"] == msgs and data["nickname"] == "果小蜜"
    assert core.delete_session(name) is True
    assert core.delete_session(name) is False          # 幂等
    with pytest.raises(FileNotFoundError):
        core.load_session(name)


def test_call_llm_non_stream_collects_events():
    """call_llm(stream=False) 要把 delta / tools / usage 三类事件收成三样东西。"""
    class FakeDelta:
        def __init__(self, text):
            self.content = text

    class FakeChoice:
        def __init__(self, text):
            self.delta = FakeDelta(text)

    class FakeChunk:
        def __init__(self, text=None, usage=None):
            self.choices = [FakeChoice(text)] if text is not None else []
            self.usage = usage

    class Usage:                        # loop 用 getattr 读属性，所以这里不能用 dict
        prompt_tokens = 10
        completion_tokens = 3

    class FakeCompletions:
        def create(self, **kw):
            return iter([FakeChunk("满 99 "), FakeChunk("免运费"),
                         FakeChunk(usage=Usage())])

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    answer, usage, calls = core.call_llm(
        FakeClient(), "m", [{"role": "user", "content": "q"}],
        tools=None, stream=False)
    assert answer == "满 99 免运费"
    assert usage["completion_tokens"] == 3      # 累加值，不是最后一片      # 累加值，不是最后一片
    assert calls == []
