"""HTTP 接口契约测试（2026-11-02 · W12）。

全部在 `MOCK_LLM=1` 下跑：假模型固定延迟，不花一分钱，也不受上游抖动影响。
被测的是**接口契约**——状态码、错误分类、并发保护、会话唯一性。

⚠️ 这里刻意**不做**「回答内容对不对」的断言。那是 `eval_run` / `eval_tools` 的职责。
接口测试混进质量断言，模型一换就得改测试，最后没人愿意跑。
"""

import importlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("MOCK_DELAY", "0.01")
    import api
    importlib.reload(api)                    # 让 MOCK_LLM 生效
    import core
    monkeypatch.setattr(core, "SESSIONS_DIR", str(tmp_path))
    api.app.config["TESTING"] = True
    return api.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["mock"] is True and body["ok"] is True


def test_chat_ok_and_persists_session(client, tmp_path):
    r = client.post("/chat", json={"message": "满多少免运费？"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"] and body["sources"]
    assert body["session_id"]
    # 会话必须真的落盘，否则「多轮」是假的
    from core import load_session
    assert load_session(body["session_id"])["messages"][0]["content"] == "满多少免运费？"


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {}, {"message": "x" * 600}])
def test_chat_rejects_bad_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 400


def test_session_ids_are_unique_under_concurrency(client):
    """压测实测过的真 bug：会话名只到「秒」，同一秒的并发请求撞同一个文件，
    读到半截 JSON 报 400。现在是时间戳 + 6 位随机后缀，这条断言锁住它。"""
    ids = []

    def hit(_):
        return client.post("/chat", json={"message": "坏果怎么赔？"}).get_json()["session_id"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        ids = list(ex.map(hit, range(16)))
    assert len(set(ids)) == len(ids)
    assert all(len(i.split("-")[-1]) == 6 for i in ids)


def test_returns_503_when_inflight_full(client):
    """在途满时立刻 503 + Retry-After，而不是让请求堆在上游超时。

    这里用信号量本身来制造「满」的状态，比真起 8 个线程快也更稳。
    """
    import api
    got = [api._slots.acquire(blocking=False) for _ in range(api.MAX_INFLIGHT)]
    try:
        r = client.post("/chat", json={"message": "满多少免运费？"})
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "1"
        assert r.get_json()["error"] == "busy"
    finally:
        for ok in got:
            if ok:
                api._slots.release()


def test_sessions_crud(client):
    sid = client.post("/chat", json={"message": "苹果多少钱一斤？"}).get_json()["session_id"]
    assert sid in client.get("/sessions").get_json()["sessions"]
    assert client.get(f"/sessions/{sid}").status_code == 200
    assert client.delete(f"/sessions/{sid}").get_json()["deleted"] is True
    assert client.get(f"/sessions/{sid}").status_code == 404
