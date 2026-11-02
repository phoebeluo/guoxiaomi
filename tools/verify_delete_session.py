"""验证 app.py 中 delete_session 的行为（不消耗 API、不启动 Streamlit、不动真实会话数据）。

做法与 verify_recent_messages.py 一致：用 AST 从 app.py 里**只抽取**目标函数，
在独立命名空间里执行，绕开 Streamlit 页面代码。
SESSIONS_DIR 被重定向到临时目录，真实 data/sessions/ 全程只读。

用法：
    E:/guoxiaomi/.venv/Scripts/python.exe tools/verify_delete_session.py
"""

import ast
import json
import os
import shutil
import sys
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
APP = os.path.join(PROJECT, "app.py")

WANT_FUNCS = ["generate_session_name", "save_messages", "delete_session", "load_sessions"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'[通过]' if ok else '[失败]'} {name}" + (f"   {detail}" if detail else ""))


class FakeSessionState(dict):
    """既支持 st.session_state.foo，也支持 st.session_state['foo']。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class FakeSt:
    def __init__(self):
        self.session_state = FakeSessionState()
        self.rerun_called = False
        self.errors = []

    def rerun(self):
        self.rerun_called = True

    def error(self, msg):
        self.errors.append(msg)


def load_symbols(path, names):
    """只执行 app.py 里指定的函数定义，不碰其它代码。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]

    if len(keep) != len(names):
        got = {n.name for n in keep}
        raise RuntimeError(f"app.py 中缺少函数：{set(names) - got}")

    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    import core                      # 2026-11-02：app.py 的函数改为调用 core，命名空间里必须补上
    ns = {"os": os, "json": json, "core": core}
    from datetime import datetime
    ns["datetime"] = datetime
    exec(compile(mod, path, "exec"), ns)
    return ns


def write_session(folder, name, payload=None):
    os.makedirs(folder, exist_ok=True)
    payload = payload or {"nickname": "果小蜜", "character": "贴心",
                          "current_session": name,
                          "messages": [{"role": "user", "content": "问"},
                                       {"role": "assistant", "content": "答"}]}
    with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 66)
    print("阶段 1 · 源码抽取")
    print("=" * 66)

    if not os.path.exists(APP):
        print(f"[失败] 找不到 {APP}")
        return 1

    ns = load_symbols(APP, WANT_FUNCS)
    check("成功抽取 4 个目标函数", True, ", ".join(WANT_FUNCS))

    tmp = tempfile.mkdtemp(prefix="gxm_verify_")
    ns["SESSIONS_DIR"] = tmp
    # 2026-11-02：落盘已下沉到 core，重定向必须打在 core 上，
    # 否则这个脚本会去动真实的 data/sessions —— 那比不测还糟
    import core
    core.SESSIONS_DIR = tmp
    fake_st = FakeSt()
    ns["st"] = fake_st

    print(f"  临时会话目录：{tmp}")
    print()
    print("=" * 66)
    print("阶段 2 · 行为验证（假数据，不花钱；真实 data/sessions 未被触碰）")
    print("=" * 66)

    # ---------- 用例 1：删除【非当前】会话 ----------
    print("\n[用例1] 删除一个【非当前】会话")
    write_session(tmp, "old_session")
    write_session(tmp, "cur_session")
    fake_st.session_state.clear()
    fake_st.session_state.messages = [{"role": "user", "content": "当前会话的问"}]
    fake_st.session_state.current_session = "cur_session"
    fake_st.session_state.nickname = "果小蜜"
    fake_st.session_state.character = "贴心"

    fake_st.rerun_called = False
    ns["delete_session"]("old_session")
    check("用例1 目标文件已删除",
          not os.path.exists(os.path.join(tmp, "old_session.json")))
    check("用例1 当前会话文件仍在",
          os.path.exists(os.path.join(tmp, "cur_session.json")))
    check("用例1 已触发 rerun（侧边栏需刷新）", fake_st.rerun_called is True)
    check("用例1 未误清当前会话的内存状态",
          fake_st.session_state.messages == [{"role": "user", "content": "当前会话的问"}])

    # ---------- 用例 2：删除【当前】会话（核心疑点） ----------
    print("\n[用例2] 删除【当前】会话 —— 期望：文件从磁盘消失")
    cur_path = os.path.join(tmp, "cur_session.json")
    check("用例2 前置条件：删除前文件存在", os.path.exists(cur_path))

    fake_st.rerun_called = False
    ns["delete_session"]("cur_session")

    still_there = os.path.exists(cur_path)
    check("用例2 目标文件已从磁盘删除", not still_there,
          "文件被重新写回磁盘，删除失效" if still_there else "")

    if still_there:
        with open(cur_path, encoding="utf-8") as f:
            resurrected = json.load(f)
        n_msg = len(resurrected.get("messages", []))
        check("用例2 复活内容是否为删除前的完整会话", n_msg == 1,
              f"复活了 {n_msg} 条消息（即删除前的内容）")

    check("用例2 已触发 rerun", fake_st.rerun_called is True)
    check("用例2 内存中的 messages 已清空",
          fake_st.session_state.messages == [])
    check("用例2 current_session 已换新名",
          fake_st.session_state.current_session != "cur_session",
          f"新名 = {fake_st.session_state.current_session}")

    # ---------- 用例 3：删除后 UI 会话列表是否还看得到 ----------
    print("\n[用例3] 删除当前会话后，侧边栏会话列表是否还能看到它")
    listed = ns["load_sessions"]()
    check("用例3 被删的当前会话应已不在列表中",
          "cur_session" not in listed,
          f"列表 = {listed}" if "cur_session" in listed else "")

    # ---------- 用例 4：删除不存在的会话，不应抛异常、不应 rerun ----------
    print("\n[用例4] 删除不存在的会话")
    fake_st.rerun_called = False
    try:
        ns["delete_session"]("never_existed")
        check("用例4 不抛异常", True)
    except Exception as e:
        check("用例4 不抛异常", False, f"{type(e).__name__}: {e}")
    check("用例4 目标不存在时不触发 rerun", fake_st.rerun_called is False)

    # ---------- 用例 5：全程不应进入 except 分支 ----------
    print("\n[用例5] 异常处理分支")
    check("用例5 全程未调用 st.error（说明没有异常被吞）",
          fake_st.errors == [], f"捕获到错误：{fake_st.errors}" if fake_st.errors else "")

    shutil.rmtree(tmp, ignore_errors=True)

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=" * 66)
    print(f"合计 {len(results)} 项，失败 {len(failed)} 项")
    if failed:
        for n in failed:
            print(f"   失败：{n}")
    else:
        print("全部通过")
    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
