"""验证 app.py 中 recent_messages 的行为（不消耗 API、不启动 Streamlit）。

做法：用 AST 从 app.py 里**只抽取** recent_messages 函数和 MAX_HISTORY 常量，
在独立命名空间里执行 —— 避免连带执行 app.py 里的 Streamlit 页面代码。

用法：
    E:/guoxiaomi/.venv/Scripts/python.exe verify_recent_messages.py
"""

import ast
import copy
import os
import sys

APP = r"E:\guoxiaomi\app.py"

WANT_FUNC = "recent_messages"
WANT_CONST = "MAX_HISTORY"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'[通过]' if ok else '[失败]'} {name}" + (f"   {detail}" if detail else ""))


def load_symbols(path):
    """只执行 app.py 里的 recent_messages 与 MAX_HISTORY，不碰其它代码。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == WANT_FUNC:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == WANT_CONST:
                    keep.append(node)

    if not keep:
        return None, src

    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, path, "exec"), ns)
    return ns, src


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


def main():
    print("=" * 62)
    print("阶段 1 · 源码级检查")
    print("=" * 62)

    if not os.path.exists(APP):
        print(f"[失败] 找不到 {APP}")
        return 1

    ns, src = load_symbols(APP)

    if ns is None:
        print(f"[失败] app.py 中找不到 {WANT_FUNC} 或 {WANT_CONST}")
        return 2

    check(f"{WANT_CONST} 已定义", WANT_CONST in ns, f"值 = {ns.get(WANT_CONST)}")
    check(
        f"{WANT_CONST} 是正整数",
        isinstance(ns.get(WANT_CONST), int) and ns.get(WANT_CONST) > 0,
    )
    check(f"{WANT_FUNC} 已定义且可调用", callable(ns.get(WANT_FUNC)))
    check(
        "调用点已改为 recent_messages",
        f"*{WANT_FUNC}(" in src.replace(" ", ""),
        "在 messages=[...] 中应当展开 recent_messages(...)",
    )

    fn = ns[WANT_FUNC]

    print()
    print("=" * 62)
    print("阶段 2 · 行为验证（假数据，不花钱）")
    print("=" * 62)

    # --- 用例 1：8 条 4 轮，截到 3 条 ---
    msgs = [u("第1轮提问"), a("第1轮回答"), u("第2轮提问"), a("第2轮回答"),
            u("第3轮提问"), a("第3轮回答"), u("第4轮提问"), a("第4轮回答")]
    snapshot = copy.deepcopy(msgs)
    try:
        out = fn(msgs, 3)
        check("用例1 返回长度 ≤ 3", len(out) <= 3, f"实际 {len(out)}")
        check("用例1 首条是 user", out and out[0]["role"] == "user",
              f"实际 {out[0]['role'] if out else '空'}")
        check("用例1 末条是最新一轮", out and out[-1]["content"] == "第4轮回答",
              f"实际 {out[-1]['content'] if out else '空'}")
    except Exception as e:
        check("用例1 不抛异常", False, f"{type(e).__name__}: {e}")

    # --- 用例 2：不得改动传入的原列表（最危险的那个坑）---
    check("用例2 原列表长度未变", len(msgs) == len(snapshot), f"原 {len(snapshot)} → 现 {len(msgs)}")
    check("用例2 原列表内容未被改动", msgs == snapshot,
          "若失败说明用户的真实历史被删除了" if msgs != snapshot else "")

    # --- 用例 3：历史不足上限，应原样返回 ---
    short = [u("问1"), a("答1")]
    try:
        out3 = fn(short, 10)
        check("用例3 不足上限时原样返回", out3 == short, f"返回 {len(out3)} 条")
    except Exception as e:
        check("用例3 不抛异常", False, f"{type(e).__name__}: {e}")

    # --- 用例 4：切片恰好落在 user 上，不应多丢 ---
    four = [u("问1"), a("答1"), u("问2"), a("答2")]
    try:
        out4 = fn(four, 2)
        check("用例4 正好从 user 开头时不额外丢弃",
              out4 == [u("问2"), a("答2")], f"返回 {out4}")
    except Exception as e:
        check("用例4 不抛异常", False, f"{type(e).__name__}: {e}")

    # --- 用例 5：空列表 ---
    try:
        out5 = fn([], 5)
        check("用例5 空列表返回空列表", out5 == [])
    except Exception as e:
        check("用例5 不抛异常", False, f"{type(e).__name__}: {e}")

    # --- 用例 6：异常历史（两条 user 连排）---
    weird = [u("问1"), a("答1"), u("问2"), u("问3"), a("答3")]
    try:
        out6 = fn(weird, 3)
        check("用例6 连续两条 user 时不崩", out6 and out6[0]["role"] == "user",
              f"返回 {len(out6)} 条")
    except Exception as e:
        check("用例6 不抛异常", False, f"{type(e).__name__}: {e}")

    # --- 用例 7：全是 assistant（极端）---
    try:
        out7 = fn([a("答1"), a("答2")], 5)
        check("用例7 全是 assistant 时返回空列表", out7 == [], f"返回 {out7}")
    except Exception as e:
        check("用例7 不抛异常", False, f"{type(e).__name__}: {e}")

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=" * 62)
    print(f"合计 {len(results)} 项，失败 {len(failed)} 项")
    if failed:
        for n in failed:
            print(f"   失败：{n}")
    else:
        print("全部通过")
    print("=" * 62)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
