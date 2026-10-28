"""O4 补完（L0-3）：验证 app.py 的错误分类与降级话术真的生效。

背景（对应总纲二期 O1–O3 的唯一遗留项 O4）
----------------------------------------
`app.py` 从 2026-10-13 起把 API 调用包了 `try/except` 并做了错误五分类
（quota / timeout / connection / content / other）+ 每类一句降级话术。
但**从未验证过这套映射对不对**——O4 一直挂着。本脚本补上。

为什么不能用真 API 来测
----------------------
要触发 `quota` 就得把额度耗光，要触发 `timeout` 就得真的断网——
既费额度又不可复现。所以改成**假异常单测**：
把异常对象喂给 app.py 里**真实抽取出来**的 `classify_error()`，
断言它落进预期的桶，并断言该桶的降级话术满足两条对客纪律。

三条断言（每条都对着一条真实约束）
--------------------------------
1. **分类正确** —— 6 个样本异常各落进预期的桶
2. **不暴露内部原因** —— 降级话术里**不得出现**「额度 / quota / 403 / 超时 / timeout / 余额」等词
   （这是本项目的对客红线：出错时给一句人话，细节只进日志）
3. **话术齐全** —— 五个桶都有对应文案，且 `classify_error()` 的未知分支落到 `other` 而非崩溃

做法：与 verify_delete_session.py 一致，用 AST **只抽取**目标符号，不启动 Streamlit。

用法
----
    E:/guoxiaomi/.venv/Scripts/python.exe tools/verify_fallback.py
"""

import ast
import os
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

WANT_FUNCS = ["classify_error", "log_request"]
WANT_CONSTS = ["ERROR_KINDS", "FALLBACK_TEXT"]

# 降级话术里绝不允许出现的内部原因词（对客红线）
LEAK_WORDS = ["额度", "quota", "Quota", "403", "401", "余额", "欠费",
              "timeout", "Timeout", "超时", "异常", "Exception", "Error",
              "invalid", "校验", "拦截", "token", "Token"]

# (异常对象, 期望桶, 说明) —— 异常特征按 app.py ERROR_KINDS 的实际关键字构造
CASES = [
    (Exception("Error code: 403 - {'error': {'message': 'Free quota exhausted'}}"),
     "quota", "免费额度耗尽（403 + quota 关键字）"),
    (Exception("Error code: 403 - {'message': 'Arrearage'}"),
     "quota", "账号欠费（403 + Arrearage）"),
    (Exception("APITimeoutError: Request timed out."),
     "timeout", "请求超时"),
    (Exception("APIConnectionError: Connection error."),
     "connection", "网络连接失败"),
    (Exception("Error code: 400 - data_inspection_failed"),
     "content", "内容安全拦截（400 + data_inspection）"),
    (ValueError("some totally unrelated failure"),
     "other", "完全未知的异常 → 必须落到 other 兜底"),
]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'[通过]' if ok else '[失败]'} {name}" + (f"   {detail}" if detail else ""))


def load_symbols(path, func_names, const_names):
    """只执行 app.py 里指定的函数与常量赋值，不碰 Streamlit 页面代码。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in func_names)
            or (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in const_names
                        for t in n.targets))]

    ns = {"os": os, "json": __import__("json")}
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, path, "exec"), ns)

    missing = [n for n in func_names + const_names if n not in ns]
    if missing:
        raise RuntimeError(f"app.py 中缺少符号（可能已被重构/改名）：{missing}")
    return ns


def main():
    print("=" * 66)
    print("阶段 1 · 从 app.py 抽取 classify_error / ERROR_KINDS / FALLBACK_TEXT")
    print("=" * 66)
    ns = load_symbols(APP, WANT_FUNCS, WANT_CONSTS)
    classify_error = ns["classify_error"]
    fallback = ns["FALLBACK_TEXT"]
    error_kinds = ns["ERROR_KINDS"]
    print(f"  ERROR_KINDS 条目 : {len(error_kinds)}")
    print(f"  FALLBACK_TEXT 桶 : {sorted(fallback.keys())}")

    print()
    print("=" * 66)
    print("阶段 2 · 假异常 → 分类是否正确")
    print("=" * 66)
    for exc, want, desc in CASES:
        got = classify_error(exc)
        check(f"{desc}  →  {got}", got == want, f"期望 {want}")

    print()
    print("=" * 66)
    print("阶段 3 · 降级话术是否泄露内部原因（对客红线）")
    print("=" * 66)
    for kind in sorted(fallback.keys()):
        text = fallback[kind]
        leaked = [w for w in LEAK_WORDS if w in text]
        check(f"{kind:<11} 话术无内部原因", not leaked,
              f"泄露：{leaked}" if leaked else "")

    print()
    print("=" * 66)
    print("阶段 4 · 话术完整性（五分类都要有文案，且非空）")
    print("=" * 66)
    for kind in ("quota", "timeout", "connection", "content", "other"):
        has = isinstance(fallback.get(kind), str) and fallback[kind].strip()
        check(f"{kind:<11} 有降级话术", bool(has), "" if has else "缺失或为空")

    print()
    print("=" * 66)
    ok_all = all(r[1] for r in results)
    total, passed = len(results), sum(1 for r in results if r[1])
    print(f"结果：{passed}/{total} 通过")
    print("=" * 66)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
