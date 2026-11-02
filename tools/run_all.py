"""一键零成本体检（L0-4）：把散落的校验器串成一条命令。

为什么需要
----------
`tools/` 下已经有十几个校验脚本，但它们**彼此独立、没有统一入口**：
改完一处要记得手动跑哪几个，全靠记性。而"一个被遗忘的校验器 = 没有校验器"。
本脚本把它们串成一条命令，统一报告 PASS/FAIL 与耗时，**全部零 API 调用**。

覆盖清单（顺序 = 从快到慢，先暴露语法/结构错误）
----------------------------------------------
  1. py_compile      —— 全部 .py 能否编译
  2. 语料冻结指纹    —— data/docs/ 是否被动过（SHA-256）
  3. eval_run --check—— 题目字段与正则合法性
  4. test_judge      —— 判据回归（22 例）
  5. check_kinds     —— kind/expect 搭配与分母分组
  6. check_prompt    —— 提示词是否残留"指向不存在能力"的表述
  7. check_injection —— 离线复现 system 消息构造
  8. audit_coverage  —— 知识库覆盖度审计
  9. audit_gold      —— 金标「独立支撑」审计（用判据判金标原文，2026-10-26 新增）
 10. check_consistency—— 文档/报告/实测三方数字口径
 11. verify_fallback —— 错误五分类与降级话术
 12. verify_recent_messages —— AST 抽单函数做无头单测
 13. verify_delete_session  —— 删会话行为验证
 14. report_ops      —— 运行侧遥测汇总（只读，恒通过）

⚠️ 不含 `verify_app_venv.py`：它要启动 Streamlit，慢且需要窗口环境；
   `--chat` 版本还会花 1 次 API。需要时请手动跑。

用法
----
    E:/guoxiaomi/.venv/Scripts/python.exe tools/run_all.py
    E:/guoxiaomi/.venv/Scripts/python.exe tools/run_all.py --verbose   # 显示每个脚本的输出
"""

import glob
import hashlib
import os
import py_compile
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
PY = sys.executable

# 语料冻结指纹（总纲 7.8 末；若 data/docs/ 变更，这里**必须**由人确认后更新）
DOCS_SHA256 = "591986c81ddbee2aca7909d5e0dd084e0098997292f5334efbdf1fd71e7495f8"
DOCS_EXPECT_COUNT = 45
DOCS_EXPECT_CHARS = 29563

VERBOSE = "--verbose" in sys.argv


def docs_fingerprint():
    files = sorted(glob.glob(os.path.join(PROJECT, "data", "docs", "*.md")))
    h = hashlib.sha256()
    chars = 0
    for p in files:
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(open(p, "rb").read())
        chars += len(open(p, encoding="utf-8").read())
    return len(files), chars, h.hexdigest()


def run_script(args, title):
    t0 = time.perf_counter()
    proc = subprocess.run([PY] + args, cwd=PROJECT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    tail = ""
    if not ok:
        # 失败时把最后若干行带出来，方便定位
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-12:])
    if VERBOSE and not ok:
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-40:])
    return ok, dt, tail


def main():
    results = []

    def record(title, ok, dt=None, detail=""):
        results.append((title, ok, dt, detail))
        flag = "[通过]" if ok else "[失败]"
        t = f"  {dt*1000:7.0f} ms" if dt is not None else "           "
        print(f"  {flag} {title}{t}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")

    print("=" * 70)
    print("果小蜜 · 一键零成本体检（不调用 API）")
    print("=" * 70)

    # 1. 编译检查
    t0 = time.perf_counter()
    bad = []
    tmp_pyc = tempfile.mkdtemp(prefix="gxm_pyc_")
    for p in glob.glob(os.path.join(PROJECT, "**", "*.py"), recursive=True):
        if ".venv" in p or "_reference_20260928" in p:
            continue
        try:
            py_compile.compile(p, cfile=os.path.join(tmp_pyc, os.path.basename(p) + "c"),
                               doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{os.path.relpath(p, PROJECT)}: {e.msg.strip().splitlines()[-1]}")
    record("py_compile 全部通过", not bad, time.perf_counter() - t0, "\n".join(bad))

    # 2. 语料冻结指纹
    t0 = time.perf_counter()
    n, chars, sha = docs_fingerprint()
    fp_ok = (n == DOCS_EXPECT_COUNT and chars == DOCS_EXPECT_CHARS and sha == DOCS_SHA256)
    detail = "" if fp_ok else (f"实测 {n} 篇 / {chars} 字 / {sha}\n"
                               f"期望 {DOCS_EXPECT_COUNT} 篇 / {DOCS_EXPECT_CHARS} 字 / {DOCS_SHA256}")
    record("语料冻结指纹未变", fp_ok, time.perf_counter() - t0, detail)

    # 3–13. 各校验脚本
    checks = [
        (["tools/eval_run.py", "--check"], "eval_run --check（题目字段/正则合法性）"),
        (["tools/test_judge.py"], "test_judge（判据回归）"),
        (["tools/check_kinds.py"], "check_kinds（kind/expect 与分母）"),
        (["tools/check_prompt.py"], "check_prompt（提示词能力对齐）"),
        (["tools/check_injection.py"], "check_injection（system 构造复现）"),
        (["tools/audit_coverage.py"], "audit_coverage（覆盖度审计）"),
        (["tools/audit_gold.py"], "audit_gold（金标独立支撑审计）"),
        (["tools/check_consistency.py"], "check_consistency（数字口径）"),
        (["tools/verify_fallback.py"], "verify_fallback（错误分类/降级话术）"),
        (["tools/verify_recent_messages.py"], "verify_recent_messages（历史截断单测）"),
        (["tools/verify_delete_session.py"], "verify_delete_session（删会话验证）"),
        (["tools/report_ops.py"], "report_ops（运行侧遥测汇总）"),
    ]
    for args, title in checks:
        path = os.path.join(PROJECT, args[0])
        if not os.path.exists(path):
            record(title, False, None, f"脚本不存在：{args[0]}")
            continue
        ok, dt, tail = run_script(args, title)
        record(title, ok, dt, tail)

    # ---------------- 汇总 ----------------
    print("=" * 70)
    passed = sum(1 for r in results if r[1])
    total = len(results)
    slow = sorted((r for r in results if r[2]), key=lambda r: -r[2])[:3]
    print(f"结果：{passed}/{total} 通过")
    if passed != total:
        print("未通过：")
        for title, ok, _, _ in results:
            if not ok:
                print(f"  ✗ {title}")
    print("耗时 top3：" + "，".join(f"{t[0].split('（')[0]} {t[2]:.2f}s" for t in slow))
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
