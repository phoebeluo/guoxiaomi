"""把 run_all 的 12 项检查搬进 pytest，并补上编译与语料指纹。

为什么用 subprocess 而不是重写断言
--------------------------------
那 12 个脚本各自内嵌了业务判据（金标是否独立支撑、数字口径是否一致……），
重写一遍等于把判据复制两份，将来改一处忘一处就出现「两个测试说不同的话」。
这里只做**包装**：脚本返回 0 即通过。真正的单元断言在 test_retrieval.py。
"""

import glob
import os
import subprocess
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CHECKS = [
    (["tools/eval_run.py", "--check"], "eval_run_check"),
    (["tools/test_judge.py"], "judge_regression"),
    (["tools/check_kinds.py"], "kinds"),
    (["tools/check_prompt.py"], "prompt_alignment"),
    (["tools/check_injection.py"], "injection"),
    (["tools/audit_coverage.py"], "coverage"),
    (["tools/audit_gold.py"], "gold_audit"),
    (["tools/check_consistency.py"], "consistency"),
    (["tools/verify_fallback.py"], "fallback"),
    (["tools/verify_recent_messages.py"], "recent_messages"),
    (["tools/verify_delete_session.py"], "delete_session"),
    (["tools/report_ops.py"], "report_ops"),
]


@pytest.mark.parametrize("args,name", CHECKS, ids=[n for _, n in CHECKS])
def test_check_script(args, name):
    """每个检查脚本都必须以返回码 0 结束。失败时把它的输出带出来，便于定位。"""
    proc = subprocess.run([PY] + args, cwd=PROJECT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"{name} 失败\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_all_py_compile():
    """全项目编译检查：语法错在 CI 第一步就该拦住，而不是等到跑评测。"""
    bad = []
    for p in glob.glob(os.path.join(PROJECT, "**", "*.py"), recursive=True):
        if ".venv" in p or "__pycache__" in p:
            continue
        proc = subprocess.run([PY, "-m", "py_compile", p], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            bad.append(f"{os.path.relpath(p, PROJECT)}: {proc.stderr.strip().splitlines()[-1:]}")
    assert not bad, "\n".join(bad)


def test_corpus_frozen():
    """语料冻结：45 篇的 SHA-256 必须与登记值一致。

    这是整套指标的地基 —— 语料变了，历史上所有数字都不可比。
    """
    import hashlib

    sys.path.insert(0, os.path.join(PROJECT, "tools"))
    import run_all as ra

    n, chars, sha = ra.docs_fingerprint()
    assert n == ra.DOCS_EXPECT_COUNT, f"篇数变了：{n}"
    assert chars == ra.DOCS_EXPECT_CHARS, f"字数变了：{chars}"
    assert sha == ra.DOCS_SHA256, f"指纹变了：{sha}"
