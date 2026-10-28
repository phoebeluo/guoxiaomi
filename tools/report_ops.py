"""运行侧遥测汇总（L0-1）：把 data/logs/requests.jsonl 变成可讲、可对比的数字。

背景
----
`app.py` 从 2026-10-13 起每轮往 `data/logs/requests.jsonl` 追加一条记录
（`classify_error()` 分类 + `log_request()` 落盘）。但**只有写入、没有读取**——
遥测是半个半成品：有数据，没结论。本脚本补上"读"的那一半。

它回答三个线上问题
------------------
1. **快不快** —— TTFT / 总时长的中位数与 P95（P95 才是用户体感，中位数会骗人）
2. **稳不稳** —— 错误五分类（quota/timeout/connection/content/other）的分布
3. **费不费** —— 输入/输出 token 的均值与总量（与 A1 离线成本口径对齐）

设计纪律
--------
- **零 API、零新依赖**（只用标准库；不引用 streamlit）。
- **旁路**：读不到文件 / 格式坏行，都只提示、不抛异常——它是观测工具，不是主流程。
- 时间用的是**服务端记录的时刻**，本脚本不做任何网络请求。

用法
----
    E:/guoxiaomi/.venv/Scripts/python.exe tools/report_ops.py
    E:/guoxiaomi/.venv/Scripts/python.exe tools/report_ops.py --json   # 同时输出机器可读结果
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
LOG_PATH = os.path.join(PROJECT, "data", "logs", "requests.jsonl")

# 与 app.py 的 ERROR_KINDS / FALLBACK_TEXT 对齐（这里只用于展示，不重复实现分类）
KIND_ORDER = ["quota", "timeout", "connection", "content", "other"]


def _percentile(sorted_vals, q):
    """线性插值分位数（numpy 不在依赖里，自己实现，避免为了 3 行数字引包）。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_records(path=LOG_PATH):
    """逐行读 JSONL。坏行只计数、不抛出——日志文件可能被编辑器截断。"""
    records, bad_lines = [], 0
    if not os.path.exists(path):
        return records, bad_lines
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return records, bad_lines


def _num(v):
    return v if isinstance(v, (int, float)) else None


def summarize(records):
    """把记录聚合成一个 dict（供命令行与 --json 两处复用）。"""
    ok = [r for r in records if r.get("ok")]
    fail = [r for r in records if not r.get("ok")]

    def stats(field, rows):
        vals = sorted(v for v in (_num(r.get(field)) for r in rows) if v is not None)
        if not vals:
            return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "sum": 0}
        return {
            "n": len(vals),
            "min": vals[0],
            "p50": _percentile(vals, 0.50),
            "p95": _percentile(vals, 0.95),
            "max": vals[-1],
            "sum": sum(vals),
        }

    kinds = Counter(r.get("error", {}).get("kind", "?") if r.get("error") else "—"
                    for r in fail)
    models = Counter(r.get("model", "?") for r in records)

    return {
        "total": len(records),
        "ok": len(ok),
        "fail": len(fail),
        "fail_rate": (len(fail) / len(records)) if records else None,
        "ttft_s": stats("ttft_s", ok),
        "total_s": stats("total_s", ok),
        "in_tokens": stats("in_tokens", ok),
        "out_tokens": stats("out_tokens", ok),
        "error_kinds": dict(kinds),
        "models": dict(models),
        "system_chars": sorted({r.get("system_chars") for r in records
                                if r.get("system_chars") is not None}),
    }


def _fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def print_report(s, bad_lines=0):
    line = "=" * 62
    print(line)
    print("运行侧遥测汇总（data/logs/requests.jsonl）")
    print(line)
    if s["total"] == 0:
        print("暂无可统计记录。先在 app.py 里真实问一句，日志才会产生。")
        return
    print(f"总轮次 : {s['total']}   成功 : {s['ok']}   失败 : {s['fail']}"
          + (f"（失败率 {s['fail_rate']:.1%}）" if s["fail_rate"] is not None else ""))
    if bad_lines:
        print(f"⚠️ 坏行（无法解析的 JSON）：{bad_lines} 行，已忽略")
    if s["system_chars"]:
        sc = s["system_chars"]
        shown = f"{sc[0]:,}" if len(sc) == 1 else "、".join(f"{x:,}" for x in sc)
        print(f"system 字符（去重）: {shown}")
    print(f"模型   : " + "、".join(f"{k}×{v}" for k, v in s["models"].items()))

    print("\n[延迟] 只统计成功轮次")
    for name, key in (("TTFT (s)", "ttft_s"), ("总时长 (s)", "total_s")):
        d = s[key]
        if d["n"] == 0:
            print(f"  {name:<12} 无数据")
            continue
        print(f"  {name:<12} n={d['n']}  中位 {_fmt(d['p50'])}  P95 {_fmt(d['p95'])}  "
              f"min {_fmt(d['min'])}  max {_fmt(d['max'])}")

    print("\n[Token] 只统计成功轮次")
    for name, key in (("输入", "in_tokens"), ("输出", "out_tokens")):
        d = s[key]
        if d["n"] == 0:
            print(f"  {name:<6} 无数据")
            continue
        print(f"  {name:<6} n={d['n']}  中位 {_fmt(d['p50'], 0)}  均值 "
              f"{_fmt(d['sum'] / d['n'], 0)}  合计 {d['sum']:,}")

    print("\n[错误分布]")
    if not s["error_kinds"]:
        print("  无失败记录")
    else:
        for kind in KIND_ORDER:
            if kind in s["error_kinds"]:
                print(f"  {kind:<11} {s['error_kinds'][kind]}")
        for kind, n in s["error_kinds"].items():
            if kind not in KIND_ORDER:
                print(f"  {kind:<11} {n}（未在五分类中，需检查 classify_error）")

    print("\n" + line)
    print("说明：P95 比中位数更接近用户体感；样本量小（n<20）时两者都不稳定，")
    print("      引用前请确认 n，并带上是哪一次运行的日志。")
    print(line)


def main():
    want_json = "--json" in sys.argv
    records, bad_lines = load_records()
    s = summarize(records)
    print_report(s, bad_lines)
    if want_json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
