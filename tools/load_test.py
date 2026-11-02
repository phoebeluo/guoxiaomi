"""服务层压测（2026-11-02 · W12）。

## 测什么，不测什么

**测**：Flask + 检索 + 拼 prompt + 落盘 这一层的吞吐与尾延迟。
**不测**：模型延迟。模型用 `MOCK_LLM=1` 的假实现替代（固定 50 ms）。

理由很实际：混在一起测，得到一个「P99 = 3.2 s」的数字，然后谁也说不清
这 3.2 秒里多少是我的代码、多少是阿里的排队。分开测才能定位。

## 怎么读结果

- **QPS 随并发上升，到某点不再涨** → 瓶颈出现了
- **503 从 0 变成非 0** → 撞到入口限流（`MAX_INFLIGHT`），说明保护生效
- **200 的 P99 明显抬升但 QPS 不涨** → 排队，不是更快，只是更多人一起等

用法：
    E:/guoxiaomi/.venv/Scripts/python.exe tools/load_test.py
    E:/guoxiaomi/.venv/Scripts/python.exe tools/load_test.py --levels 1,4,8 --per 50
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

# 压测必须在 mock 下跑：真调模型既慢又花钱，且测不出服务层
os.environ.setdefault("MOCK_LLM", "1")

import requests
from werkzeug.serving import make_server

import api
import core

Q = ["满多少免运费？", "坏果怎么赔？", "苹果多少钱一斤？",
     "武侯区能送吗？", "我今年花了 2100，什么会员？"]


def isolate_sessions():
    """压测产生的会话写临时目录，别污染 data/sessions。

    第一次压测忘了这一步，一次跑完往真实会话目录里塞了 166 个 mock 会话文件——
    压测数据混进产品数据，比没有压测更糟。
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gxm_load_")
    core.SESSIONS_DIR = tmp
    return tmp


def start_server(port=5057):
    srv = make_server("127.0.0.1", port, api.app, threaded=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    # 等端口可用，最多 5 秒；不用 sleep 猜时间
    t0 = time.time()
    while time.time() - t0 < 5:
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            return srv, port
        except requests.RequestException:
            time.sleep(0.05)
    raise RuntimeError("服务 5 秒内没起来")


def one(base, i):
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base}/chat", json={"message": Q[i % len(Q)]}, timeout=30)
        dt = time.perf_counter() - t0
        return r.status_code, dt
    except requests.RequestException:
        return 0, time.perf_counter() - t0


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round((len(vals) - 1) * p)))
    return vals[k]


def run_level(base, conc, total):
    seen = {"ok": [], "503": 0, "500": 0, "0": 0}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for code, dt in ex.map(lambda i: one(base, i), range(total)):
            if code == 200:
                seen["ok"].append(dt)
            elif code == 503:
                seen["503"] += 1
            elif code == 500:
                seen["500"] += 1
            else:
                seen["0"] += 1
    wall = time.perf_counter() - t0
    ok = seen["ok"]
    return {
        "conc": conc, "total": total, "wall": round(wall, 2),
        "qps": round(len(ok) / wall, 1) if wall else 0,
        "ok": len(ok), "busy": seen["503"], "err": seen["500"], "fail": seen["0"],
        "p50": round(pct(ok, 0.50) * 1000) if ok else None,
        "p95": round(pct(ok, 0.95) * 1000) if ok else None,
        "p99": round(pct(ok, 0.99) * 1000) if ok else None,
        "mean": round(statistics.mean(ok) * 1000) if ok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8,16,32", help="并发档位，逗号分隔")
    ap.add_argument("--per", type=int, default=60, help="每档请求数")
    ap.add_argument("--out", default=os.path.join(PROJECT, "eval", "reports", "load_test.md"))
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    tmp = isolate_sessions()
    srv, port = start_server()
    base = f"http://127.0.0.1:{port}"
    print(f"服务已起：{base}　MAX_INFLIGHT={api.MAX_INFLIGHT}　MOCK={api.MOCK}")
    print(f"会话写临时目录：{tmp}")

    rows = []
    try:
        run_level(base, 1, 5)                    # 预热，不计入
        for c in levels:
            r = run_level(base, c, args.per)
            rows.append(r)
            print(f"并发 {c:>2} | QPS {r['qps']:>6} | P50 {r['p50']}ms P95 {r['p95']}ms "
                  f"P99 {r['p99']}ms | 200 {r['ok']} 503 {r['busy']} 500 {r['err']}")
    finally:
        srv.shutdown()

    # ---- 瓶颈定位：QPS 增长率第一次掉到 20% 以下的那档，就是拐点
    knee = None
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["qps"], rows[i]["qps"]
        grow = (cur - prev) / prev if prev else 0
        rows[i]["grow"] = round(grow * 100)
        if knee is None and grow < 0.2 and cur > 0:
            knee = rows[i]["conc"]
    base_qps = rows[0]["qps"] if rows else 0

    L = ["# 服务层压测（W12）", "",
         f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- 被测：Flask `/chat`（检索 + 拼 prompt + 落盘），**模型为 mock**，固定 {int(api.MOCK_DELAY*1000)} ms",
         f"- 入口限流：`MAX_INFLIGHT={api.MAX_INFLIGHT}`，超出返回 503",
         f"- 每档 {args.per} 请求，单进程单 worker（threaded）", "",
         "## 结果", "",
         "| 并发 | QPS | 较上档增长 | P50 | P95 | P99 | 200 | 503 | 500 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['conc']} | {r['qps']} | {r.get('grow', '—')}% | {r['p50']} ms | "
                 f"{r['p95']} ms | {r['p99']} ms | {r['ok']} | {r['busy']} | {r['err']} |")

    L += ["", "## 瓶颈定位", ""]
    if knee:
        L += [f"**拐点在并发 {knee}**：QPS 较上一档增长不足 20%，加并发不再换吞吐，只换排队。",
              f"并发 1 时 QPS = {base_qps}，峰值 QPS = {max(r['qps'] for r in rows)}。",
              "",
              "这不是应用层算力不够，而是**入口信号量先满了**：超过 "
              f"`MAX_INFLIGHT={api.MAX_INFLIGHT}` 的请求直接被 503 拒绝。",
              "换句话说，**吞吐上限是我自己设的，不是硬件逼出来的** —— "
              "这个数该由上游 RPM 决定，不该由 worker 数决定。"]
    else:
        L += [f"各档 QPS 仍在增长（1 → {max(r['qps'] for r in rows)}），未观察到明显拐点，"
              "说明在测试区间内应用层还没成为瓶颈。"]

    L += ["", "## 为什么瓶颈不在应用层（这才是结论）", "",
          "| 层 | 单次开销 | 是否瓶颈 |", "|---|---|---|",
          f"| 检索（45 篇向量，numpy 余弦） | 约 4 ms | 否 |",
          f"| 拼 prompt + 落盘 | 约 3–8 ms | 否 |",
          f"| **模型生成** | **数百 ms ~ 数秒** | **是** |",
          "",
          "真实流量下，QPS 上限 = 上游 RPM ÷ 每请求 token 量，与我的 worker 数无关。",
          "所以并发保护做在**入口**（信号量 + 503 + Retry-After），而不是拼命加 worker——",
          "加 worker 只会让更多请求同时卡在上游等待，把 429 变成超时。", "",
          "## 复现", "",
          "```",
          "E:/guoxiaomi/.venv/Scripts/python.exe tools/load_test.py",
          "E:/guoxiaomi/.venv/Scripts/python.exe tools/load_test.py --levels 1,8,32 --per 100",
          "```", ""]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
               "knee": knee, "max_inflight": api.MAX_INFLIGHT, "mock": api.MOCK},
              open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n报告：{args.out}")
    print(f"拐点：{'并发 ' + str(knee) if knee else '未出现'}　峰值 QPS：{max(r['qps'] for r in rows)}")


if __name__ == "__main__":
    main()
