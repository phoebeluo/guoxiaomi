"""A1 成本与延迟量化：把「全量注入多贵多慢」从估数变成实测数。

为什么需要这个脚本
------------------
1. `eval/reports/*.json` 只存了 answer，**没有 usage** —— 跑过的调用没留下 token 记录。
2. `eval_run.py` 的 `_ask()` 用的是**非流式**调用，物理上拿不到 TTFT。
于是「单次约 21K token / system 占 90%」一直只是**从字符数推算的估数**，
而「首字延迟」从来没有人量过。

本脚本用【流式 + `stream_options.include_usage`】补齐这两项，并把 system 的
构成拆开（提示词本体 vs 语料），因为「语料占 system 的百分之多少」才是
「该不该上检索」这个问题的真正依据。

用法
----
    # 零 API：拆 system 构成 + 按字符估算整轮 token
    python tools/measure_cost.py

    # 额外实测（流式，拿真实 token / TTFT）
    python tools/measure_cost.py --probe

额度提示：免费额度按 token 计。--probe 默认 1 题 × 3 次 + 1 道多轮题 × 2 次
≈ 5 次调用 ≈ 10 万 token 量级，请确认额度后再跑。
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)        # 为了 import eval_run（同目录）
sys.path.insert(0, PROJECT)     # 为了 rag.load_docs

import eval_run

# 与 _sys_inject() 里的拼接串必须逐字一致，否则拆解出来的数字是错的
SEP = "\n\n## 知识库全文\n\n"

# 免费额度是按 token 计的 —— 估算是为了「一轮评测吃多少额度」，
# 不是为了钱。所以这里只给 token，不碰价格。
CHARS_PER_TOKEN_HINT = 1.5      # 中文为主文本的经验值，只用于零 API 估算


# ======================================================================
# 第一段：零 API —— 拆 system 构成
# ======================================================================

def split_system():
    """把 system_content 拆成 提示词本体 / 分隔符 / 语料 三段。"""
    prompt = eval_run.SYSTEM_PROMPT
    docs_info = eval_run._docs_content()
    docs = docs_info["content"]
    total = len(prompt) + len(SEP) + len(docs)
    return {
        "prompt_chars": len(prompt),
        "sep_chars": len(SEP),
        "docs_chars": len(docs),
        "docs_count": docs_info["count"],
        "total_chars": total,
    }


def report_system(s):
    total = s["total_chars"]
    print("=" * 66)
    print("A1-1  system 构成拆解（零 API）")
    print("=" * 66)
    rows = [
        ("提示词本体 system_prompt", s["prompt_chars"]),
        ("分隔串 '## 知识库全文'", s["sep_chars"]),
        (f"语料正文（{s['docs_count']} 篇）", s["docs_chars"]),
    ]
    for name, n in rows:
        print(f"  {name:<28} {n:>8,} 字符   {n / total:>6.1%}")
    print(f"  {'合计 = system_content':<28} {total:>8,} 字符   100.0%")
    print()
    # 这个占比就是整件事的锚点：它回答「检索能省掉多少」
    print(f"  ★ 语料占 system 的 {s['docs_chars'] / total:.1%}"
          f"  →  检索能压缩的上限就在这里")
    print(f"  平均每篇语料 {s['docs_chars'] / s['docs_count']:.0f} 字符")
    return s


# ======================================================================
# 第二段：零 API —— 按 chars.jsonl 的 msgs 结构，估算整轮消耗
# ======================================================================

def estimate_round(cases, system_chars, avg_answer_chars):
    """逐题复现 run() 的 messages 结构，累计字符数。

    为什么要复现：多轮题会把「历史轮的用户问题 + 模型回答」也塞进下一次请求，
    所以「一轮消耗」不是 22 × system，而是要把回放的部分算进去。
    这里 assistant 的历史回答长度用实测均值代替（零 API 只能这样估）。
    """
    calls = []          # 每次调用的 (题号, 角色, 字符数)
    for case in cases:
        cid = case["id"]
        history = case.get("history", [])

        # 回放历史：每条 history 都要真调一次模型（见 eval_run.py:395-397）
        for i, t in enumerate(history):
            used = system_chars + sum(len(h) for h in history[:i + 1]) \
                   + avg_answer_chars * i + len(t)
            calls.append((cid, f"回放{i + 1}", used))

        # 最终一轮：system + 历史(问+答) + 本题问题
        final = system_chars + sum(len(t) for t in history) \
                + avg_answer_chars * len(history) + len(case["q"])
        calls.append((cid, "作答", final))

    total_chars = sum(c for _, _, c in calls)
    return calls, total_chars


# ======================================================================
# 第三段：动 API —— 流式实测 token / TTFT / 耗时
# ======================================================================

def stream_once(client, model, messages):
    """流式调用一次，返回 dict(ttft, total, usage, text)。

    TTFT（Time To First Token）只能在流式下测：非流式要等整个回答结束
    才返回，第一个字什么时候到根本无从得知。
    """
    t0 = time.perf_counter()
    ttft = None
    usage = None
    parts = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,                       # 与 eval_run._ask 保持一致
        max_tokens=1024,
        stream=True,
        # include_usage：最后一个数据包只带用量信息、choices 为空。
        # app.py:303 的注释早就提到过这个机制，但项目从没用它量过成本。
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": False},   # 同样与 _ask 对齐
    )

    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.choices[0].delta.content)

    return {
        "ttft": ttft,
        "total": time.perf_counter() - t0,
        "usage": usage,
        "text": "".join(parts),
    }


def probe(cases, system_content, first_id, multi_id, repeats, model=None):
    """实测：对单轮题跑 repeats 次，对多轮题完整回放一次。

    model 参数的意义：百炼的免费额度**按模型 ID 分桶**（本项目已实测 3 次），
    基线模型额度耗尽时，必须能换一个桶来跑 —— 所以这里允许命令行覆盖，
    **不动 .env**（避免污染其它脚本的运行口径）。
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(PROJECT, ".env"))
    if not model:
        model = os.environ.get("LLM_MODEL")
    if not model:
        model = "qwen3.5-plus-2026-02-15"
        print(f"⚠️ 未设置 LLM_MODEL，回退到 {model}")
    else:
        env_model = os.environ.get("LLM_MODEL")
        if env_model and model != env_model:
            print(f"⚠️ 本次用 --model 覆盖了 .env 的 {env_model} → {model}")
            print("   报告必须标注该模型名（跨模型指标不可比）。")
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    by_id = {c["id"]: c for c in cases}
    records = []

    print("=" * 66)
    print(f"A1-2  流式实测（模型 {model}）")
    print("=" * 66)

    # --- 单轮题：重复多次，取 TTFT 中位数 ---
    case = by_id[first_id]
    print(f"\n单轮题 #{case['id']}：{case['q']}（跑 {repeats} 次）")
    single_runs = []
    for i in range(repeats):
        msgs = [{"role": "system", "content": system_content},
                {"role": "user", "content": case["q"]}]
        r = stream_once(client, model, msgs)
        single_runs.append(r)
        u = r["usage"]
        print(f"  #{i + 1}  prompt={u.prompt_tokens if u else '?':>6}"
              f"  completion={u.completion_tokens if u else '?':>4}"
              f"  TTFT={r['ttft']:.2f}s  总={r['total']:.2f}s")
        records.append({"case": case["id"], "kind": "single",
                        "prompt_tokens": u.prompt_tokens if u else None,
                        "completion_tokens": u.completion_tokens if u else None,
                        "ttft": r["ttft"], "total": r["total"]})

    # --- 多轮题：完整回放，看「历史轮把输入撑大多少」---
    mcase = by_id[multi_id]
    print(f"\n多轮题 #{mcase['id']}：{mcase['q']}（回放 {len(mcase.get('history', []))} 轮）")
    msgs = [{"role": "system", "content": system_content}]
    for t in mcase.get("history", []):
        msgs.append({"role": "user", "content": t})
        r = stream_once(client, model, msgs)
        msgs.append({"role": "assistant", "content": r["text"]})
        u = r["usage"]
        print(f"  回放 prompt={u.prompt_tokens if u else '?':>6}"
              f"  completion={u.completion_tokens if u else '?':>4}"
              f"  TTFT={r['ttft']:.2f}s  总={r['total']:.2f}s")
        records.append({"case": mcase["id"], "kind": "replay",
                        "prompt_tokens": u.prompt_tokens if u else None,
                        "completion_tokens": u.completion_tokens if u else None,
                        "ttft": r["ttft"], "total": r["total"]})
    msgs.append({"role": "user", "content": mcase["q"]})
    r = stream_once(client, model, msgs)
    u = r["usage"]
    print(f"  作答 prompt={u.prompt_tokens if u else '?':>6}"
          f"  completion={u.completion_tokens if u else '?':>4}"
          f"  TTFT={r['ttft']:.2f}s  总={r['total']:.2f}s")
    records.append({"case": mcase["id"], "kind": "multi-final",
                    "prompt_tokens": u.prompt_tokens if u else None,
                    "completion_tokens": u.completion_tokens if u else None,
                    "ttft": r["ttft"], "total": r["total"]})

    # --- 汇总 ---
    pts = [x["prompt_tokens"] for x in records if x["prompt_tokens"]]
    ttf = [x["ttft"] for x in records if x["ttft"]]
    print("\n" + "-" * 66)
    if pts:
        print(f"  输入 token：min {min(pts):,} / max {max(pts):,} / 均值 {int(statistics.mean(pts)):,}")
    if ttf:
        print(f"  TTFT：中位数 {statistics.median(ttf):.2f}s"
              f"（{min(ttf):.2f} ~ {max(ttf):.2f}）")
    single_ttft = [x["ttft"] for x in records if x["kind"] == "single" and x["ttft"]]
    if len(single_ttft) >= 3:
        print(f"  单轮 TTFT 中位数：{statistics.median(single_ttft):.2f}s"
              f"（同一题 {len(single_ttft)} 次，说明单次值是抽样值）")
    return records


# ======================================================================
# main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="额外做流式实测（消耗 API 额度）")
    ap.add_argument("--single", type=int, default=1,
                    help="单轮实测用哪道题（默认 #1）")
    ap.add_argument("--multi", type=int, default=16,
                    help="多轮实测用哪道题（默认 #16）")
    ap.add_argument("--repeats", type=int, default=3,
                    help="单轮题重复次数，用于取 TTFT 中位数（默认 3）")
    ap.add_argument("--model", default=None,
                    help="覆盖 .env 的 LLM_MODEL（额度分桶耗尽时换桶用，不改 .env）")
    args = ap.parse_args()

    cases = eval_run.load_cases()
    s = report_system(split_system())

    # 零 API 的整轮估算：assistant 历史长度先用经验值 200 字符
    calls, total_chars = estimate_round(cases, s["total_chars"], 200)
    print("\n" + "=" * 66)
    print("A1-3  整轮消耗估算（零 API，复现 run() 的 messages 结构）")
    print("=" * 66)
    print(f"  一轮调用次数：{len(calls)} 次"
          f"（{len(cases)} 题；多轮题每轮历史各算 1 次）")
    print(f"  累计输入字符：{total_chars:,}")
    # 字符→token 只是量级估算，实测值出来后请替换
    print(f"  按 {CHARS_PER_TOKEN_HINT} 字符/token 粗估："
          f"≈ {int(total_chars / CHARS_PER_TOKEN_HINT):,} token")
    print("  ⚠️ 字符→token 是估算；真实值见 --probe 的 prompt_tokens")

    if args.probe:
        since = time.time()
        try:
            probe(cases, eval_run._sys_inject(),
                  args.single, args.multi, args.repeats, args.model)
        except Exception as e:
            # 额度耗尽是**预期内**的失败（免费额度按模型分桶，随时可能烧完），
            # 所以不给 traceback，直接给换桶指令 —— fail fast 且可操作。
            if "quota" in str(e).lower():
                used = args.model or os.environ.get("LLM_MODEL")
                print(f"\n❌ 额度耗尽：{used}")
                print("   → 换一个桶重跑，例如：")
                print("     python tools/measure_cost.py --probe --model qwen3.5-plus")
                return
            raise
        print(f"\n（实测耗时 {time.time() - since:.1f}s）")
    else:
        print("\n提示：加 --probe 可拿到真实 token 与 TTFT（消耗 API 额度）。")


if __name__ == "__main__":
    main()
