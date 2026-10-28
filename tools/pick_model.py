# -*- coding: utf-8 -*-
"""模型池预检（跑分前用，零成本 —— 每个候选只发 1 次 max_tokens=1）。

为什么需要它
------------
百炼的**免费额度按「模型 ID」分桶**：某个模型 403 Free quota exhausted，
不代表别的模型也用完了。本项目已实测到 4 次（快照耗尽、别名还有额度）。

但「额度用完就切下一个」有两个陷阱，所以本工具刻意**不自动顺延**：

  ① 跑**中途**切换 —— 同一次评测里混了两个模型，报告直接作废
     （项目红线：跨模型指标不可比）。
     正解：**跑前**选好、整轮锁死。

  ② **跨档位**切换 —— 池子里 24 个模型不是一个档次（2026-10-26 全池重探后）：
         plus 系列   qwen3.7-plus / qwen-plus-latest（对外口径档）
         light      qwen3.7-flash-2026-07-15（带日期快照）/ qwen3.5-flash / qwen3.8-flash
                    / qwen-flash / qwen-turbo / qwen-flash-character-2026-02-26（角色向）
         flagship   qwen-max / qwen3.5-397b-a17b / qwen3.5-122b-a10b
         other      qwen3.6-27b / qwen-long / qwen3.5-35b-a3b
                    / qwen-vl-ocr / qwen-mt-lite / gui-plus（后三者**专用模型，勿用于评测**）
         third      glm-5.3 / glm-5.2 / glm-5.1 / glm-5 / kimi-k2.x
                    （**第三方血统，绝不与 qwen 数字互比**）
     从 plus 掉到 turbo，分数变化就**没法归因**了 ——
     你以为「只是换了个额度桶」，其实换的是模型能力。

所以默认行为：**只在「对外口径档（plus）」内预检**；
要跨档必须显式加 `--tier any`，工具会大声警告。

用法（CMD）
-----------
    E:\\guoxiaomi\\.venv\\Scripts\\python.exe tools\\pick_model.py
    E:\\guoxiaomi\\.venv\\Scripts\\python.exe tools\\pick_model.py --all
    E:\\guoxiaomi\\.venv\\Scripts\\python.exe tools\\pick_model.py --tier any
    E:\\guoxiaomi\\.venv\\Scripts\\python.exe tools\\pick_model.py --set-env
"""

import argparse
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# (档位, 模型名, 说明)。顺序 = 预检优先级：
#   同档内「带日期快照」优先（可复现），浮动别名兜底（额度通常更宽）。
MODEL_POOL = [
    # ── plus 档：对外口径档，唯一能与历史 plus 数字比较 ───────────────────
    ("plus", "qwen3.7-plus", "plus 系列浮动别名（2026-10-26 起为 .env 主口径）"),
    ("plus", "qwen-plus-latest", "plus 浮动别名（latest，会静默升级）"),

    # ── light 档：快、便宜，能力弱于 plus，与 plus 不可比 ─────────────────
    ("light", "qwen3.7-flash-2026-07-15", "**带日期快照**：light 档里唯一可复现的（10-26 新增）"),
    ("light", "qwen3.5-flash", "轻量：与 plus 不可比"),
    ("light", "qwen3.8-flash", "轻量：与 plus 不可比"),
    ("light", "qwen-flash", "轻量：与 plus 不可比"),
    ("light", "qwen-turbo", "轻量：10-26 探测为 RateLimitError（限流非耗尽，保留待用）"),
    ("light", "qwen-flash-character-2026-02-26", "角色扮演特化（system 会被角色化，勿用于评测）"),

    # ── flagship 档：强、贵，与 plus 不可比 ─────────────────────────────
    ("flagship", "qwen-max", "旗舰：与 plus 不可比"),
    ("flagship", "qwen3.5-397b-a17b", "超大 MoE：与 plus 不可比"),
    ("flagship", "qwen3.5-122b-a10b", "MoE 122B/10B 激活（10-26 新增）：与 plus 不可比"),

    # ── other 档：非通用对话模型一律标「勿用于评测」──────────────────────
    ("other", "qwen3.6-27b", "27B 中等规模（10-26 新增）：与 plus 不可比"),
    ("other", "qwen-long", "长文档：上下文极长，与 plus 不可比"),
    ("other", "qwen3.5-35b-a3b", "小 MoE：与 plus 不可比"),
    ("other", "qwen-vl-ocr-2025-11-20", "多模态 OCR（纯文本对话弱，**勿用于评测**）"),
    ("other", "qwen-mt-lite", "机器翻译专用（**勿用于评测**）"),
    ("other", "gui-plus", "GUI 智能体专用（**勿用于评测**）"),

    # ── third 档：第三方血统，任何情况下不与 qwen 数字互比 ────────────────
    ("third", "glm-5.3", "智谱 GLM（10-26 新增）"),
    ("third", "glm-5.2", "智谱 GLM"),
    ("third", "glm-5.1", "智谱 GLM（10-26 新增）"),
    ("third", "glm-5", "智谱 GLM（10-26 新增）"),
    ("third", "kimi-k2.6", "月之暗面 Kimi"),
    ("third", "kimi-k2.5", "月之暗面 Kimi"),
    ("third", "kimi-k2.7-code", "月之暗面 Kimi 代码特化"),
]

# 能用 ≠ 该用：这几个是专用模型，探测能通过，但它们的输出不能当"通用客服回答"来评。
# 选到就大声警告 —— 否则某天 plus 档全空时，工具会「顺手」选上 OCR 模型跑评测。
NOT_FOR_EVAL = {
    "qwen-vl-ocr-2025-11-20",
    "qwen-mt-lite",
    "gui-plus",
    "qwen-flash-character-2026-02-26",
}

TIER_ORDER = ["plus", "light", "flagship", "other", "third"]

# 已确认失效（2026-10-12 `--discover` 实测 39 个，其中 22 个不可用）：**默认不再探测**。
# 额度会重置、平台也会新增模型 —— 想复查时把名字加回 MODEL_POOL 即可。
KNOWN_DEAD = [
    # 403 免费额度耗尽
    "qwen3.5-plus-2026-02-15", "qwen3.7-plus-2026-05-26", "qwen-plus",
    # 2026-10-26 全池重探：这三条实测 403 额度耗尽，已从 MODEL_POOL 移出
    "qwen3.5-plus", "qwen3.6-plus", "qwen3.6-flash",
    # 404 不存在 / 已下线
    "qwen-flash-latest", "qwen3-max", "qwen3.6-max", "qwen3-235b-a22b",
    "qwen3-32b", "qwen3-30b-a3b", "qwen3-next-80b-a3b-instruct",
    "qwq-plus", "qwq-32b", "qwen3-coder-plus",
    "qwen-vl-max", "qwen-vl-plus", "deepseek-v3", "deepseek-r1",
    # 403 无权限（当前账号 / 地域未开通）
    "qwen-turbo-latest", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct", "qwen2.5-7b-instruct",
]

# --discover 要扫的名单 = 池内全部（失效项已剔除，见上）。
# 目的：回答「这张账号到底还有多少个**独立额度桶**可用」。
DISCOVER_CANDIDATES = [m for _, m, _ in MODEL_POOL]


def classify(exc):
    """把异常翻译成人能看懂的桶位状态。"""
    msg = str(exc).lower()
    if "quota" in msg or "exhausted" in msg:
        return "403 额度耗尽"
    if "not exist" in msg or "model not" in msg:
        return "404 不存在/已下线"
    if "permission" in msg or "denied" in msg or "forbidden" in msg:
        return "403 无权限"
    if "timeout" in msg or "connection" in msg or "unreach" in msg:
        return "网络不可达"
    return type(exc).__name__


def probe(client, model):
    """探测单个模型是否可用。返回 (ok, 状态描述)。"""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, "可用"
    except Exception as exc:  # noqa: BLE001 —— 探测就是要吞掉一切异常
        return False, classify(exc)


def set_env_model(model):
    """把 .env 里的 LLM_MODEL 改成指定值（保留其余行与注释）。"""
    env_path = os.path.join(PROJECT, ".env")
    with open(env_path, encoding="utf-8") as f:
        lines = f.readlines()
    hit = False
    for i, line in enumerate(lines):
        if line.strip().startswith("LLM_MODEL="):
            lines[i] = f"LLM_MODEL={model}\n"
            hit = True
    if not hit:
        lines.append(f"LLM_MODEL={model}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return env_path


def main():
    ap = argparse.ArgumentParser(description="模型池预检：跑分前选定一个可用模型")
    ap.add_argument("--all", action="store_true",
                    help="列出池内全部候选的可用性（默认只报结论）")
    ap.add_argument("--tier", default="plus",
                    choices=["plus", "light", "flagship", "other", "third", "any"],
                    help="限制在哪个档位内预检（默认 plus = 对外口径档）")
    ap.add_argument("--set-env", action="store_true",
                    help="把选中的模型写回 .env 的 LLM_MODEL")
    ap.add_argument("--discover", action="store_true",
                    help="扫宽候选名单，列出全部可用的额度桶（不做选型）")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(PROJECT, ".env"))
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未找到 DASHSCOPE_API_KEY，请检查 .env")
        return 2

    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("DASHSCOPE_BASE_URL"))

    if args.discover:
        names = list(dict.fromkeys(DISCOVER_CANDIDATES))  # 去重且保序
        avail, dead = [], []
        print(f"扫描 {len(names)} 个候选模型（每个 1 token）…\n")
        for m in names:
            ok, status = probe(client, m)
            (avail if ok else dead).append(m)
            print(f"  {'OK  ' if ok else 'NG  '} {m:<30} {status}")
        print()
        print(f"✅ 可用的独立额度桶：{len(avail)} 个")
        for m in avail:
            print(f"     · {m}")
        print(f"❌ 不可用：{len(dead)} 个（额度耗尽 / 已下线 / 无权限）")
        print()
        print("提醒：桶多 ≠ 可以混用。**一次评测必须锁一个模型**；")
        print("      桶多的意义是「这个桶空了就换下一个桶继续跑」，不是「一锅里混着跑」。")
        return 0

    allowed = TIER_ORDER if args.tier == "any" else [args.tier]
    candidates = [c for c in MODEL_POOL if c[0] in allowed]

    current = os.environ.get("LLM_MODEL", "(未设置，将回退兜底值)")
    print(f"当前 .env 口径 : {current}")
    print(f"预检档位       : {args.tier}"
          + ("（⚠️ 允许跨档：混档会使分数不可归因）" if args.tier == "any" else ""))
    print()

    results = []
    for tier, model, note in candidates:
        ok, status = probe(client, model)
        results.append((tier, model, note, ok, status))
        if args.all:
            mark = "OK  " if ok else "NG  "
            print(f"  {mark} [{tier:<8}] {model:<26} {status}")

    picked = next((r for r in results if r[3]), None)

    print()
    if args.all:
        print("—" * 62)

    if picked:
        tier, model, note, _, _ = picked
        print(f"✅ 选定：{model}   （{tier} 档 · {note}）")
        if model in NOT_FOR_EVAL:
            print("   ⚠️⚠️ 专用模型（OCR / 翻译 / GUI / 角色向）——**不要拿它跑评测**：")
            print("        它测出来的是另一件事，不是果小蜜的客服能力。")
        if args.set_env:
            path = set_env_model(model)
            print(f"   已写回 {path} 的 LLM_MODEL")
        else:
            print(f"   要写入 .env，加参数 --set-env（或手动改 LLM_MODEL={model}）")
    else:
        print(f"❌ 「{args.tier}」档内**无一可用** —— 额度可能整体耗尽。")
        if args.tier != "any":
            print("   可查看全池：加参数 --all --tier any")
            print("   ⚠️ 跨档运行前先想清楚：换档后得到的数字**不能**与 plus 档历史数字比较。")
        return 1

    # 同档内还有别的可用 → 提示，但不自动切（避免"跑中途换模型"）
    same_tier_ok = [r for r in results if r[3] and r[1] != picked[1] and r[0] == picked[0]]
    if same_tier_ok:
        names = "、".join(r[1] for r in same_tier_ok)
        print(f"   同档备选（该桶耗尽时再手动换）：{names}")

    print()
    print("提醒：换模型 → 历史指标不可比。报告里必须写清模型名 + 日期。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
