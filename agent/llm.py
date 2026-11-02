"""模型选择与降级链（2026-11-02 · W11 收尾）。

## 为什么需要这个文件

11-02 跑工具评测时，`LLM_MODEL=qwen3.7-plus` 突然全线 403
（`Free quota exhausted`），整个评测直接停摆——明明账号里还有
qwen3.8-max / qwen3.6-flash / qwen3-coder-flash 等模型可用。

根因不是模型不通，是**代码把「哪个模型」写死成单值**：
一个模型不可用 = 整个系统不可用。

## 降级链的两个纪律

1. **降级必须留痕。** 报告和遥测里记的是「实际回答的模型」，
   不是「配置里写的模型」。否则「以为在跑 A、其实在跑 B」，
   所有跨模型不可比的指标会悄悄混在一起。
2. **探活结果进程内缓存。** 探一次约 20 token，每个进程只探一次；
   不缓存会把探活成本乘以调用次数。
"""

import os
import threading

# 有日期后缀的版本走付费配额；无后缀的别名走免费额度
# （2026-11-02 实测：qwen3.6-flash 403，qwen3.6-flash-2026-04-16 正常）
DEFAULT_FALLBACKS = [
    "qwen3.8-max-0902",
    "qwen3.6-flash-2026-04-16",
    "qwen3-coder-flash",
    "qwen3.5-27b",
    "qwen3-vl-plus",
    "qwen3.7-max-2026-05-17",
    "qwen-plus-2025-01-25",
]

_lock = threading.Lock()
_resolved: dict = {}
_probed: dict = {}


def candidates(prefer=None):
    """优先项 + .env 里的 LLM_FALLBACKS + 内置默认，去重保序。"""
    out = []
    for m in [prefer, os.environ.get("LLM_MODEL")]:
        if m and m not in out:
            out.append(m)
    for m in (os.environ.get("LLM_FALLBACKS") or "").split(","):
        m = m.strip()
        if m and m not in out:
            out.append(m)
    for m in DEFAULT_FALLBACKS:
        if m not in out:
            out.append(m)
    return out


def _alive(client, model):
    """1 token 探活。403/429/404 视为不可用，其余异常不当作模型问题。"""
    if model in _probed:
        return _probed[model]
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "1"}], max_tokens=1)
        ok = True
    except Exception as e:
        s = str(e)
        ok = not any(k in s for k in ("Free quota", "quota", "429", "404",
                                      "does not exist", "Model not exist"))
    _probed[model] = ok
    return ok


def resolve(client, prefer=None, verbose=True):
    """返回 (model, dropped)。dropped 是被跳过的模型及原因，供告警用。"""
    key = prefer or os.environ.get("LLM_MODEL") or ""
    with _lock:
        if key in _resolved:
            return _resolved[key]
        dropped = []
        for m in candidates(prefer):
            if _alive(client, m):
                if dropped and verbose:
                    print(f"⚠️ 模型降级：跳过 {dropped}，改用 {m}")
                _resolved[key] = (m, dropped)
                return m, dropped
            dropped.append(m)
        raise RuntimeError(f"所有候选模型都不可用：{dropped}")
