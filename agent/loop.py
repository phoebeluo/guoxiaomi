"""W9 · function calling 对话循环（流式）。

设计要点
========
1. **首轮就开流式**，不先做一次非流式探路。
   原因：绝大多数提问不需要工具（检索已经能答），若为少数情况牺牲全部提问的流式体验，
   等于用 90% 的体验换 10% 的功能。所以第一轮必须仍然逐字输出。

2. **工具调用靠流式增量拼装**。OpenAI 兼容协议里 `tool_calls` 是分片下发的：
   第 1 片给 `name`，后面若干片给 `arguments` 的字符串片段。
   拼装时按 `index` 归组，不能按到达顺序直接拼——多工具并发时顺序会交错。

3. **最后一轮不带 tools**。否则模型可能一直调工具、不收敛（尤其它调完发现参数不对想再试）。
   收敛条件是「回合数用完」而不是「模型自己停」，这是可控性上的必要之恶。

4. **工具结果原样回灌**，不做自然语言改写。改写会让「工具说了什么」和「用户看到什么」
   之间多一层无测试覆盖的中转；W11 要统计参数正确率，必须拿到原始字段。
"""

from __future__ import annotations

import json

from agent.tools import TOOL_SCHEMAS, safe_dispatch

MAX_ROUNDS = 2          # 1 轮工具 + 1 轮作答。再复杂的需求属于 W11 之后


def _accumulate(buf, tc):
    """把流式 tool_call 分片合并进 buf[index]。"""
    idx = tc.index if tc.index is not None else 0
    slot = buf.setdefault(idx, {"id": None, "name": "", "arguments": ""})
    if getattr(tc, "id", None):
        slot["id"] = tc.id
    fn = getattr(tc, "function", None)
    if fn is not None:
        if getattr(fn, "name", None):
            slot["name"] += fn.name
        if getattr(fn, "arguments", None):
            slot["arguments"] += fn.arguments


def stream_chat(client, model, messages, tools=None, max_rounds=MAX_ROUNDS,
                extra_body=None):
    """流式对话 + 工具调用。

    yield 事件（元组）：
      ("delta", str)      正文增量，调用方拼接后逐字显示
      ("tools", list)     本轮要执行的工具调用（name/args/ok/error/result）
      ("usage", obj)      用量（多轮会累加，调用方只需在结束时读最后一个）
    """
    msgs = list(messages)
    total_in = total_out = 0
    rounds = max(1, max_rounds)

    for rnd in range(rounds):
        last = rnd == rounds - 1
        kwargs = dict(
            model=model,
            messages=msgs,
            stream=True,
            stream_options={"include_usage": True},
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        # 最后一轮撤掉 tools，强制收敛成自然语言回答
        if tools and not last:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = client.chat.completions.create(**kwargs)

        content = ""
        calls = {}
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                total_in += getattr(usage, "prompt_tokens", 0) or 0
                total_out += getattr(usage, "completion_tokens", 0) or 0
                yield ("usage", {"prompt_tokens": total_in, "completion_tokens": total_out})

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content += delta.content
                yield ("delta", delta.content)
            for tc in (getattr(delta, "tool_calls", None) or []):
                _accumulate(calls, tc)

        if not calls or last:
            return

        # ---- 执行工具 ----
        ordered = [calls[i] for i in sorted(calls)]
        results = []
        for i, c in enumerate(ordered):
            cid = c["id"] or f"call_{rnd}_{i}"
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                out = {"ok": False, "error_code": "BAD_JSON",
                       "message": f"参数不是合法 JSON：{c['arguments'][:60]}"}
            else:
                out = safe_dispatch(c["name"], args)
            results.append((cid, c["name"], out))

        yield ("tools", [{"name": n, "result": o} for _, n, o in results])

        # 回灌：assistant 的 tool_calls + 每个工具的 tool 消息
        msgs.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": n, "arguments": c["arguments"] or "{}"}}
                for (cid, n, _), c in zip(results, ordered)
            ],
        })
        for cid, n, o in results:
            msgs.append({
                "role": "tool",
                "tool_call_id": cid,
                "content": json.dumps(o, ensure_ascii=False),
            })
        # 下一轮要把上一轮的开场白和正式回答分开，故在正文前加一个换行
        yield ("delta", "\n\n")
