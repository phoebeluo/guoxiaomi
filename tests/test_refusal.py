"""能力边界拒答测试。

最重要的一条是 `test_no_false_reject_on_in_library_cases`：
库内 54 道题一条都不能被规则拦住。误拒的代价（能答的问题被推开）
远高于漏拒（靠 prompt 兜底），所以这条一旦失败，宁可删规则。
"""

import json
import os

import pytest

from rag.refusal import all_rules, check

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(PROJECT, "eval", "cases.jsonl")


@pytest.fixture(scope="module")
def cases():
    with open(CASES, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestNoFalseReject:
    def test_no_false_reject_on_in_library_cases(self, cases):
        """库内题（有 gold_sources）不允许被拦下。"""
        bad = [(c["id"], c["q"], check(c["q"])[0])
               for c in cases if c.get("gold_sources") and check(c["q"])]
        assert not bad, f"误拒 {len(bad)} 题：{bad}"

    def test_catches_capability_cases(self, cases):
        """能力类库外题必须被拦（#8 内部流程 / #9 查物流 / #15 登记回访）。"""
        out = [c for c in cases if not c.get("gold_sources")]
        hit = {c["id"] for c in out if check(c["q"])}
        assert {8, 9, 15} <= hit, f"应拦截但漏了：{ {8, 9, 15} - hit }"

    def test_absolute_promise_not_blocked(self):
        """绝对化承诺类不归拒答管，交给 prompt 的「禁用绝对化用语」。"""
        assert check("能保证我的果子一个都不会坏吗？") is None
        assert check("这水果能保证绝对新鲜吗？") is None


class TestRuleBehavior:
    @pytest.mark.parametrize("q,rule", [
        ("我的订单物流三天没更新了，你帮我查一下到底到哪了。", "查订单物流"),
        ("帮我改一下收货地址", "代改订单"),
        ("你帮我登记一下，让专人明天联系我处理。", "登记回访"),
        ("帮我申请一下赔付", "赔付审批"),
        ("你们内部对投诉是怎么分级的？", "内部流程细节"),
    ])
    def test_expected_rule(self, q, rule):
        got = check(q)
        assert got is not None and got[0] == rule

    def test_reply_is_actionable(self):
        """拒答话术必须含「用户自己能走的路」，不能只说做不到。"""
        got = check("我的快递到哪了")
        assert got is not None
        assert "联系客服" in got[1] or "订单详情" in got[1]

    def test_empty_and_short(self):
        assert check("") is None
        assert check("   ") is None
        assert check("在吗") is None
        assert check(None) is None

    def test_rules_have_names_and_replies(self):
        rules = all_rules()
        assert len(rules) >= 5
        for name, pattern, reply in rules:
            assert name and pattern and len(reply) > 20
