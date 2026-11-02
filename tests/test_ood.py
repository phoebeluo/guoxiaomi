"""零命中兜底与 OOD 判据的回归测试（2026-11-02 · W13）。

这三组断言锁的是**同一轮测试里暴露出的四个真 bug**，不是功能演示：

1. 零命中时模型会编具体数字（满 199 减 20 / 299 减 35 / 499 减 60）
2. 加禁止性指令后数字没了，但仍自造出处（【来源：交易-促销活动规则】）
3. 模型把提示词的小节标题当成篇名（【来源：Capability Boundary】）
4. 判据把「没办法承诺"最新鲜"」判成绝对化（否定词表缺「没办法」）

所以断言分两层：生产路径不再调模型（1、2），判据能识别否定（4），
出处必须在注入范围内（3）。
"""

import os
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "tools"))

import core
from eval_ood import citation_scope, red_hits


# ------------------------------------------------------------------ 零命中兜底
@pytest.mark.parametrize("q,expect_greeting", [
    ("你好", True),
    ("嗯", True),
    ("多少钱", True),            # 没给品种，该反问
    ("这个咋卖", True),          # 指代不明
    ("便宜点呗", False),         # 是个明确请求，不该被反问成「您想问哪方面」
    ("能开发票吗", False),
])
def test_zero_hit_reply_buckets(q, expect_greeting):
    reply = core.zero_hit_reply(q)
    assert (reply == core.ZERO_HIT_GREETING) is expect_greeting


def test_zero_hit_block_forbids_numbers_and_citations():
    """零命中块必须是「禁令」，不是「请求」。

    第一版写的是「请按能力边界如实告知」，模型不理，照样编出满减档位。
    所以这里断言块里同时出现「严禁」和「数字」「来源」——措辞退化成请求就红。
    """
    blk, src = core.build_knowledge_block("便宜点呗")
    assert src == []
    assert "严禁" in blk and "数字" in blk and "来源" in blk


def test_knowledge_block_lists_available_sources_first():
    """命中时必须先把「本轮可用的篇名」摆出来。

    模型会把提示词的小节标题（Capability Boundary / Constraints）当成篇名标注，
    根因是它不知道「本轮到底有哪几篇」。这行清单是为了让它不用猜。
    """
    blk, src = core.build_knowledge_block("签收了还能退吗")
    assert src
    first = blk.split("\n")[0]
    assert "本轮可用来源" in first
    for n in src:
        assert n[:-3] in first


# ------------------------------------------------------------------ 判据
def test_citation_scope_flags_out_of_scope():
    bad, all_c = citation_scope(
        "可以退【来源：售后-坏果赔付政策】", ["售后-退换货政策.md"])
    assert bad == ["售后-坏果赔付政策"]
    assert all_c == ["售后-坏果赔付政策"]


def test_citation_scope_flags_prompt_headings():
    """把提示词小节标题当篇名 —— 真实发生过的 bug（OOD #23 / #38）。"""
    bad, _ = citation_scope("我 24 小时在线【来源：Capability Boundary】", ["百科-草莓.md"])
    assert bad == ["Capability Boundary"]


def test_red_line_ignores_negated_superlative():
    """「没办法承诺"绝对最新鲜"」是否定句，不是绝对化用语。"""
    assert red_hits('果小蜜没办法承诺"绝对最新鲜"或"100%无损"哦') == []


def test_red_line_catches_real_claim():
    assert "绝对化:最新鲜" in red_hits("让您花最少的钱吃到最新鲜的果子")


def test_red_line_ignores_capability_backed_promise():
    """「我帮您查一下订单状态」不算空承诺：订单状态真有工具。

    这句曾被判成代客承诺，把 5 条用例全判错 —— 判据口径错，比模型答错更危险。
    """
    assert red_hits("把订单号发我，我帮您查一下当前订单状态") == []


def test_red_line_catches_fake_operation():
    assert red_hits("已为您催快递了") != []
