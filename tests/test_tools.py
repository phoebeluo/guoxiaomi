"""W9 · 三个工具的单测：正常 / 缺参 / 越权 各 3 例，另加「与语料同源」校验。

为什么单独测「与语料同源」
------------------------
工具里写死了一张事实表（行政区、等级阈值、运费门槛）。它和 `data/docs/` 是两份副本，
**改了一处忘另一处，就会出现「工具说 V3 满 2000、知识库说满 3000」**——
用户听到的回答取决于走没走工具，这是最难查的一类 bug。
所以这里直接读语料，校验数字与名单都在原文里出现。
"""

import os
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from agent.tools import (          # noqa: E402
    EXCLUDED, MEMBER_LEVELS, ORDERS, SUBURBAN_DISTRICTS, TOOL_SCHEMAS,
    URBAN_DISTRICTS, ToolError, calc_member_level, check_delivery,
    query_order_status, safe_dispatch,
)

DOCS = os.path.join(PROJECT, "data", "docs")


def _read(name):
    return open(os.path.join(DOCS, name), encoding="utf-8").read()


# ======================================================================
# 正常路径
# ======================================================================

def test_delivery_urban_charges_base_freight():
    r = check_delivery("武侯", 79)
    assert r["serviceable"] and r["area_type"] == "urban"
    assert r["freight_yuan"] == 8.0
    assert r["district"] == "武侯区"          # 口语「武侯」应被规范化


def test_delivery_suburban_adds_remote_surcharge():
    r = check_delivery("都江堰市", 50)
    assert r["area_type"] == "suburban"
    assert r["freight_yuan"] == 12.0           # 8 + 4


def test_delivery_v3_free_shipping_beats_amount():
    # 会员等级优先于金额：V3 满 50 元也免运费
    assert check_delivery("武侯区", 50, member_level=3)["freight_yuan"] == 0.0


def test_delivery_out_of_city_not_serviceable():
    r = check_delivery("重庆市渝中区")
    assert r["serviceable"] is False and "不在" in r["note"]


def test_member_level_boundary_is_inclusive():
    # 「满 500 元」→ 恰好 500 就是 V2，边界取「达到即升级」
    assert calc_member_level(500)["level"] == "V2"
    assert calc_member_level(499.99)["level"] == "V1"
    assert calc_member_level(5000)["level"] == "V4"


def test_member_level_gap_to_next():
    r = calc_member_level(2100)
    assert r["level"] == "V3" and r["next_level"] == "V4"
    assert r["gap_to_next"] == 2900.0


def test_order_status_lookup_is_case_insensitive():
    assert query_order_status("gxm20261031002")["status"] == "配送中"


# ======================================================================
# 缺参（模型漏给必填字段）
# ======================================================================

@pytest.mark.parametrize("name,args", [
    ("check_delivery", {}),
    ("calc_member_level", {}),
    ("query_order_status", {}),
])
def test_missing_required_arg(name, args):
    r = safe_dispatch(name, args)
    assert r["ok"] is False and r["error_code"] == "MISSING_ARG"


# ======================================================================
# 越权 / 非法输入（模型编造工具名、塞多余字段、给越界值）
# ======================================================================

def test_unknown_tool_rejected():
    r = safe_dispatch("refund_approve", {"order_id": "GXM20261101003"})
    assert r["ok"] is False and r["error_code"] == "UNKNOWN_TOOL"


def test_extra_arg_rejected():
    r = safe_dispatch("calc_member_level", {"annual_paid": 100, "user_id": "u1"})
    assert r["ok"] is False and r["error_code"] == "UNKNOWN_ARG"


def test_wrong_type_rejected():
    r = safe_dispatch("calc_member_level", {"annual_paid": "两千"})
    assert r["ok"] is False and r["error_code"] == "BAD_TYPE"


def test_order_id_format_rejected():
    r = safe_dispatch("query_order_status", {"order_id": "12345"})
    assert r["ok"] is False and r["error_code"] == "BAD_FORMAT"


def test_order_not_found_is_explicit():
    # 查不到必须显式报错，不能返回空对象（否则模型会编状态）
    r = safe_dispatch("query_order_status", {"order_id": "GXM99999999999"})
    assert r["ok"] is False and r["error_code"] == "ORDER_NOT_FOUND"


def test_amount_out_of_range_rejected():
    r = safe_dispatch("calc_member_level", {"annual_paid": 10 ** 9})
    assert r["ok"] is False and r["error_code"] == "OUT_OF_RANGE"


def test_dispatch_raises_for_unknown_tool():
    with pytest.raises(ToolError):
        from agent.tools import dispatch
        dispatch("transfer_to_human", {})


# ======================================================================
# 与语料同源（防止工具事实表与知识库漂移）
# ======================================================================

def test_districts_all_appear_in_corpus():
    text = _read("配送-服务城市与区域.md")
    missing = [d for d in URBAN_DISTRICTS + SUBURBAN_DISTRICTS if d not in text]
    assert not missing, f"这些行政区在语料里找不到，工具自己编的：{missing}"


def test_member_thresholds_appear_in_corpus():
    text = _read("会员-等级与权益.md")
    for _, code, _, need, _ in MEMBER_LEVELS:
        assert code in text, f"语料里没有等级 {code}"
        if need:
            assert str(int(need)) in text, f"语料里没有 {code} 的门槛 {int(need)}"


def test_freight_numbers_appear_in_corpus():
    text = _read("配送-运费与包邮门槛.md")
    for num in ("99", "8", "4", "12"):
        assert num in text, f"运费数字 {num} 不在语料里"


def test_schemas_match_implementations():
    """schema 里声明的名字必须都有实现，反之亦然 —— 否则模型会调到一个空函数。"""
    from agent.tools import _DISPATCH
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(_DISPATCH)


def test_excluded_capabilities_are_not_tools():
    """排除清单里的能力绝不能出现在工具里（W9 的核心判断）。"""
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    for key in EXCLUDED:
        assert key not in " ".join(names), f"{key} 不该被做成工具"


def test_mock_orders_have_consistent_status_code():
    for oid, o in ORDERS.items():
        assert o["order_id"] == oid
        assert o["status_code"] in {"DELIVERED", "SHIPPING", "PENDING"}
