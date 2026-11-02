"""W9 · 三个「真能兑现」的工具（Function Calling）。

为什么只有三个
==============
计划里明确删掉了库存、优惠券试算两个工具。理由是同一个：
**做工具的目的是消灭虚假承诺，不是变多。**
库存和优惠券必须有真实数据源才能算，用假数据搭一个「能查库存」的工具，
等于用更精致的方式制造新的虚假承诺——比原来老实说「我查不到」更糟。

同理，下面这几个**永远不做成工具**（见 EXCLUDED，W9 第二天的排除清单）：
  转人工        —— 没有工单系统，做了就是假装能转
  实时物流      —— 没有承运商接口，编个「已到达成都转运中心」是纯编造
  赔付审批      —— 涉及资金与人工核验，机器不能代批

三个工具的数据来源
==================
全部与 `data/docs/` 的语料**同源**，不是另起一套规则：

  配送范围/时效  ← 配送-服务城市与区域.md、配送-范围与时效.md、配送-运费与包邮门槛.md
  会员等级       ← 会员-等级与权益.md
  订单状态       ← 本地 fixture（见 ORDERS），代表 W12 之后接真实订单库的位置

⚠️ 订单状态是 **mock**。它存在的意义是演示「调用 → 返回结构化事实 → 组织回答」这条链路，
   并让「我帮您查一下」这句话第一次变成真的。W12 服务化时把 ORDERS 换成 DB 查询即可，
   工具的 schema 与返回结构不需要改。

为什么工具返回 dict 而不是自然语言
==================================
让模型自己组织话术，避免把客服语气写死在代码里；
同时结构化结果可以直接进评测（W11 的「参数正确率 / 任务完成率」要的是字段，不是句子）。
"""

from __future__ import annotations

import re

# ======================================================================
# 与语料同源的事实表
# ======================================================================

CITY = "成都"

# 市区及周边城区：当日达（配送-服务城市与区域.md）
URBAN_DISTRICTS = (
    "锦江区", "青羊区", "金牛区", "武侯区", "成华区",
    "龙泉驿区", "青白江区", "新都区", "温江区", "双流区",
    "郫都区", "新津区", "高新区", "经开区",
)

# 远郊区县与乡镇：次日达，运费加价（同上 + 运费文档）
SUBURBAN_DISTRICTS = (
    "都江堰市", "彭州市", "崇州市", "邛崃市", "简阳市",
    "金堂县", "大邑县", "蒲江县",
)

# 会员等级：V1 注册即享 → V4 满 5000（会员-等级与权益.md）
MEMBER_LEVELS = (
    (1, "V1", "普卡", 0.0, "生日当月双倍积分"),
    (2, "V2", "银卡", 500.0, "会员专享价、每月 5 元无门槛券"),
    (3, "V3", "金卡", 2000.0, "全年不限额免运费、优先发货"),
    (4, "V4", "黑卡", 5000.0, "专属客服、新品优先购、免运费"),
)

# 运费：满 99 免；未满收 8；远郊未满再加 4（=12）；V3 及以上全免
FREE_SHIP_THRESHOLD = 99.0
BASE_FREIGHT = 8.0
REMOTE_SURCHARGE = 4.0
FREE_SHIP_MIN_LEVEL = 3

# 时效（配送-范围与时效.md）
URBAN_CUTOFF_HOUR = 15

# 订单 mock：字段刻意与真实订单系统一致，W12 换 DB 时不用改上层
ORDERS = {
    "GXM20261028001": {
        "order_id": "GXM20261028001",
        "status": "已签收",
        "status_code": "DELIVERED",
        "placed_at": "2026-10-28 10:12",
        "items": [("车厘子 2斤装", 2), ("脐橙 5斤装", 1)],
        "amount_paid": 218.0,
        "carrier_note": "物流详情请到订单详情页查看",
    },
    "GXM20261031002": {
        "order_id": "GXM20261031002",
        "status": "配送中",
        "status_code": "SHIPPING",
        "placed_at": "2026-10-31 16:40",
        "items": [("金煌芒 3斤装", 1)],
        "amount_paid": 79.0,
        "carrier_note": "预计次日 12:00 前送达（市区 15:00 后下单）",
    },
    "GXM20261101003": {
        "order_id": "GXM20261101003",
        "status": "待出库",
        "status_code": "PENDING",
        "placed_at": "2026-11-01 09:05",
        "items": [("琯溪蜜柚 4个装", 1), ("蓝莓 2盒", 1)],
        "amount_paid": 109.0,
        "carrier_note": "当日 15:00 前下单，当日出库",
    },
}

# 明确不做成工具的能力（W9 排除清单，含理由）
EXCLUDED = {
    "转人工": "没有工单系统与坐席排班，做了就是假装能转接；正确做法是给「我的→联系客服」路径",
    "实时物流": "没有承运商接口；编一个「已到达 XX 转运中心」属于纯编造，比不回答更糟",
    "赔付审批": "涉及资金与凭证核验，必须人工；机器只能说明政策与所需材料",
    "库存查询": "无实时库存源，假数据会诱导「有货」承诺",
    "优惠券试算": "依赖用户账户券包，无数据源则只能编造可用券",
}

ORDER_ID_RE = re.compile(r"^GXM\d{11}$")


class ToolError(Exception):
    """工具执行失败。code 给机器判（评测用），message 给人看。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ======================================================================
# Function Calling schema（OpenAI 兼容）
# ======================================================================
# 参数约束写进 schema，而不是只写在代码里：模型看到 enum/范围才会自己收敛，
# 代码里的校验是**兜底**（模型仍可能给越界值）。

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_delivery",
            "description": (
                "查询某个行政区是否在配送范围内，并给出预计送达时效与所需运费。"
                "用户问「能不能送到我这」「多久到」「运费多少」时调用。"
                "只支持成都市行政区域。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "成都市下辖行政区名称，如「武侯区」「都江堰市」",
                        "maxLength": 12,
                    },
                    "order_amount": {
                        "type": "number",
                        "description": "订单实付金额（元），用于算运费；用户没提就不传",
                        "minimum": 0,
                        "maximum": 100000,
                    },
                    "member_level": {
                        "type": "integer",
                        "description": "会员等级数字 1-4；V3 及以上免运费。用户没提就不传",
                        "minimum": 1,
                        "maximum": 4,
                    },
                },
                "required": ["district"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_member_level",
            "description": (
                "按自然年累计实付金额计算会员等级、对应权益，以及距离下一级还差多少。"
                "用户问「我是什么会员」「怎么升级」「金卡有什么权益」「还差多少到金卡」时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_paid": {
                        "type": "number",
                        "description": "本自然年累计实付金额（元）",
                        "minimum": 0,
                        "maximum": 1000000,
                    },
                },
                "required": ["annual_paid"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order_status",
            "description": (
                "按订单号查询订单当前状态与商品明细。"
                "用户问「我的订单到哪了」「发货了没」「订单状态」并给出订单号时调用。"
                "订单号形如 GXM 加 11 位数字。没有订单号时必须先向用户索取，不得猜测。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，格式 GXM + 11 位数字",
                        "pattern": r"^GXM\d{11}$",
                        "maxLength": 14,
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]


# ======================================================================
# 参数校验（schema 的子集实现：required / type / enum / 范围 / 正则 / 多余字段）
# ======================================================================

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _validate(name: str, args: dict) -> dict:
    schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    params = schema["parameters"]
    props = params.get("properties", {})

    if not isinstance(args, dict):
        raise ToolError("BAD_ARGS", "参数必须是对象")

    missing = [k for k in params.get("required", []) if k not in args]
    if missing:
        raise ToolError("MISSING_ARG", f"缺少必填参数：{', '.join(missing)}")

    if params.get("additionalProperties") is False:
        extra = [k for k in args if k not in props]
        if extra:
            raise ToolError("UNKNOWN_ARG", f"不认识的参数：{', '.join(extra)}")

    for key, val in args.items():
        spec = props[key]
        want = spec.get("type")
        if want in _TYPE_MAP:
            # bool 是 int 的子类，必须排除，否则 True 会被当成 1
            if want != "boolean" and isinstance(val, bool):
                raise ToolError("BAD_TYPE", f"{key} 应为 {want}")
            if not isinstance(val, _TYPE_MAP[want]):
                raise ToolError("BAD_TYPE", f"{key} 应为 {want}，实际是 {type(val).__name__}")
        if spec.get("enum") and val not in spec["enum"]:
            raise ToolError("BAD_VALUE", f"{key} 只能是 {spec['enum']} 之一")
        if "minimum" in spec and val < spec["minimum"]:
            raise ToolError("OUT_OF_RANGE", f"{key} 不能小于 {spec['minimum']}")
        if "maximum" in spec and val > spec["maximum"]:
            raise ToolError("OUT_OF_RANGE", f"{key} 不能大于 {spec['maximum']}")
        if "maxLength" in spec and isinstance(val, str) and len(val) > spec["maxLength"]:
            raise ToolError("BAD_VALUE", f"{key} 超过 {spec['maxLength']} 字")
        if "pattern" in spec and isinstance(val, str) and not re.match(spec["pattern"], val):
            raise ToolError("BAD_FORMAT", f"{key} 格式不符合 {spec['pattern']}")
    return args


# ======================================================================
# 工具 1：配送范围 / 时效 / 运费
# ======================================================================

def check_delivery(district: str, order_amount: float | None = None,
                   member_level: int | None = None) -> dict:
    """判断配送范围，返回时效与运费。

    行政区名做**规范化**再匹配：用户常说「武侯」「武侯区」「成都市武侯区」，
    直接字符串相等会把前两种判成「不配送」——那是工具自己制造的误判。
    """
    raw = district.strip().replace(" ", "")
    for prefix in ("成都市", "成都"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    name = raw if raw.endswith(("区", "市", "县")) else raw + "区"

    if name in URBAN_DISTRICTS:
        area, eta = "urban", "当日 15:00 前下单当日 18:00 前送达；15:00 后下单次日 12:00 前"
        remote = False
    elif name in SUBURBAN_DISTRICTS:
        area, eta = "suburban", "次日 18:00 前送达"
        remote = True
    else:
        return {
            "serviceable": False,
            "district": district,
            "city": CITY,
            "note": f"{district} 不在果小蜜配送范围内（目前仅服务成都市行政区域）。"
                    "建议更换收货地址，或在「我的 → 联系客服」咨询自提与转寄。",
        }

    freight = None
    if order_amount is not None:
        if member_level is not None and member_level >= FREE_SHIP_MIN_LEVEL:
            freight = 0.0
        elif order_amount >= FREE_SHIP_THRESHOLD:
            freight = 0.0
        else:
            freight = BASE_FREIGHT + (REMOTE_SURCHARGE if remote else 0.0)

    return {
        "serviceable": True,
        "district": name,
        "city": CITY,
        "area_type": area,
        "eta": eta,
        "freight_yuan": freight,
        "free_ship_threshold": FREE_SHIP_THRESHOLD,
        "note": "运费按实付金额（扣券与积分抵扣后）判断是否满 99 元" if freight else "",
    }


# ======================================================================
# 工具 2：会员等级
# ======================================================================

def calc_member_level(annual_paid: float) -> dict:
    """按自然年累计实付金额算等级。边界取「达到即升级」（满 500 = V2）。"""
    cur = MEMBER_LEVELS[0]
    for lv in MEMBER_LEVELS:
        if annual_paid >= lv[3]:
            cur = lv
    nxt = next((lv for lv in MEMBER_LEVELS if lv[3] > annual_paid), None)
    return {
        "level": cur[1],
        "level_no": cur[0],
        "name": cur[2],
        "perks": cur[4],
        "annual_paid": annual_paid,
        "next_level": nxt[1] if nxt else None,
        "next_level_at": nxt[3] if nxt else None,
        "gap_to_next": round(nxt[3] - annual_paid, 2) if nxt else 0.0,
        "reset_note": "累计金额按自然年计算，每年 1 月 1 日重置后重新累计",
    }


# ======================================================================
# 工具 3：订单状态（mock）
# ======================================================================

def query_order_status(order_id: str) -> dict:
    """按订单号查状态。查不到时**明确报错**，不返回空对象。

    为什么不返回 {}：空对象会让模型以为「查到了但没有内容」，进而编造状态；
    显式 ToolError 才能让它老实说「这个订单号查不到」。
    """
    oid = order_id.strip().upper()
    if not ORDER_ID_RE.match(oid):
        raise ToolError("BAD_FORMAT", "订单号格式应为 GXM 加 11 位数字")
    order = ORDERS.get(oid)
    if not order:
        raise ToolError("ORDER_NOT_FOUND", f"没有查询到订单 {oid}，请核对订单号")
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "status_code": order["status_code"],
        "placed_at": order["placed_at"],
        "items": [{"name": n, "qty": q} for n, q in order["items"]],
        "amount_paid": order["amount_paid"],
        "note": order["carrier_note"],
    }


# ======================================================================
# 统一入口
# ======================================================================

_DISPATCH = {
    "check_delivery": check_delivery,
    "calc_member_level": calc_member_level,
    "query_order_status": query_order_status,
}


def dispatch(name: str, args: dict) -> dict:
    """模型给出的 name + args → 工具结果。

    任何失败都转成 {"ok": False, "error_code", "message"} 而不是抛异常：
    调用方是对话循环，抛异常会打断整轮；而且错误信息本身要回给模型，
    让它据此向用户解释（「订单号格式不对」比静默失败有用得多）。
    """
    fn = _DISPATCH.get(name)
    if not fn:
        raise ToolError("UNKNOWN_TOOL", f"没有名为 {name} 的工具")
    return fn(**_validate(name, args))


def safe_dispatch(name: str, args: dict) -> dict:
    """dispatch 的不抛异常版本，供对话循环直接使用。"""
    try:
        return {"ok": True, "result": dispatch(name, args)}
    except ToolError as e:
        return {"ok": False, "error_code": e.code, "message": e.message}
    except TypeError as e:
        return {"ok": False, "error_code": "BAD_ARGS", "message": str(e)}


if __name__ == "__main__":
    # 手工冒烟：不调模型，只验工具本身
    print("工具数：", len(TOOL_SCHEMAS))
    print("配送（武侯区 实付 79 元）：", check_delivery("武侯", 79))
    print("配送（都江堰 未满 99）：", check_delivery("都江堰市", 50))
    print("会员（2100 元）：", calc_member_level(2100))
    print("订单（存在）：", query_order_status("gxm20261031002")["status"])
    print("订单（不存在）：", safe_dispatch("query_order_status", {"order_id": "GXM00000000000"}))
    print("未知工具：", safe_dispatch("refund_approve", {}))
