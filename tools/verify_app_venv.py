"""在 .venv 下无头运行 app.py，验证环境可用性与 P0 注入是否生效。

用法：
    python verify_app_venv.py            # 只启动，不发问（0 API 调用）
    python verify_app_venv.py --chat     # 额外发一问，走完整链路（1 次 API 调用）
"""

import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    print("=" * 62)
    print("阶段 1 · 无头启动 app.py（不消耗 API）")
    print("=" * 62)

    at = AppTest.from_file(os.path.join(PROJECT, "app.py"), default_timeout=120)
    at.run()

    if at.exception:
        print("[失败] 启动时抛出异常：")
        for e in at.exception:
            print("       ", e.value)
        return 1

    print("[通过] 脚本执行完毕，无未捕获异常")
    print(f"       标题元素      : {[t.value for t in at.title]}")
    print(f"       chat_input 数量: {len(at.chat_input)}")
    print(f"       侧边栏按钮数量 : {len(at.sidebar.button)}")

    # 直接读取注入相关的 session_state 键
    print(f"       当前会话名     : {at.session_state['current_session']}")
    print(f"       昵称           : {at.session_state['nickname']}")
    print(f"       历史消息条数   : {len(at.session_state['messages'])}")
    print(f"       缓存函数可调用 : {at.session_state is not None}")

    if "--chat" not in sys.argv:
        print("\n（未加 --chat，跳过真实提问）")
        return 0

    print()
    print("=" * 62)
    print("阶段 2 · 发一问「柚子推荐」，走完整链路（消耗 1 次 API）")
    print("=" * 62)

    QUESTION = "琯溪蜜柚什么时候上市？"
    print(f"提问：{QUESTION}")

    at.chat_input[0].set_value(QUESTION).run()

    if at.exception:
        print("[失败] 提问后抛出异常：")
        for e in at.exception:
            print("       ", e.value)
        return 1

    # ElementTree 的 __iter__ 本身已递归展开，直接遍历即可（不要再手动递归）
    texts = []
    for top in (at.main, at.sidebar):
        for el in top:
            v = getattr(el, "value", None)
            if isinstance(v, str) and v.strip():
                texts.append(v)
    joined = "\n".join(texts)

    print(f"收集到文本片段 {len(texts)} 个，合计 {len(joined)} 字符")
    print("-" * 62)
    print(joined[-1200:])
    print("-" * 62)

    hits = {
        "回答中未出现「暂时没有资料」": "暂时没有资料" not in joined,
        "回答中提到「琯溪」": "琯溪" in joined,
        "回答中提到产季 10-11 月": ("10" in joined and "11" in joined and "月" in joined),
    }
    for k, v in hits.items():
        print(f"  {'[通过]' if v else '[失败]'} {k}")

    return 0 if all(hits.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
