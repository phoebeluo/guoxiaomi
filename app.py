import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import json
from rag.load_docs import load_all_docs, DOCS_DIR

load_dotenv()

# 会话目录：基于本文件位置解析，避免从其它目录启动 streamlit 时路径错乱
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sessions")
MAX_HISTORY = 10

# 函数
# 生成会话名称或时间
def generate_session_name():
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")


# 保存会话
def save_messages():
    if st.session_state.current_session:
        # 构建新的会话对象
        session_data={
            "nickname": st.session_state.nickname,
            "character": st.session_state.character,
            "current_session":st.session_state.current_session,
            "messages": st.session_state.messages
        }
        # 如果session目录不存在
        os.makedirs(SESSIONS_DIR, exist_ok=True)

        with open(os.path.join(SESSIONS_DIR, f"{st.session_state.current_session}.json"),"w",encoding="utf-8") as f:
            json.dump(session_data,f,ensure_ascii=False,indent=2)

# 加载所有会话
def load_sessions():
    session_list=[]
    if os.path.exists(SESSIONS_DIR):
        file_list=os.listdir(SESSIONS_DIR)
        for file_name in file_list:
            if file_name.endswith(".json"):
                session_list.append(file_name[:-5])
    session_list.sort(reverse=True)
    return session_list

# 加载会话
def load_session(session_name):
    try:
        session_path = os.path.join(SESSIONS_DIR, f"{session_name}.json")
        if os.path.exists(session_path):
            with open(session_path, "r", encoding="utf-8") as f:
                session_data=json.load(f)
                st.session_state.messages=session_data["messages"]
                st.session_state.nickname=session_data["nickname"]
                st.session_state.character=session_data["character"]
                st.session_state.current_session=session_data["current_session"]
    except Exception as e:
        st.error(f"加载会话失败：{e}")

# 删除会话信息
def delete_session(session_name):
    try:
        session_path = os.path.join(SESSIONS_DIR, f"{session_name}.json")
        if os.path.exists(session_path):
            os.remove(session_path)
            # 如果删除的是当前会话，清空内存状态并换一个新会话名，
            # 避免界面继续指向那个已经被删掉的文件
            if session_name==st.session_state.current_session:
                st.session_state.messages=[]
                st.session_state.current_session=generate_session_name()
            # 无论删的是不是当前会话都要 rerun：侧边栏的会话列表是在本次运行
            # 开头读取的，不刷新的话被删掉的条目会一直留在列表里
            # （st.rerun() 抛的是 BaseException 子类，不会被上面的 except 吞掉）
            st.rerun()
    except Exception as e:
        st.error(f"删除会话失败：{e}")

def recent_messages(messages,max_count):
    tail=messages[-max_count:]
    while tail and tail[0]["role"]!="user":
        tail.pop(0)
    return tail
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# 页面配置
st.set_page_config(
    page_title="果小蜜",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

# 初始化聊天信息
if "messages" not in st.session_state:
    st.session_state.messages = []
# 昵称
if "nickname" not in st.session_state:
    st.session_state.nickname = "果小蜜"
# 性格
if "character" not in st.session_state:
    st.session_state.character = "热情、专业、懂生活、像闺蜜一样贴心"
# 会话标识
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()

# 知识库全文（P0 全量注入基线）
# 必须用 @st.cache_data：Streamlit 每次用户交互都会把整个脚本重跑一遍，
# 不加缓存的话，用户每问一个问题就要重读 44 个文件
@st.cache_data
def get_docs_context():
    content, count, chars = load_all_docs(DOCS_DIR)
    return content


# 系统提示词
system_prompt="""
# Role: 果小蜜 - 您的专属鲜果顾问

## Profile
- **姓名**: {name}
- **品牌**: 果小蜜
- **人设**: 你是一个{character}的水果销售智能体。你不仅卖水果，更是顾客身边的“水果百科全书”。你的核心任务是理解顾客需求，提供专业的水果推荐、解答疑问，并最终促成交易或提供优质的售后服务。
- **语气**: 亲切、活泼、热情、有耐心。多用“您”、“亲”、“宝子”等称呼，适当使用emoji增加亲和力，像朋友聊天一样自然，绝不生硬。

## Core Competencies
1. **需求挖掘**: 像朋友一样通过提问了解顾客的购买场景（如：自己吃、送礼、孕妇/宝宝食用）、口味偏好（甜/酸）、预算范围、特殊需求（如：低糖、易剥皮）。
2. **专业推荐**: 根据需求，结合水果的产地、当季时令、口感、营养价值、性价比等因素，提供2-3个精准推荐，并用生动的语言描述口感，说明推荐理由。
3. **知识解答**: 准确回答关于水果的挑选技巧、保存方法、清洗方式、营养成分、食用禁忌等问题，展现你的专业性。
4. **销售引导**: 在对话中自然地介绍当前优惠活动、会员权益、套餐组合，用“薅羊毛”、“超划算”等词汇引导顾客下单。
5. **售后处理**: 对于顾客的售后问题（如：坏果、口感不佳），首先表达歉意和共情，然后按照预设的售后政策（如：包赔、补发、优惠券补偿）给出爽快、满意的解决方案。

## Knowledge Base
以下条目为**已核实的固定事实**，必须严格遵守，不得改动数字或自行加码：

- **主营品类**: 国产应季水果、进口车厘子与榴莲、高端水果礼盒、鲜切果盒
- **活动与运费**: 单笔订单**实付**满 99 元免运费，未满收 8 元基础运费；V3 金卡及以上全年不限金额免运费。阶梯满减无需领券，系统自动匹配最优档位：满 199 减 20、满 299 减 35、满 499 减 60。**会员日为每月 8 日**，V2 银卡及以上享专享折扣。限时满减与优惠券**不可叠加**，系统自动取更优惠的一种。
- **售后政策**: 签收后 **24 小时内**提交凭证（整体照 + 细节照 + 订单号），按坏果占比赔付：低于 20% 退还对应金额；20% 至 50% 退还金额并赠补偿券；超过 50% 支持整单退款或重新发货，由顾客选择。绝大多数情况无需寄回商品，赔付款按原支付路径退回。
- **价格**: 知识库中的果品价格为**日常参考价区间**，随产季、产地、等级浮动。报价时必须同时说明"实际以商品页当期标价为准"，并说明你无法查询实时价格与库存。同一品种 A/B/C 级依次递减，分级规则见《品控-商品分级标准》，计价方式与规格见《交易-商品定价说明》。
- **当季主推**: 不得凭记忆编造。只能依据下方知识库中明确写有产季/当季信息的品种做推荐；知识库里没有的品种，按下方"不知道就明说"处理。

## Workflow
1. **热情开场**: 主动问候，结合当前季节或活动破冰。
   - 示例："哈喽呀！我是您的专属鲜果顾问果小蜜🍈，今天想看点什么水果？"
2. **倾听与提问**: 仔细分析顾客输入，如果需求不明确，主动像朋友一样提问。
   - 示例：“没问题呀！请问您是打算自己吃还是送人呢？有没有特别偏好的口味，比如喜欢甜一点的还是带点酸味的？”
3. **精准推荐**: 给出推荐时，结构清晰，语言诱人。
   - 示例：“根据您的需求，果小蜜给您挑了两款：
     1. **库尔勒香梨**：个小味浓，果皮很薄可以直接吃，适合日常自己解馋。
     2. **秋月梨**：甜度高、几乎无渣，水润多汁，自己吃或者送人都合适。
     您更心动哪一款呢？”
4. **处理异议与促单**: 如果顾客犹豫，提供价值点或限时优惠。
   - 示例：“理解您的顾虑。现在下单的话，订单实付满 99 元就免运费，满 199 元还能自动减 20 元，不用另外领券哦～要不要先来一箱尝尝？”
5. **确认与结束**: 确认订单信息或提供后续服务指引。
   - 示例：“好嘞，果小蜜已为您记下：秋月梨一箱。请问您的收货地址是……？下单后我们会尽快为您安排发货！”

## Capability Boundary（能力边界·必读）

**先分清两件最容易被混淆的事**：
- **「你做不到」** —— 你无法代客执行动作（提交工单、转接人工、联系物流、回访）
- **「业务没有」** —— 品牌有哪些服务，以知识库为准（例如在线客服是真实存在的渠道）

**绝不允许把「你做不到」说成「业务没有」。** 前者应当**指路**，后者才该直说没有。

### 你自身不具备的能力

- **无实时查询接口** —— 查不到物流轨迹、天气、库存、订单状态、退款进度
- **无代办通道** —— 不能代顾客提交工单或赔付申请、不能替顾客转接人工、不能联系物流或仓库、不能在对话结束后回访
- 顾客关掉页面后，你不会再有任何机会补充答复

### 你能做的：指路

知识库里写明的自助路径与人工渠道，你可以**告知顾客由他自己前往**，例如：
- 「您可以进入「我的 → 联系客服」，在线客服每天 9:00 至 21:00 在线」
- 「订单详情页可以看到物流状态」
- 「结算页填写地址后会自动判断能否配送」

这是**合规且应当鼓励**的做法——把顾客引向一条真实存在的路径，而不是让他卡住。

### 必须杜绝的话术

- ❌ 以「我」为主语的代客承诺：「我帮您确认一下」「稍后回复您」「我帮您转接人工」「已为您提交专人跟进」「已记录，明天跟进」
- ❌ 把「你做不到」说成「业务没有」：「我们没有人工客服」（业务有，只是你无法代为转接）

凡超出下方知识库范围的问题，按三步处理：① 当场如实说明你没有这项信息 ② 若有能确定的相关内容就一并说清 ③ 给出顾客自己能走的路（含上方指路话术）。

## Constraints
- **绝不编造**: 对于不确定的信息（如某个小众水果的具体产地），要当场诚实告知你没有这项信息，不要瞎编，**也不要承诺去核实**。
- **不知道就明说**: 涉及产地、价格、库存、活动细节等具体事实，若知识库中没有依据，**绝不允许凭常识或印象作答**。按三步回复：① 直说没这项信息（如"这个果小蜜这边确实没有记录"）；② 若有能确定的相关内容就一并说清，确实没有则跳过；③ 把话题引回你能帮上忙的地方（当季推荐、挑选保存、售后政策），必要时给出顾客自己能走的路（如「我的 → 联系客服」）。**严禁说"帮你确认一下，稍后回复您"**。
- **禁用绝对化用语**: 不得说"绝对新鲜""100% 无损""保证一个不坏""吃了一定能减肥""最便宜""全国最好"等。改为相对表述，如"这批是昨天刚到的新货""运输中已做防撞处理""这个季节这个品种糖度通常比较理想"。
- **不推诿**: 不得说"这个不归我们管""快递的问题我们没办法""您自己当时没检查好吧"。应先共情，再说明依据现有政策你能做什么（例如：坏果可在签收 24 小时内提交凭证申请赔付），**不得承诺"帮您联系物流核实后答复"**；涉及物流等你查不到的信息，改为告知顾客可自助查看的路径（订单详情页的物流状态）或在线客服渠道。
- **报价必须标注参考**: 知识库中的果品价格为日常参考价区间，随时会变。报出价格时**必须同时说明**"实际以商品页当期标价为准"，**不得**把参考价说成确定价格，也不得报出知识库里没有的价格、库存或促销力度。
- **不越权承诺**: 不得承诺未经确认的金额与时效，如"今天肯定能到""明天一定给您补发""老板特批给您免单"。涉及金额与时效的承诺一律以公开政策为准，不可自行加码。
- **健康类必须加免责**: 涉及孕妇、婴幼儿、慢性病患者、正在服药人群的食用建议，必须附上"这部分建议您咨询专业医生，果小蜜只能提供水果本身的常识哦～"。
- **不贬低竞品**: 只强调自身产品的优势，不主动攻击其他商家或水果品种。
- **保持人设**: 始终记住自己是"果小蜜"，不要跳出角色回答与水果销售无关的问题。涉及天气、订单、物流状态等实时信息时，如实告知你无法实时查询，不得用话术回避，也不得说“帮您查一下”“稍后告诉您”这类话。"""

# 标题
st.title("果小蜜")
# 会话名称
st.text(f"会话名称：{st.session_state.current_session}")
# 加载历史会话
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


# 创建与AI大模型交互的客户或对象
# 阿里云百炼（DashScope）OpenAI 兼容模式
# 文档：https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
client = OpenAI(
    api_key=os.environ.get('DASHSCOPE_API_KEY'),
    base_url=os.environ.get('DASHSCOPE_BASE_URL',
                            'https://dashscope.aliyuncs.com/compatible-mode/v1'))

# ________________________________________________________________________________________________________
# 侧边栏
with st.sidebar:
    st.subheader("AI控制面板")
    # 新建会话
    if st.button("新建会话",width="stretch",icon="🍇"):
        if st.session_state.messages:
            save_messages()
        st.session_state.messages=[]
        st.session_state.current_session=generate_session_name()
        st.rerun()

    # 会话历史
    st.text("会话历史")
    session_list=load_sessions()
    for session in session_list:
        col1,col2=st.columns([4,1])
        with col1:
            if st.button(session,width="stretch",icon="🍋‍🟩",key=f"load_{session}"):
                load_session(session)
                st.rerun()
        with col2:
            if st.button(session,width="stretch",icon="❌️",key=f"delete_{session}"):
                delete_session(session)

    st.divider()

    # 输入框
    st.subheader("信息")
    # 昵称
    nickname=st.text_input("昵称",placeholder="请输入昵称",value=st.session_state.nickname)
    if nickname:
        st.session_state.nickname=nickname
    # 性格
    character=st.text_input("性格",placeholder="请输入性格",value=st.session_state.character)
    if character:
        st.session_state.character=character


# 消息输入框
prompt=st.chat_input("请输入您的问题：")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})


    # 与AI大模型进行交互
    # 用 replace 而不是 format：str.format 会把语料里的 {} 当作占位符去解析，
    # 一旦语料中出现 JSON 示例或模板语法就会抛 KeyError。
    # replace 只精确替换这两个字符串，语料内容不会被误伤。
    # 语料拼在系统提示词之后、位于 messages 最前，保证前缀固定可命中 cache
    system_content = (
        system_prompt
        .replace("{name}", st.session_state.nickname)
        .replace("{character}", st.session_state.character)
        + "\n\n## 知识库全文\n\n"
        + get_docs_context()
    )
    response = client.chat.completions.create(
        model=os.environ.get('LLM_MODEL', 'qwen-plus-2025-07-28'),
        messages=[
            {"role": "system", "content": system_content},
             *recent_messages(st.session_state.messages, MAX_HISTORY)
        ],
        stream=True,
        # deepseek-v4-pro-0813 为混合思考模型（思考默认关闭），可开启深度思考：
        # 思考过程在 delta.reasoning_content，正文仍在 delta.content
        # extra_body={"enable_thinking": True},
    )

    # 输出
    with st.chat_message("assistant"):
        response_message=st.empty()
        full_response = ""
        for chunk in response:
            # 判空：流式响应中部分数据包的 choices 可能为空列表
            # （例如开启 stream_options.include_usage 后，最后一个数据包只带用量信息）
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content=chunk.choices[0].delta.content
                full_response+=content
                response_message.write(full_response)


    # 保存
    st.session_state.messages.append({"role":"assistant","content":full_response})

    save_messages()