import streamlit as st
import openai
import json
import datetime
import pandas as pd
import traceback
import re
from supabase import create_client

# ================= 1. 核心 Prompt (MVP 2.0 升级版：加入 NVC 模块) =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你不仅是一位结合了身心灵修行理论、实修和数据分析的“情绪资产管理专家”，同时也是一位深谙“非暴力沟通(NVC)”的心理咨询师。

你的任务是：
1. 接收用户输入的非结构化情绪日记。
2. 将其转化为结构化的情绪资产数据（量化评分）。
3. 运用非暴力沟通技术，将用户的情绪转化为深层的需求和建设性的表达。

# ================= 模块一：情绪资产量化 (核心资产，严格保持标准) =================

【情绪标签体系与评分标准】
请严格基于以下3个维度进行量化分析（分数范围：-5到+5）。你必须参考下表中的描述来判断分数：

1. 平静度 (Peace)
-5: 暴躁, 心绪发狂, 躁动不安
-4: 恐慌, 恐惧
-3: 焦虑, 迷茫, 困惑
-2: 不安, 担忧
-1: 轻度不安, 心绪不宁
0:  安静
+1: 平静, 内心平静，没有波澜
+2: 宁静, 内心一片祥和，无纷扰
+3: 安详, 内心安详，安稳
+4: 喜悦, 专注，注意力灌注，心流体验
+5: 狂喜, 意识清明，全然临在

2. 觉察度 (Awareness)
-5: 没有觉察概念，完全认同念头、情绪
-4: 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪，无法自控
-3: 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪
-2: 没有觉察，被情绪、念头带着跑，与其无意识认同；较多陷入极端情绪
-1: 没有觉察，被情绪、念头带着跑，与其无意识认同；偶尔陷入极端情绪
0:  没有觉察，被情绪、念头带着跑
+1: 偶尔有觉察，反省。事后一段时间才觉察、反省到情绪、念头
+2: 较多觉察，看见自己的情绪、念头；多数是事后觉察，少有事情发生当下觉察到
+3: 很多觉察，看见自己的情绪、念头；事后觉察，和事情发生当下觉察到都有
+4: 非常多觉察，看见自己的情绪、念头；当下觉察占比更高
+5: 全然临在，对念头、情绪完全觉知，且不被其影响

3. 能量水平 (Energy)
-5: 无法支配行动
-4: 极度累, 筋疲力尽, 提不起劲, 只想躺平
-3: 非常累
-2: 很累
-1: 累, 疲惫
0:  没有力气，但是不累，需要注入点能量的状态
+1: 稍微有点力气
+2: 有点力气但不多
+3: 有力气，能正常应对事物
+4: 活力满满, 干劲十足
+5: 精力过剩

# ================= 模块二：NVC 非暴力沟通引导 (新增价值模块) =================

【NVC 分析任务】
请深入洞察用户文字背后的“未满足的需要”，并将其转化为非暴力沟通的四要素（观察、感受、需要、请求）。
1. 识别评判：如果用户使用了评判性语言（如“他很懒”、“这很烦”），请在内心将其转化为客观观察。
2. 挖掘需要：情绪背后一定隐藏着某种需要（如：被尊重、安全感、连接、休息、秩序等）。
3. 生成引导：生成一段话，引导用户用 NVC 句式重组自己的表达。

# ================= 输出控制 =================

【输出要求】
1. 必须输出且仅输出纯净的 JSON。
2. 严禁 Markdown 标记。
3. 严禁尾部逗号。

【JSON输出格式】
{
  "summary": "对用户情绪日记的简短总结，不超过30字。",
  "scores": {
    "平静度": 整数, 
    "觉察度": 整数, 
    "能量水平": 整数
  },
  "nvc_guide": {
    "observation": "客观发生了什么（去除形容词）",
    "feeling": "识别到的情绪关键词",
    "need": "情绪背后未被满足的核心需要（如：被理解、空间、效率）",
    "empathy_response": "一句极具同理心的回应，示范如何用 NVC 表达用户的心声（不超过50字）。"
  },
  "key_insights": [
    "洞察点1",
    "洞察点2"
  ],
  "recommendations": {
    "身心灵调适建议": "不超过50字。"
  }
}
"""

# ================= 2. 数据库连接层 (保持原样) =================
@st.cache_resource
def init_supabase():
    try:
        if "SUPABASE_URL" in st.secrets:
            return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except:
        return None
    return None

def save_to_db(user_id, text, json_result):
    sb = init_supabase()
    if sb:
        try:
            sb.table("emotion_logs").insert({
                "user_id": user_id,
                "user_input": text,
                "ai_result": json_result
            }).execute()
        except: pass

def get_history(user_id):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
            return res.data
        except: return []
    return []

# ================= 3. AI 分析逻辑 (保持原样，含 +1 修复) =================
def clean_json_string(s):
    match = re.search(r'\{[\s\S]*\}', s)
    if match: s = match.group()
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    s = re.sub(r':\s*\+', ': ', s) # 保持去加号逻辑
    return s

def analyze_emotion(text, api_key):
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    content = ""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": f"【输入文本】\n{text}"}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content
        return json.loads(clean_json_string(content))
    except Exception as e:
        return {"error": str(e), "raw_content": content}

# ================= 4. 视觉组件 (保持原样：右侧刻度、单行HTML、高层级) =================
def get_gauge_html(label, score, icon, theme="peace"):
    percent = (score + 5) * 10
    
    colors = {
        "peace": ["#11998e", "#38ef7d", "#11998e"],
        "awareness": ["#8E2DE2", "#4A00E0", "#6a0dad"],
        "energy": ["#f12711", "#f5af19", "#e67e22"]
    }
    c = colors.get(theme, colors["peace"])
    
    return f"<div style='display: flex; flex-direction: column; align-items: center; width: 80px;'><div style='height: 160px; width: 44px; background: #f0f2f6; border-radius: 22px; position: relative; margin-top: 5px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);'><div style='position: absolute; top: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>+5</div><div style='position: absolute; top: 50%; transform: translateY(-50%); left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>0</div><div style='position: absolute; bottom: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>-5</div><div style='position: absolute; bottom: 0; width: 100%; height: {percent}%; background: linear-gradient(to top, {c[0]}, {c[1]}); border-radius: 22px; transition: height 0.8s; z-index: 1;'></div><div style='position: absolute; bottom: {percent}%; left: 50%; transform: translate(-50%, 50%); background: #fff; color: {c[2]}; font-weight: 800; font-size: 13px; padding: 3px 8px; border-radius: 10px; border: 1.5px solid {c[2]}; box-shadow: 0 3px 8px rgba(0,0,0,0.15); z-index: 10; min-width: 28px; text-align: center; line-height: 1.2;'>{score}</div></div><div style='margin-top: 10px; font-size: 13px; font-weight: 600; color: #666; text-align: center;'>{icon}<br>{label}</div></div>"

# ================= 5. 主程序 =================
st.set_page_config(page_title="AI情绪资产助手", page_icon="🦁", layout="centered")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; font-weight: bold; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state.user_id = "guest_001"

with st.sidebar:
    st.header("⚙️ 系统设置")
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ 神经网络已连接")
    else:
        api_key = st.text_input("输入 DeepSeek Key", type="password")
    st.session_state.user_id = st.text_input("账户 ID", value=st.session_state.user_id)

st.title("🦁 AI情绪资产助手")

tab1, tab2 = st.tabs(["📝 觉察录入", "📊 情绪资产大盘"])

# --- Tab 1 ---
with tab1:
    st.write("")
    user_input = st.text_area("记录当下身心感受...", height=100, placeholder="在此输入你的觉察记录...")
    
    if st.button("⚡️ 铸造情绪资产", type="primary"):
        if not user_input or not api_key:
            st.toast("⚠️ 请输入内容或检查 Key")
        else:
            with st.spinner("🤖 AI 正在倾听并进行 NVC 转化..."): # 文案微调，更有温度
                result = analyze_emotion(user_input, api_key)
                
                if "error" in result:
                    st.error("系统故障，请重试")
                    with st.expander("查看详细报错"): st.code(result.get('raw_content'))
                else:
                    save_to_db(st.session_state.user_id, user_input, result)
                    st.toast("✅ 觉察已铸造")
                    
                    st.info(f"📝 {result.get('summary')}")

                    st.markdown("##### 📊 情绪资产水平")
                    
                    sc = result.get("scores", {})
                    h1 = get_gauge_html("平静度", sc.get("平静度", 0), "🕊️", "peace")
                    h2 = get_gauge_html("觉察度", sc.get("觉察度", 0), "👁️", "awareness")
                    h3 = get_gauge_html("能量值", sc.get("能量水平", 0), "🔋", "energy")
                    
                    # 图表容器
                    container_html = f"<div style='display: flex; justify-content: space-around; align-items: flex-end; margin: 20px 0; width: 100%;'>{h1}{h2}{h3}</div>"
                    st.markdown(container_html, unsafe_allow_html=True)

                    # === 新增模块：NVC 转化展示 ===
                    nvc = result.get("nvc_guide", {})
                    if nvc:
                        st.markdown("##### 🦒 NVC 深度转化")
                        st.markdown(f"""
                        <div style="background-color:#f3e5f5; padding:15px; border-radius:10px; border-left: 5px solid #9c27b0; margin-bottom: 20px; color: #4a148c;">
                            <p style="margin-bottom: 6px; font-size: 14px;"><b>👁️ 观察:</b> {nvc.get('observation', '...')}</p>
                            <p style="margin-bottom: 6px; font-size: 14px;"><b>❤️ 感受:</b> {nvc.get('feeling', '...')}</p>
                            <p style="margin-bottom: 6px; font-size: 14px;"><b>🌱 需要:</b> {nvc.get('need', '...')}</p>
                            <hr style="border-top: 1px dashed #ce93d8; margin: 10px 0;">
                            <p style="font-style: italic; font-weight: bold; font-size: 15px;">" {nvc.get('empathy_response', '')} "</p>
                        </div>
                        """, unsafe_allow_html=True)
                    # ================================

                    st.write("---")
                    
                    with st.expander("💡 深度洞察 (Deep Insights)", expanded=True):
                        for insight in result.get('key_insights', []):
                            st.markdown(f"**•** {insight}")
                    
                    st.markdown(f"""
                    <div style="background-color:#eafaf1; padding:15px; border-radius:8px; border: 1px dashed #27ae60; margin-top: 10px;">
                        <strong style="color:#27ae60;">💊 行动指南：</strong><br>
                        {result.get('recommendations', {}).get('身心灵调适建议')}
                    </div>
                    """, unsafe_allow_html=True)

# --- Tab 2 ---
with tab2:
    st.subheader("📈 情绪资产走势")
    if st.button("🔄 刷新大盘"):
        st.rerun()
    
    data = get_history(st.session_state.user_id)
    
    if data:
        chart_data = []
        for item in data:
            try:
                res = item['ai_result']
                if isinstance(res, str): res = json.loads(res)
                sc = res.get('scores', {})
                t = pd.to_datetime(item['created_at']) + pd.Timedelta(hours=8)
                chart_data.append({
                    "时间": t, 
                    "平静度": sc.get("平静度", 0),
                    "觉察度": sc.get("觉察度", 0),
                    "能量": sc.get("能量水平", 0)
                })
            except: continue
        
        if chart_data:
            df = pd.DataFrame(chart_data).sort_values('时间')
            st.line_chart(df, x='时间', y=['平静度', '觉察度', '能量'], color=["#2ecc71", "#9b59b6", "#e67e22"])
            st.markdown("---")
            for item in data:
                try:
                    res = item['ai_result']
                    if isinstance(res, str): res = json.loads(res)
                    t_str = (pd.to_datetime(item['created_at']) + pd.Timedelta(hours=8)).strftime('%m-%d %H:%M')
                    with st.expander(f"{t_str} | {res.get('summary', '无摘要')}"):
                        sc = res.get('scores', {})
                        st.markdown(f"""<small>🕊️ <b style='color:#2ecc71'>{sc.get('平静度')}</b> | 👁️ <b style='color:#9b59b6'>{sc.get('觉察度')}</b> | 🔋 <b style='color:#e67e22'>{sc.get('能量水平')}</b></small>""", unsafe_allow_html=True)
                        
                        # 历史记录里也顺便展示 NVC，如果存在的话
                        if 'nvc_guide' in res:
                            st.markdown(f"""<div style="background-color:#f3e5f5; padding:8px; border-radius:5px; margin: 5px 0; font-size: 13px; color: #6a1b9a;">🦒 <b>转化:</b> {res['nvc_guide'].get('empathy_response')}</div>""", unsafe_allow_html=True)
                            
                        st.info(f"建议: {res.get('recommendations', {}).get('身心灵调适建议')}")
                except: continue
        else: st.info("暂无有效数据")
    else: st.info("暂无数据")
