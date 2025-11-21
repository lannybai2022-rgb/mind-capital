import streamlit as st
import openai
import json
import datetime
import pandas as pd
import traceback
import re
import altair as alt  # <--- 新增：用于绘制高级平滑曲线
from supabase import create_client

# ================= 1. 核心 Prompt (3.0 终极版：评分 + NVC + 注意力) =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你是一位结合了身心灵修行理论、实修、数据分析的“情绪资产管理专家”和“NVC心理咨询师”。

【任务目标】
1. 量化情绪资产（评分）。
2. 侦测注意力焦点（坐标系定位）。
3. NVC 深度转化（非暴力沟通）。

# === 模块一：情绪量化 (标准不变) ===
评分范围：-5(极差) ~ +5(极佳)
1. 平静度: -5(暴躁) ~ 0(安静) ~ +5(临在)
2. 觉察度: -5(无明) ~ 0(昏沉) ~ +5(全然觉知)
3. 能量水平: -5(瘫痪) ~ 0(平稳) ~ +5(充盈)

# === 模块二：注意力焦点侦测 (3.0 新增) ===
请分析用户当下的念头处于“时空坐标系”的哪个位置：
1. 时间维度 (Time): 
   - "Past": 纠结过去、回忆、后悔、复盘。
   - "Present": 此时此刻的身体感受、正在做的事、心流。
   - "Future": 计划、担忧未来、期待、焦虑。
2. 对象维度 (Target):
   - "Internal": 关注自我感受、身体、想法。
   - "External": 关注他人、环境、任务、客观事件。

# === 模块三：NVC 转化 ===
1. 观察：客观发生了什么（去评判）。
2. 感受：情绪关键词。
3. 需要：情绪背后未满足的渴望。
4. 共情回应：一句温暖的、基于NVC的互动回应。

# === 输出要求 ===
1. 纯净 JSON，无 Markdown，无尾部逗号。
2. 格式如下：

{
  "summary": "30字总结",
  "scores": { "平静度": 0, "觉察度": 0, "能量水平": 0 },
  "focus_analysis": {
    "time_orientation": "Past" | "Present" | "Future",
    "focus_target": "Internal" | "External"
  },
  "nvc_guide": {
    "observation": "...",
    "feeling": "...",
    "need": "...",
    "empathy_response": "..."
  },
  "key_insights": ["洞察1", "洞察2"],
  "recommendations": { "身心灵调适建议": "..." }
}
"""

# ================= 2. 数据库连接 =================
@st.cache_resource
def init_supabase():
    try:
        if "SUPABASE_URL" in st.secrets:
            return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except: return None
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

def get_history(user_id, limit=50):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return res.data
        except: return []
    return []

# ================= 3. AI 逻辑 (含数据清洗) =================
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
                {"role": "user", "content": text}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content
        return json.loads(clean_json_string(content))
    except Exception as e:
        return {"error": str(e), "raw_content": content}

# ================= 4. 视觉组件 (严格保留你满意的“刻度在右侧”版本) =================
def get_gauge_html(label, score, icon, theme="peace"):
    percent = (score + 5) * 10
    colors = {
        "peace": ["#11998e", "#38ef7d", "#11998e"],
        "awareness": ["#8E2DE2", "#4A00E0", "#6a0dad"],
        "energy": ["#f12711", "#f5af19", "#e67e22"]
    }
    c = colors.get(theme, colors["peace"])
    
    # 80px宽，数字在右侧 (left: 50px)
    return f"<div style='display: flex; flex-direction: column; align-items: center; width: 80px;'><div style='height: 160px; width: 44px; background: #f0f2f6; border-radius: 22px; position: relative; margin-top: 5px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);'><div style='position: absolute; top: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>+5</div><div style='position: absolute; top: 50%; transform: translateY(-50%); left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>0</div><div style='position: absolute; bottom: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>-5</div><div style='position: absolute; bottom: 0; width: 100%; height: {percent}%; background: linear-gradient(to top, {c[0]}, {c[1]}); border-radius: 22px; transition: height 0.8s; z-index: 1;'></div><div style='position: absolute; bottom: {percent}%; left: 50%; transform: translate(-50%, 50%); background: #fff; color: {c[2]}; font-weight: 800; font-size: 13px; padding: 3px 8px; border-radius: 10px; border: 1.5px solid {c[2]}; box-shadow: 0 3px 8px rgba(0,0,0,0.15); z-index: 10; min-width: 28px; text-align: center; line-height: 1.2;'>{score}</div></div><div style='margin-top: 10px; font-size: 13px; font-weight: 600; color: #666; text-align: center;'>{icon}<br>{label}</div></div>"

# ================= 5. 图表组件 (新增：Altair 平滑曲线 & 注意力地图) =================

def render_smooth_trend(data_list):
    """Tab 1: 今日平滑曲线"""
    if not data_list: return
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    df_list = []
    for item in data_list:
        try:
            created_at = pd.to_datetime(item['created_at']) + pd.Timedelta(hours=8)
            if created_at.strftime('%Y-%m-%d') == today_str:
                res = item['ai_result']
                if isinstance(res, str): res = json.loads(res)
                df_list.append({
                    "Time": created_at,
                    "平静度": res['scores'].get('平静度', 0)
                })
        except: continue
        
    if not df_list: return

    df = pd.DataFrame(df_list)
    chart = alt.Chart(df).mark_line(interpolate='basis', strokeWidth=3).encode(
        x=alt.X('Time', axis=alt.Axis(format='%H:%M', title='')),
        y=alt.Y('平静度', scale=alt.Scale(domain=[-5, 5])),
        color=alt.value('#11998e')
    ).properties(height=120, title="今日心流 (Today's Flow)")
    st.altair_chart(chart, use_container_width=True)

def render_focus_map(data_list):
    """Tab 2: 注意力焦点地图 (灰-紫-蓝背景)"""
    if not data_list: return
    processed_data = []
    for item in data_list:
        try:
            res = item['ai_result']
            if isinstance(res, str): res = json.loads(res)
            focus = res.get('focus_analysis', {})
            time_orient = focus.get('time_orientation', 'Present')
            target_orient = focus.get('focus_target', 'Internal')
            
            y_map = {"Past": 3, "Present": 2, "Future": 1} # 映射坐标
            
            processed_data.append({
                "Time": pd.to_datetime(item['created_at']) + pd.Timedelta(hours=8),
                "Y_Val": y_map.get(time_orient, 2),
                "Target": target_orient,
                "Color": "#FF9800" if target_orient == "External" else "#9C27B0",
                "Summary": res.get('summary', '')
            })
        except: continue
        
    if not processed_data: return
    df = pd.DataFrame(processed_data)
    
    # 背景层数据
    bg_data = pd.DataFrame([
        {"start": 2.5, "end": 3.5, "color": "#F2F4F6"}, # 过去-灰
        {"start": 1.5, "end": 2.5, "color": "#F3E5F5"}, # 当下-紫
        {"start": 0.5, "end": 1.5, "color": "#E1F5FE"}, # 未来-蓝
    ])
    
    background = alt.Chart(bg_data).mark_rect(opacity=0.8).encode(
        y=alt.Y('start', scale=alt.Scale(domain=[0.5, 3.5]), axis=None),
        y2='end', color=alt.Color('color', scale=None)
    )
    
    points = alt.Chart(df).mark_circle(size=120).encode(
        x=alt.X('Time', title='', axis=alt.Axis(format='%m-%d %H:%M')),
        y=alt.Y('Y_Val', title='', axis=alt.Axis(tickCount=3, values=[1, 2, 3], labelExpr="datum.value == 3 ? '过去' : datum.value == 2 ? '当下' : '未来'")),
        color=alt.Color('Color', scale=None),
        tooltip=['Time', 'Summary']
    )
    
    st.altair_chart((background + points).properties(height=250).interactive(), use_container_width=True)
    st.caption("🟣 紫点: 关注内在 | 🟠 橙点: 关注外在")

# ================= 6. 主程序 =================
st.set_page_config(page_title="AI情绪资产助手", page_icon="🦁", layout="centered")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; font-weight: bold; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

if "user_id" not in st.session_state: st.session_state.user_id = "guest_001"

with st.sidebar:
    st.header("⚙️ 设置")
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ 已连接")
    else:
        api_key = st.text_input("DeepSeek Key", type="password")
    st.session_state.user_id = st.text_input("账户 ID", value=st.session_state.user_id)

st.title("🦁 AI情绪资产助手")

history_data = get_history(st.session_state.user_id)

tab1, tab2 = st.tabs(["📝 觉察录入", "🗺️ 注意力地图"])

# --- Tab 1: 录入 & 即时反馈 ---
with tab1:
    # 1. 顶部展示今日心流曲线
    render_smooth_trend(history_data)
    
    st.write("")
    user_input = st.text_area("", height=100, placeholder="在此记录当下身心感受...")
    
    if st.button("⚡️ 铸造资产", type="primary"):
        if not user_input or not api_key:
            st.toast("⚠️ 请输入内容或 Key")
        else:
            with st.spinner("🧠 AI 正在侦测注意力坐标并进行 NVC 转化..."):
                result = analyze_emotion(user_input, api_key)
                
                if "error" in result:
                    st.error("系统故障")
                    with st.expander("详情"): st.code(result.get('raw_content'))
                else:
                    save_to_db(st.session_state.user_id, user_input, result)
                    st.toast("✅ 觉察已铸造")
                    st.rerun()

    if history_data:
        # 取最新的一条数据展示（如果刚铸造完，这就是新的；如果是刚进来，就是上一条）
        # 这样保证 Tab 1 永远有内容看，不会空荡荡
        latest_res = history_data[0]['ai_result']
        if isinstance(latest_res, str): latest_res = json.loads(latest_res)
        
        st.write("---")
        st.info(f"📝 最近觉察: {latest_res.get('summary')}")
        
        # 核心仪表盘 (你最满意的版本)
        sc = latest_res.get("scores", {})
        h1 = get_gauge_html("平静度", sc.get("平静度", 0), "🕊️", "peace")
        h2 = get_gauge_html("觉察度", sc.get("觉察度", 0), "👁️", "awareness")
        h3 = get_gauge_html("能量值", sc.get("能量水平", 0), "🔋", "energy")
        st.markdown(f"<div style='display: flex; justify-content: space-around; align-items: flex-end; margin: 20px 0; width: 100%;'>{h1}{h2}{h3}</div>", unsafe_allow_html=True)
        
        # NVC 转化卡片
        nvc = latest_res.get("nvc_guide", {})
        if nvc:
            st.markdown(f"""
            <div style="background-color:#f3e5f5; padding:15px; border-radius:10px; border-left: 5px solid #9c27b0; margin-bottom: 20px; color: #4a148c;">
                <p style="margin-bottom: 4px; font-size: 14px;"><b>👁️ 观察:</b> {nvc.get('observation')}</p>
                <p style="margin-bottom: 4px; font-size: 14px;"><b>❤️ 感受:</b> {nvc.get('feeling')}</p>
                <p style="margin-bottom: 4px; font-size: 14px;"><b>🌱 需要:</b> {nvc.get('need')}</p>
                <hr style="border-top: 1px dashed #ce93d8; margin: 8px 0;">
                <p style="font-style: italic; font-weight: bold;">" {nvc.get('empathy_response')} "</p>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("💡 深度洞察", expanded=True):
            for insight in latest_res.get('key_insights', []):
                st.markdown(f"**•** {insight}")

# --- Tab 2: 注意力焦点地图 ---
with tab2:
    st.subheader("🗺️ 你的注意力去了哪里？")
    if st.button("🔄 刷新"): st.rerun()
    
    render_focus_map(history_data)
    
    if history_data:
        # 这里展示最新一条的 NVC 旁白，作为地图的注解
        latest_res = history_data[0]['ai_result']
        if isinstance(latest_res, str): latest_res = json.loads(latest_res)
        nvc = latest_res.get("nvc_guide", {})
        
        st.markdown("### 🦒 AI 陪伴旁白")
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; background: #fff;">
            <p>AI 咨询师轻声对你说：<br>
            <span style="color: #6a1b9a; font-style: italic; font-weight: bold;">“{nvc.get('empathy_response', '时刻保持觉察...')}”</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("暂无数据，请先去首页记录。")
