import streamlit as st
import openai
import json
import datetime
import pandas as pd
import traceback
import re
import altair as alt
from supabase import create_client

# ================= 1. 核心 Prompt (已更新为 prompt.txt 内容) =================
STRICT_SYSTEM_PROMPT = """
【Role Definition】
你是一位结合了身心灵修行理论、实修和数据分析的“情绪资产管理专家”。你的任务是接收用户输入的非结构化情绪日记，并将其转化为结构化的情绪资产数据，并提供专业的管理建议。

【Task Objectives】
1. 量化情绪资产（评分 -5 到 +5）：严格基于【Module 1: 情绪标签体系与评分标准】。
2. 侦测注意力焦点（时空坐标系）。
3. NVC 深度转化（非暴力沟通）。

【Module 1: 情绪标签体系与评分标准 (Strict Rubric)】
请严格基于以下3个维度进行量化分析（分数范围：-5到+5）。你必须参考下表中的描述来判断分数：

| Score | 平静度 (Calmness) | 觉察度 (Awareness) | 能量水平 (Energy) |
| :--- | :--- | :--- | :--- |
| -5 | 暴躁, 心绪发狂, 躁动不安 | 没有觉察概念，完全认同念头、情绪； | 无法支配行动 |
| -4 | 恐慌, 恐惧 | 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪，无法自控； | 极度累, 筋疲力尽, 提不起劲, 只想躺平 |
| -3 | 焦虑, 迷茫, 困惑 | 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪； | 非常累 |
| -2 | 不安, 担忧 | 没有觉察，被情绪、念头带着跑，与其无意识认同；较多陷入极端情绪； | 很累 |
| -1 | 轻度不安, 心绪不宁 | 没有觉察，被情绪、念头带着跑，与其无意识认同；偶尔陷入极端情绪； | 累, 疲惫 |
| 0 | 安静 | 没有觉察，被情绪、念头带着跑，与其无意识认同； | 没有力气，但是不累，需要注入点能量的状态； |
| +1 | 平静, 内心平静，没有波澜； | 偶尔有觉察，反省。事后一段时间才觉察、反省到情绪、念头； | 稍微有点力气 |
| +2 | 宁静, 内心平静，无纷扰； | 较多觉察，看见自己的情绪、念头；多数是事后觉察，少有事情发生当下觉察到； | 有点力气但不多 |
| +3 | 安详, 内心安详，安稳； | 很多觉察，看见自己的情绪、念头；事后觉察，和事情发生当下觉察到都有； | 有力气，能正常应对事物； |
| +4 | 喜悦, 专注，注意力灌注，心流体验； | 非常多觉察，看见自己的情绪、念头；当下觉察占比更高； | 活力满满, 干劲十足 |
| +5 | 狂喜, 意识清明，全然临在； | 全然临在，对念头、情绪完全觉知，且不被其影响； | 精力过剩 |

【Module 2: 注意力焦点侦测 (Attention Focus)】
分析用户当下的念头处于“时空坐标系”的哪个位置：
1. 时间维度 (Time): 
   - "Past": 纠结过去、回忆、后悔、复盘。
   - "Present": 此时此刻的身体感受、正在做的事、心流。
   - "Future": 计划、担忧未来、期待、焦虑。
2. 对象维度 (Target):
   - "Internal": 关注自我感受、身体、想法。
   - "External": 关注他人、环境、任务、客观事件。

【Module 3: NVC 转化 (Non-Violent Communication)】
1. 观察 (Observation)：客观发生了什么（去评判）。
2. 感受 (Feeling)：情绪关键词。
3. 需要 (Need)：情绪背后未满足的渴望。
4. 共情回应 (Empathy Response)：一句温暖的、基于NVC的互动回应。

# === 输出要求 (Output Format) ===
为了系统能正确读取数据，请务必遵守以下 JSON 格式输出，不要包含 Markdown 代码块：

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

def get_history(user_id, limit=200):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return res.data
        except: return []
    return []

# ================= 3. AI 逻辑 =================
def clean_json_string(s):
    match = re.search(r'\{[\s\S]*\}', s)
    if match: s = match.group()
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    s = re.sub(r':\s*\+', ': ', s)
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

# ================= 4. 视觉组件 =================
def get_gauge_html(label, score, icon, theme="peace"):
    percent = (score + 5) * 10
    colors = {
        "peace": ["#11998e", "#38ef7d", "#11998e"],
        "awareness": ["#8E2DE2", "#4A00E0", "#6a0dad"],
        "energy": ["#f12711", "#f5af19", "#e67e22"]
    }
    c = colors.get(theme, colors["peace"])
    
    return f"<div style='display: flex; flex-direction: column; align-items: center; width: 80px;'><div style='height: 160px; width: 44px; background: #f0f2f6; border-radius: 22px; position: relative; margin-top: 5px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);'><div style='position: absolute; top: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>+5</div><div style='position: absolute; top: 50%; transform: translateY(-50%); left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>0</div><div style='position: absolute; bottom: 4px; left: 50px; color: #bdc3c7; font-size: 10px; font-weight: bold;'>-5</div><div style='position: absolute; bottom: 0; width: 100%; height: {percent}%; background: linear-gradient(to top, {c[0]}, {c[1]}); border-radius: 22px; transition: height 0.8s; z-index: 1;'></div><div style='position: absolute; bottom: {percent}%; left: 50%; transform: translate(-50%, 50%); background: #fff; color: {c[2]}; font-weight: 800; font-size: 13px; padding: 3px 8px; border-radius: 10px; border: 1.5px solid {c[2]}; box-shadow: 0 3px 8px rgba(0,0,0,0.15); z-index: 10; min-width: 28px; text-align: center; line-height: 1.2;'>{score}</div></div><div style='margin-top: 10px; font-size: 13px; font-weight: 600; color: #666; text-align: center;'>{icon}<br>{label}</div></div>"

# ================= 5. 图表函数 (防弹版) =================

def parse_to_beijing(t_str):
    """
    返回一个 无时区 (Naive) 的北京时间 datetime 对象
    """
    try:
        dt = pd.to_datetime(t_str)
        # 如果带时区，转为北京时间并移除时区信息
        if dt.tzinfo is not None:
            dt = dt.tz_convert('Asia/Shanghai').tz_localize(None)
        else:
            # 如果不带时区，默认它是UTC，+8小时
            dt = dt + pd.Timedelta(hours=8)
        return dt
    except:
        return datetime.datetime.now()

def render_smooth_trend(data_list):
    """Tab 1: 今日平滑曲线"""
    try:
        # 获取北京时间当天的 00:00 - 23:59
        now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        today_str = now_bj.strftime('%Y-%m-%d')
        start_dt = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = now_bj.replace(hour=23, minute=59, second=59, microsecond=0)

        df_list = []
        if data_list:
            for item in data_list:
                try:
                    dt = parse_to_beijing(item['created_at'])
                    if dt.strftime('%Y-%m-%d') == today_str:
                        res = item['ai_result']
                        if isinstance(res, str): res = json.loads(res)
                        df_list.append({
                            "Time": dt, # Naive Datetime
                            "平静度": res['scores'].get('平静度', 0)
                        })
                except: continue
        
        # 构造 DataFrame
        if not df_list:
             # 空数据时，造两个虚拟点撑开坐标轴
             df = pd.DataFrame({'Time': [start_dt, end_dt], '平静度': [0, 0]})
             op_val = 0 # 隐藏线条
        else:
             df = pd.DataFrame(df_list)
             op_val = 1

        st.caption(f"🌊 今日心流 ({today_str})")
        
        chart = alt.Chart(df).mark_line(
            interpolate='monotone', 
            strokeWidth=3
        ).encode(
            x=alt.X('Time', scale=alt.Scale(domain=[start_dt, end_dt]), axis=alt.Axis(format='%H:%M', title='')),
            y=alt.Y('平静度', scale=alt.Scale(domain=[-5, 5]), title=''),
            color=alt.value('#11998e'),
            opacity=alt.value(op_val),
            tooltip=['Time', '平静度']
        ).properties(height=120)
        
        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"图表加载失败: {str(e)}")

def render_focus_map(data_list):
    """Tab 2: 注意力地图 (重构版 - 解决图层打架)"""
    try:
        now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        today_str = now_bj.strftime('%Y-%m-%d')
        start_dt = now_bj.replace(hour=0, minute=0, second=0)
        end_dt = now_bj.replace(hour=23, minute=59, second=59)
        
        processed_data = []
        if data_list:
            for item in data_list:
                try:
                    dt = parse_to_beijing(item['created_at'])
                    
                    if dt.strftime('%Y-%m-%d') == today_str:
                        res = item['ai_result']
                        if isinstance(res, str): res = json.loads(res)
                        focus = res.get('focus_analysis', {})
                        time_orient = focus.get('time_orientation', 'Present')
                        target_orient = focus.get('focus_target', 'Internal')
                        
                        y_map = {"Past": 3, "Present": 2, "Future": 1}
                        t_check = str(target_orient).strip().lower()
                        color_hex = "#FF9800" if "external" in t_check else "#9C27B0"
                        
                        processed_data.append({
                            "Time": dt,
                            "Y_Val": y_map.get(time_orient, 2),
                            "Target": target_orient,
                            "Color": color_hex,
                            "Summary": res.get('summary', '')
                        })
                except: continue
        
        if not processed_data:
            df = pd.DataFrame({'Time': [start_dt], 'Y_Val': [2], 'Color': ['#fff']})
            point_size = 0
        else:
            df = pd.DataFrame(processed_data)
            point_size = 150

        # --- 构建图表 ---
        
        # 1. 背景层 (使用简单的 Rect，不依赖数据源)
        # 我们使用一个独立的 DataFrame 来画背景，确保它不受主数据影响
        bg_df = pd.DataFrame([
            {"y_start": 2.5, "y_end": 3.5, "color": "#F2F4F6"},
            {"y_start": 1.5, "y_end": 2.5, "color": "#F3E5F5"},
            {"y_start": 0.5, "y_end": 1.5, "color": "#E1F5FE"}
        ])
        
        # 为了让背景铺满X轴，我们使用 trick：不映射X字段，而是直接覆盖
        # 但Altair需要X轴定义。所以我们把背景图层改为“Rule”或者使用 Layer 的独立 Data
        
        # 简单粗暴法：给背景数据加上今天的 Start/End
        bg_df['start_time'] = start_dt
        bg_df['end_time'] = end_dt
        
        background = alt.Chart(bg_df).mark_rect(opacity=0.8).encode(
            x=alt.X('start_time', scale=alt.Scale(domain=[start_dt, end_dt]), axis=None),
            x2='end_time',
            y=alt.Y('y_start', scale=alt.Scale(domain=[0.5, 3.5]), axis=None),
            y2='y_end',
            color=alt.Color('color', scale=None)
        )
        
        # 2. 散点层
        points = alt.Chart(df).mark_circle(size=point_size, opacity=0.9).encode(
            x=alt.X('Time', scale=alt.Scale(domain=[start_dt, end_dt]), axis=alt.Axis(format='%H:%M', title='')),
            y=alt.Y('Y_Val', title='', axis=None),
            color=alt.Color('Color', scale=None),
            tooltip=['Time', 'Summary', 'Target']
        )
        
        # 3. 文字层 (硬编码位置)
        # 这里的 X 轴使用 datum 稍微偏离起点一点点
        text_data = pd.DataFrame([
            {"y": 3, "text": "过去 Past", "time": start_dt + datetime.timedelta(minutes=30)},
            {"y": 2, "text": "当下 Present", "time": start_dt + datetime.timedelta(minutes=30)},
            {"y": 1, "text": "未来 Future", "time": start_dt + datetime.timedelta(minutes=30)}
        ])
        
        texts = alt.Chart(text_data).mark_text(
            align='left', baseline='middle', color='#B0BEC5', fontSize=14, fontWeight='bold'
        ).encode(
            x=alt.X('time'),
            y=alt.Y('y'),
            text='text'
        )

        # 组合
        final_chart = (background + texts + points).properties(height=300) # 移除 interactive 以防冲突

        st.altair_chart(final_chart, use_container_width=True)
        st.caption("说明：🟣 紫点=关注内在 | 🟠 橙点=关注外在")
        
    except Exception as e:
        st.error(f"地图渲染错误: {str(e)}")

# ================= 6. 主程序 =================
st.set_page_config(page_title="AI情绪资产助手", page_icon="🦁", layout="centered")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; font-weight: bold; }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
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

# 数据库容错
try:
    history_data = get_history(st.session_state.user_id)
except:
    history_data = []

tab1, tab2 = st.tabs(["📝 觉察录入", "🗺️ 注意力地图"])

# --- Tab 1 ---
with tab1:
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
        latest_res = history_data[0]['ai_result']
        if isinstance(latest_res, str): latest_res = json.loads(latest_res)
        
        st.write("---")
        st.info(f"📝 最近记录: {latest_res.get('summary')}")
        
        sc = latest_res.get("scores", {})
        h1 = get_gauge_html("平静度", sc.get("平静度", 0), "🕊️", "peace")
        h2 = get_gauge_html("觉察度", sc.get("觉察度", 0), "👁️", "awareness")
        h3 = get_gauge_html("能量值", sc.get("能量水平", 0), "🔋", "energy")
        st.markdown(f"<div style='display: flex; justify-content: space-around; align-items: flex-end; margin: 20px 0; width: 100%;'>{h1}{h2}{h3}</div>", unsafe_allow_html=True)
        
        insights = latest_res.get('key_insights', [])
        if insights:
            insights_html = "".join([f"<li style='margin-bottom:5px;'>{item}</li>" for item in insights])
            st.markdown(f"""
            <div style="background-color:#f3e5f5; padding:15px; border-radius:10px; border-left: 5px solid #9c27b0; margin-top: 20px; color: #4a148c;">
                <h4 style="margin-top:0; margin-bottom:10px; color: #6a1b9a; font-size:16px;">💡 深度洞察</h4>
                <ul style="margin-bottom: 0; padding-left: 20px; font-size: 14px;">
                    {insights_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div style="background-color:#eafaf1; padding:15px; border-radius:8px; border: 1px dashed #27ae60; margin-top: 15px;">
            <strong style="color:#27ae60;">💊 行动指南：</strong><br>
            {latest_res.get('recommendations', {}).get('身心灵调适建议')}
        </div>
        """, unsafe_allow_html=True)

# --- Tab 2 ---
with tab2:
    st.subheader("🗺️ 你的注意力去了哪里？")
    if st.button("🔄 刷新"): st.rerun()
    
    render_focus_map(history_data)
    
    if history_data:
        latest_nvc = history_data[0]['ai_result']
        if isinstance(latest_nvc, str): latest_nvc = json.loads(latest_nvc)
        nvc = latest_nvc.get("nvc_guide", {})
        
        st.markdown("### 🦒 AI 陪伴旁白")
        st.info("此处展示基于你 **最近一次觉察** 的深度解读：")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; background: #fff;">
            <p>AI 咨询师轻声对你说：<br>
            <span style="color: #6a1b9a; font-style: italic; font-weight: bold; font-size: 18px; line-height: 1.5;">
            “ {nvc.get('empathy_response', '保持觉察，回到当下...')} ”
            </span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("暂无数据，请先去首页记录。")
