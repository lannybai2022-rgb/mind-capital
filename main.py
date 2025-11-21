import streamlit as st
import openai
import json
import datetime
import pandas as pd
import traceback
import re
import altair as alt
from supabase import create_client

# ================= 1. 核心 Prompt (保持不变) =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你是一位结合了身心灵修行理论、实修、数据分析的“情绪资产管理专家”和“NVC心理咨询师”。

【任务目标】
1. 量化情绪资产（评分）。
2. 侦测注意力焦点（坐标系定位）。
3. NVC 深度转化（非暴力沟通）。

# === 模块一：情绪量化 ===
评分范围：-5(极差) ~ +5(极佳)
1. 平静度: -5(暴躁) ~ 0(安静) ~ +5(临在)
2. 觉察度: -5(无明) ~ 0(昏沉) ~ +5(全然觉知)
3. 能量水平: -5(瘫痪) ~ 0(平稳) ~ +5(充盈)

# === 模块二：注意力焦点侦测 ===
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

# ================= 5. 图表函数 (类型一致性修复版) =================

def parse_to_beijing(t_str):
    """返回 datetime 对象，而非字符串"""
    try:
        dt = pd.to_datetime(t_str)
        if dt.tzinfo is not None:
            dt = dt.tz_convert('Asia/Shanghai').tz_localize(None)
        else:
            dt = dt + pd.Timedelta(hours=8)
        return dt
    except:
        return datetime.datetime.now()

def render_smooth_trend(data_list):
    """Tab 1: 今日平滑曲线"""
    try:
        now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        today_str = now_bj.strftime('%Y-%m-%d')
        
        # 保持为 datetime 对象，不要转字符串
        start_of_day = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now_bj.replace(hour=23, minute=59, second=59, microsecond=0)

        df_list = []
        if data_list:
            for item in data_list:
                try:
                    dt = parse_to_beijing(item['created_at'])
                    if dt.strftime('%Y-%m-%d') == today_str:
                        res = item['ai_result']
                        if isinstance(res, str): res = json.loads(res)
                        df_list.append({
                            "Time": dt, # 保持对象
                            "平静度": res['scores'].get('平静度', 0)
                        })
                except: continue
        
        if not df_list:
             df = pd.DataFrame([
                 {"Time": start_of_day, "平静度": 0},
                 {"Time": end_of_day, "平静度": 0}
             ])
             df['opacity'] = 0 # 没数据透明
        else:
             df = pd.DataFrame(df_list)
             df['opacity'] = 1

        st.caption(f"🌊 今日心流 ({today_str})")
        
        chart = alt.Chart(df).mark_line(
            interpolate='monotone', 
            strokeWidth=3
        ).encode(
            x=alt.X('Time:T', scale=alt.Scale(domain=[start_of_day, end_of_day]), axis=alt.Axis(format='%H:%M', title='')),
            y=alt.Y('平静度', scale=alt.Scale(domain=[-5, 5]), title=''),
            color=alt.value('#11998e'),
            opacity=alt.value(1) if df_list else alt.value(0),
            tooltip=['Time:T', '平静度']
        ).properties(height=120)
        
        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.warning(f"图表加载中... ({str(e)})")

def render_focus_map(data_list):
    """Tab 2: 注意力地图 (类型一致性修复)"""
    try:
        now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        today_str = now_bj.strftime('%Y-%m-%d')
        
        # 保持为 datetime 对象
        start_of_day = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now_bj.replace(hour=23, minute=59, second=59, microsecond=0)
        
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
                            "Time": dt, # 保持对象
                            "Y_Val": y_map.get(time_orient, 2),
                            "Target": target_orient,
                            "Color": color_hex,
                            "Summary": res.get('summary', '')
                        })
                except: continue
        
        # 处理空数据
        if not processed_data:
            # 创建一个空的 DataFrame，但必须带有正确的列名和类型
            df = pd.DataFrame({
                'Time': pd.to_datetime([start_of_day]), # 强制时间类型
                'Y_Val': [2], 
                'Color': ['#fff']
            })
            # 标记为空，不画点，只画背景
            draw_points = False
        else:
            df = pd.DataFrame(processed_data)
            draw_points = True

        # --- 背景层 ---
        bg_data = pd.DataFrame([
            {"y_start": 2.5, "y_end": 3.5, "y_mid": 3, "color": "#F2F4F6", "label": "过去 Past"},
            {"y_start": 1.5, "y_end": 2.5, "y_mid": 2, "color": "#F3E5F5", "label": "当下 Present"},
            {"y_start": 0.5, "y_end": 1.5, "y_mid": 1, "color": "#E1F5FE", "label": "未来 Future"},
        ])
        # 关键：背景的时间范围也必须是 datetime 对象
        bg_data['x_start'] = start_of_day
        bg_data['x_end'] = end_of_day
        
        background = alt.Chart(bg_data).mark_rect(opacity=0.8).encode(
            x=alt.X('x_start:T', scale=alt.Scale(domain=[start_of_day, end_of_day]), axis=None),
            x2='x_end:T',
            y=alt.Y('y_start', scale=alt.Scale(domain=[0.5, 3.5]), axis=None),
            y2='y_end', 
            color=alt.Color('color', scale=None)
        )
        
        text_layer = alt.Chart(bg_data).mark_text(
            align='left', baseline='middle', dx=10, color='#B0BEC5', fontSize=14, fontWeight='bold'
        ).encode(
            x=alt.X('x_start:T'),
            y=alt.Y('y_mid'),
            text='label'
        )
        
        # 组合图表
        final_chart = background + text_layer
        
        # --- 只有当有真实数据时，才叠加散点层 ---
        if draw_points:
            points = alt.Chart(df).mark_circle(size=150, opacity=0.9).encode(
                x=alt.X('Time:T', scale=alt.Scale(domain=[start_of_day, end_of_day]), axis=alt.Axis(format='%H:%M', title='')),
                y=alt.Y('Y_Val', title='', axis=None),
                color=alt.Color('Color', scale=None),
                tooltip=['Time:T', 'Summary', 'Target']
            )
            final_chart = final_chart + points

        st.altair_chart(final_chart.properties(height=300).interactive(), use_container_width=True)
        st.caption("说明：🟣 紫点=关注内在 | 🟠 橙点=关注外在")
        
    except Exception as e:
        st.warning(f"地图加载中... ({str(e)})")

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
