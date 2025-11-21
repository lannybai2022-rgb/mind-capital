import streamlit as st
import openai
import json
import datetime
import pandas as pd
from supabase import create_client

# ================= 1. 核心 Prompt (完全还原你的严格标准) =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你是一位结合了身心灵修行理论、实修和数据分析的“情绪资产管理专家”。

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

【任务要求】
1. 评分：仔细阅读输入文本，根据上述标准量化评分。
2. 洞察：提取核心情绪模式，提供身心灵建议。
3. 格式：必须严格以JSON格式输出。

【JSON输出格式】
{
  "summary": "30字内总结（一针见血）",
  "scores": { "平静度": 整数, "觉察度": 整数, "能量水平": 整数 },
  "key_insights": ["洞察1", "洞察2"],
  "recommendations": { "身心灵调适建议": "50字建议" }
}
"""

# ================= 2. 数据库连接层 =================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except:
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
            return True
        except Exception as e:
            st.error(f"保存失败: {str(e)}")
            return False
    return False

def get_history(user_id):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
            return res.data
        except:
            return []
    return []

# ================= 3. AI 分析逻辑 =================
def analyze_emotion(text, api_key):
    client = openai.OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com"
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": f"【输入文本】\n{text}"}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ================= 4. 高级 UI 组件 (双向进度条) =================

def render_tech_bar(label, score, icon):
    """
    渲染双向能量条：左红右绿，中间为0
    """
    width_percent = abs(score) * 10 
    
    if score > 0:
        bar_color = "linear-gradient(90deg, #00b09b 0%, #96c93d 100%)" # 绿
        position_left = "50%"
        border_radius = "0 4px 4px 0"
        value_color = "#27ae60"
        prefix = "+"
    elif score < 0:
        bar_color = "linear-gradient(90deg, #ff5f6d 0%, #ffc371 100%)" # 红
        position_left = f"{50 - width_percent}%"
        border_radius = "4px 0 0 4px"
        value_color = "#e74c3c"
        prefix = ""
    else:
        bar_color = "#ddd"
        position_left = "50%"
        width_percent = 0
        border_radius = "0"
        value_color = "#95a5a6"
        prefix = ""

    html_code = f"""
    <div style="margin-bottom: 12px; font-family: 'Source Sans Pro', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 600; font-size: 14px; color: #4a4a4a;">{icon} {label}</span>
            <span style="font-weight: 700; font-size: 16px; color: {value_color};">{prefix}{score}</span>
        </div>
        <div style="width: 100%; background-color: #f0f2f6; height: 10px; border-radius: 5px; position: relative; overflow: hidden;">
            <div style="position: absolute; left: 50%; width: 2px; height: 100%; background-color: #d1d5db; z-index: 1; opacity: 0.5;"></div>
            <div style="position: absolute; height: 100%; left: {position_left}; width: {width_percent}%; background: {bar_color}; border-radius: {border_radius}; transition: all 0.6s ease;"></div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

# ================= 5. 前端页面主逻辑 =================
st.set_page_config(page_title="Mind Assets", page_icon="🦁", layout="centered")

# CSS 注入
st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; font-weight: bold; }
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

st.title("🦁 情绪资产")
st.caption("将每一次心跳，量化为可增值的心灵财富")

tab1, tab2 = st.tabs(["⚡️ 资产铸造 (录入)", "📊 趋势大盘 (报表)"])

# --- Tab 1: 录入 ---
with tab1:
    st.write("")
    user_input = st.text_area("✍️ 记录当下的觉察...", height=120, placeholder="在此输入你的心流记录...")
    
    if st.button("⚡️ 铸造情绪资产 (Mint Assets)", type="primary"):
        if not user_input or not api_key:
            st.toast("⚠️ 请输入内容或检查 Key")
        else:
            with st.spinner("🤖 AI 正在进行深度量化审计..."):
                result = analyze_emotion(user_input, api_key)
                
                if "error" in result:
                    st.error(f"系统故障: {result['error']}")
                else:
                    save_to_db(st.session_state.user_id, user_input, result)
                    st.toast("✅ 资产已上链存证！")
                    
                    # === 结果展示 ===
                    st.markdown(f"""
                    <div style="background-color:#e8f4f8; padding:15px; border-radius:8px; border-left: 5px solid #3498db; margin-bottom: 20px;">
                        <span style="font-size:18px;">📝</span> 
                        <span style="font-weight:500; color:#2c3e50;">{result.get('summary')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.container():
                        st.markdown("### 📊 资产穿透分析")
                        sc = result.get("scores", {})
                        render_tech_bar("平静度 (Peace)", sc.get("平静度", 0), "🕊️")
                        render_tech_bar("觉察度 (Awareness)", sc.get("觉察度", 0), "👁️")
                        render_tech_bar("能量值 (Energy)", sc.get("能量水平", 0), "🔋")

                    st.write("")
                    with st.expander("💡 深度洞察 (Deep Insights)", expanded=True):
                        for insight in result.get('key_insights', []):
                            st.markdown(f"**•** {insight}")
                    
                    st.markdown(f"""
                    <div style="background-color:#eafaf1; padding:15px; border-radius:8px; border: 1px dashed #27ae60; margin-top: 10px;">
                        <strong style="color:#27ae60;">💊 行动指南：</strong><br>
                        {result.get('recommendations', {}).get('身心灵调适建议')}
                    </div>
                    """, unsafe_allow_html=True)

# --- Tab 2: 报表 ---
with tab2:
    st.subheader("📈 资产K线图")
    if st.button("🔄 刷新大盘"):
        st.rerun()
    
    data = get_history(st.session_state.user_id)
    
    if data:
        chart_data = []
        for item in data:
            res = item['ai_result']
            scores = res.get('scores', {})
            utc_time = pd.to_datetime(item['created_at'])
            bj_time = utc_time + pd.Timedelta(hours=8)
            
            chart_data.append({
                "时间": bj_time, 
                "平静度": scores.get("平静度", 0),
                "觉察度": scores.get("觉察度", 0),
                "能量": scores.get("能量水平", 0)
            })
        
        df = pd.DataFrame(chart_data)
        df = df.sort_values('时间')
        
        st.line_chart(df, x='时间', y=['平静度', '觉察度', '能量'], color=["#2ecc71", "#3498db", "#f1c40f"])
        
        st.markdown("---")
        
        for item in data:
            utc_time = pd.to_datetime(item['created_at'])
            time_str = (utc_time + pd.Timedelta(hours=8)).strftime('%m-%d %H:%M')
            summary = item['ai_result'].get('summary', '无摘要')
            
            with st.expander(f"{time_str} | {summary}"):
                sc = item['ai_result'].get('scores', {})
                st.markdown(f"""
                <small>平静: <b style='color:{'#27ae60' if sc.get('平静度',0)>0 else '#e74c3c'}'>{sc.get('平静度')}</b> | 
                觉察: <b>{sc.get('觉察度')}</b> | 
                能量: <b>{sc.get('能量水平')}</b></small>
                """, unsafe_allow_html=True)
                st.info(f"建议: {item['ai_result'].get('recommendations', {}).get('身心灵调适建议')}")
    else:
        st.info("暂无数据")
