import streamlit as st
import openai
import json
import datetime
import pandas as pd
from supabase import create_client

# ================= 1. 核心 Prompt (保持不变，严格标准) =================
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

# ================= 4. 全新视觉组件：纵向霍金森能量柱 =================

def render_vertical_gauge(label, score, icon):
    """
    渲染纵向能量柱 (Hawkins Style)
    Score: -5 到 +5
    """
    # 映射逻辑：把 -5到+5 映射到 0%到100% 的高度
    # -5 => 0%, 0 => 50%, +5 => 100%
    percent = (score + 5) * 10
    
    # 颜色逻辑 (参考霍金森能量表色谱)
    # 低频(负分): 红/橙/褐
    # 中频(0分): 灰/蓝
    # 高频(正分): 亮绿/青/紫/金
    
    if score <= -3:
        # 羞愧/内疚/冷漠区
        color = "linear-gradient(to top, #8B0000, #FF4500)" 
        text_color = "#FF4500"
    elif -3 < score < 0:
        # 恐惧/欲望/愤怒区
        color = "linear-gradient(to top, #FF8C00, #FFD700)"
        text_color = "#E67E22"
    elif score == 0:
        # 中性
        color = "#BDC3C7"
        text_color = "#7F8C8D"
    elif 0 < score <= 3:
        # 宽容/理智区
        color = "linear-gradient(to top, #3498DB, #2ECC71)"
        text_color = "#2ECC71"
    else:
        # 爱/喜悦/开悟区
        color = "linear-gradient(to top, #9B59B6, #00FFFF)"
        text_color = "#9B59B6"

    # 纵向柱状图 HTML
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%;">
        <!-- 分数值 -->
        <div style="font-size: 24px; font-weight: 800; color: {text_color}; margin-bottom: 8px; font-family: sans-serif;">
            {score}
        </div>
        
        <!-- 能量槽容器 -->
        <div style="
            height: 160px; 
            width: 40px; 
            background-color: #f0f2f6; 
            border-radius: 20px; 
            position: relative; 
            overflow: hidden;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
        ">
            <!-- 动态能量液 -->
            <div style="
                position: absolute; 
                bottom: 0; 
                left: 0; 
                width: 100%; 
                height: {percent}%; 
                background: {color}; 
                border-radius: 0 0 20px 20px;
                transition: height 1s cubic-bezier(0.25, 0.8, 0.25, 1);
            "></div>
            
            <!-- 刻度线 (装饰) -->
            <div style="position: absolute; bottom: 50%; width: 100%; height: 1px; background: rgba(255,255,255,0.5);"></div>
        </div>
        
        <!-- 标签 -->
        <div style="margin-top: 12px; font-weight: 600; color: #555; font-size: 14px; text-align: center;">
            {icon}<br>{label}
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

# ================= 5. 前端页面主逻辑 =================
st.set_page_config(page_title="Mind Assets", page_icon="🦁", layout="centered")

# CSS 优化
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
# st.caption("将每一次心跳，量化为可增值的心灵财富") 

# 1. 修改文案：Tab 名称
tab1, tab2 = st.tabs(["📝 觉察录入", "📊 趋势大盘"])

# --- Tab 1: 录入 ---
with tab1:
    st.write("")
    # 2. 修改文案：Label
    user_input = st.text_area("记录当下的感受...", height=120, placeholder="在此输入你的觉察记录...")
    
    if st.button("⚡️ 铸造情绪资产", type="primary"):
        if not user_input or not api_key:
            st.toast("⚠️ 请输入内容或检查 Key")
        else:
            with st.spinner("🤖 AI 正在进行霍金森能量层级分析..."):
                result = analyze_emotion(user_input, api_key)
                
                if "error" in result:
                    st.error(f"系统故障: {result['error']}")
                else:
                    save_to_db(st.session_state.user_id, user_input, result)
                    st.toast("✅ 觉察已记录")
                    
                    # === 结果展示 ===
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:15px; border-radius:8px; margin-bottom: 25px; color: #444; line-height: 1.6;">
                        {result.get('summary')}
                    </div>
                    """, unsafe_allow_html=True)

                    # 3. 核心视觉：三列布局 + 纵向能量柱
                    st.markdown("### 📊 能量层级 (Energy Levels)")
                    col1, col2, col3 = st.columns(3)
                    
                    sc = result.get("scores", {})
                    
                    with col1:
                        render_vertical_gauge("平静度", sc.get("平静度", 0), "🕊️")
                    with col2:
                        render_vertical_gauge("觉察度", sc.get("觉察度", 0), "👁️")
                    with col3:
                        render_vertical_gauge("能量值", sc.get("能量水平", 0), "🔋")

                    st.write("---")
                    
                    # 洞察与建议
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
    st.subheader("📈 能量走势")
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
