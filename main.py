import streamlit as st
import openai
import json
import datetime
import pandas as pd
from supabase import create_client

# ================= 1. 核心 Prompt (严格的身心灵标准) =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你是一位结合了身心灵修行理论、实修和数据分析的“情绪资产管理专家”。

【情绪标签体系与评分标准】
请严格基于以下3个维度进行量化分析（分数范围：-5到+5）。

维度 | Score -5 | Score 0 | Score +5
--- | --- | --- | ---
平静度 | 暴躁, 躁动 | 安静 | 狂喜, 临在
觉察度 | 完全认同念头 | 无觉察 | 全然临在
能量水平 | 无法支配行动 | 没力气 | 精力过剩

【任务要求】
1. 评分：严格基于标准量化（-5到+5）。
2. 洞察：提取核心情绪模式，提供身心灵建议。
3. 格式：必须严格输出 JSON。

【JSON输出格式】
{
  "summary": "30字内总结",
  "scores": { "平静度": 整数, "觉察度": 整数, "能量水平": 整数 },
  "key_insights": ["洞察1", "洞察2"],
  "recommendations": { "身心灵调适建议": "50字建议" }
}
"""

# ================= 2. 数据库连接层 =================
@st.cache_resource
def init_supabase():
    try:
        # 从后台读取配置
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
            st.error(f"存库失败: {e}")
            return False
    return False

def get_history(user_id):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(30).execute()
            return res.data
        except:
            return []
    return []

# ================= 3. AI 分析逻辑 (DeepSeek 版) =================
def analyze_emotion(text, api_key):
    # 这里指定连接 DeepSeek 的服务器
    client = openai.OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com"
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 指定模型
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

# ================= 4. 前端页面 UI =================
st.set_page_config(page_title="MindCapital", page_icon="🧘", layout="mobile")

if "user_id" not in st.session_state:
    st.session_state.user_id = "guest_001"

with st.sidebar:
    st.header("⚙️ 设置")
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ DeepSeek 已连接")
    else:
        api_key = st.text_input("DeepSeek Key", type="password")
    
    st.session_state.user_id = st.text_input("当前用户ID", value=st.session_state.user_id)

st.title("🧘 情绪资产管理")
st.caption("Powered by DeepSeek AI")

tab1, tab2 = st.tabs(["📝 觉察录入", "📊 资产报表"])

# --- Tab 1: 录入 ---
with tab1:
    user_input = st.text_area("✏️ 记录当下的感受...", height=150)
    
    if st.button("提交审计", type="primary"):
        if not user_input or not api_key:
            st.warning("请检查配置")
        else:
            with st.spinner("AI 正在量化身心灵数据..."):
                result = analyze_emotion(user_input, api_key)
                
                if "error" in result:
                    st.error(f"出错: {result['error']}")
                else:
                    save_to_db(st.session_state.user_id, user_input, result)
                    st.toast("✅ 数据已保存")
                    
                    sc = result.get("scores", {})
                    c1, c2, c3 = st.columns(3)
                    c1.metric("平静度", sc.get("平静度", 0))
                    c2.metric("觉察度", sc.get("觉察度", 0))
                    c3.metric("能量", sc.get("能量水平", 0))
                    
                    st.info(result.get("summary"))
                    st.success(result.get("recommendations", {}).get("身心灵调适建议"))

# --- Tab 2: 报表 ---
with tab2:
    st.subheader("📈 能量走势")
    if st.button("🔄 刷新"): st.rerun()
    data = get_history(st.session_state.user_id)
    if data:
        chart_data = []
        for item in data:
            res = item['ai_result']
            sc = res.get('scores', {})
            chart_data.append({
                "时间": item['created_at'],
                "平静度": sc.get("平静度", 0),
                "觉察度": sc.get("觉察度", 0),
                "能量": sc.get("能量水平", 0)
            })
        df = pd.DataFrame(chart_data)
        df['时间'] = pd.to_datetime(df['时间'])
        st.line_chart(df.sort_values('时间'), x='时间', y=['平静度', '觉察度', '能量'], color=["#4CAF50", "#2196F3", "#FFC107"])
    else:
        st.info("暂无数据")
