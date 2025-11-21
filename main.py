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

# ================= 5. 图表函数 (背景图层修复版) =================

def parse_to_beijing(t_str):
    """时间清洗机"""
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
        now_utc = datetime.datetime.utcnow()
        now_bj = now_utc + datetime.timedelta(hours=8)
        start_of_day = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now_bj.replace(hour=23, minute=59, second=59, microsecond=0)
        today_str = now_bj.strftime('%Y-%m-%d')

        df_list = []
        if data_list:
            for item in data_list:
                try:
                    created_at = parse_to_beijing(item['created_at'])
                    if created_at.strftime('%Y-%m-%d') == today_str:
                        res = item['ai_result']
                        if isinstance(res, str): res = json.loads(res)
                        df_list.append({
                            "Time": created_at,
                            "平静度": res['scores'].get('平静度', 0)
                        })
                except: continue
