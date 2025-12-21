import streamlit as st
import openai
import json
import datetime
import pandas as pd
import re
import altair as alt
from supabase import create_client
import html
import hashlib
import base64
import extra_streamlit_components as stx

# ================= 1. 核心 Prompt =================
STRICT_SYSTEM_PROMPT = """
【角色设定】
你是一位结合了身心灵修行理论、实修和数据分析的"情绪资产管理专家"。你的任务是接收用户输入的非结构化情绪日记，并将其转化为结构化的情绪资产数据，并提供专业的管理建议。

【情绪标签体系与评分标准】
请严格基于以下3个维度进行量化分析（分数范围：-5到+5）。你必须参考下表中的描述来判断分数：

## 平静度评分标准
| 分数 | 描述 |
| -5 | 暴躁, 心绪发狂, 躁动不安 |
| -4 | 恐慌, 恐惧 |
| -3 | 焦虑, 迷茫, 困惑 |
| -2 | 不安, 担忧 |
| -1 | 轻度不安, 心绪不宁 |
| 0 | 安静 |
| +1 | 平静, 内心平静，没有波澜 |
| +2 | 宁静, 内心一片祥和，无纷扰 |
| +3 | 安详, 内心安详，安稳 |
| +4 | 喜悦, 专注，注意力灌注，心流体验 |
| +5 | 狂喜, 意识清明，全然临在 |

## 觉察度评分标准
| 分数 | 描述 |
| -5 | 没有觉察概念，完全认同念头、情绪 |
| -4 | 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪，无法自控 |
| -3 | 没有觉察，被情绪、念头带着跑，与其无意识认同；经常陷入极端情绪 |
| -2 | 没有觉察，被情绪、念头带着跑，与其无意识认同；较多陷入极端情绪 |
| -1 | 没有觉察，被情绪、念头带着跑，与其无意识认同；偶尔陷入极端情绪 |
| 0 | 没有觉察，被情绪、念头带着跑，与其无意识认同 |
| +1 | 偶尔有觉察，反省。事后一段时间才觉察、反省到情绪、念头 |
| +2 | 较多觉察，看见自己的情绪、念头；多数是事后觉察，少有事情发生当下觉察到 |
| +3 | 很多觉察，看见自己的情绪、念头；事后觉察，和事情发生当下觉察到都有 |
| +4 | 非常多觉察，看见自己的情绪、念头；当下觉察占比更高 |
| +5 | 全然临在，对念头、情绪完全觉知，且不被其影响 |

## 能量水平评分标准
| 分数 | 描述 |
| -5 | 无法支配行动 |
| -4 | 极度累, 筋疲力尽, 提不起劲, 只想躺平 |
| -3 | 非常累 |
| -2 | 很累 |
| -1 | 累, 疲惫 |
| 0 | 没有力气，但是不累，需要注入点能量的状态 |
| +1 | 稍微有点力气 |
| +2 | 有点力气但不多 |
| +3 | 有力气，能正常应对事物 |
| +4 | 活力满满, 干劲十足 |
| +5 | 精力过剩 |

【任务要求】
1. 分析与评分：仔细阅读输入文本，根据【情绪标签体系与评分标准】对用户的情绪状态进行量化评分（-5到+5）。
2. 洞察与建议：贴合用户情境，提取核心情绪模式，可引用用户原话进行解读；并提供一条具体可操作的身心灵调适建议。
3. 风险提示：仅当用户情绪处于较激烈状态（平静度≤-3）时，温和提醒"暂缓重大决策"，并给出一个身体层面的刹车动作。
4. 注意力侦测：判断用户的注意力焦点在时空坐标系中的位置。
5. 输出格式：必须严格以JSON格式输出，不包含任何额外解释性文字。

【注意力焦点侦测】
分析用户当下的念头处于"时空坐标系"的哪个位置：

1. 时间维度 (time_orientation):
   - "Past": 纠结过去、回忆、后悔、复盘
   - "Present": 此时此刻的身体感受、正在做的事、心流
   - "Future": 计划、担忧未来、期待、焦虑

2. 对象维度 (focus_target):
   - "Internal": 关注自我感受、身体、想法
   - "External": 关注他人、环境、任务、客观事件

【JSON输出格式】
{
  "summary": "不超过30字",
  "scores": {
    "平静度": 0,
    "觉察度": 0,
    "能量水平": 0
  },
  "key_insights": [
    "洞察点1",
    "洞察点2"
  ],
  "recommendations": {
    "身心灵调适建议": "不超过50字"
  },
  "risk_alert": "仅在平静度≤-3时输出提示和刹车动作，否则为null",
  "focus_analysis": {
    "time_orientation": "Past/Present/Future",
    "focus_target": "Internal/External"
  }
}
"""

# ================= 2. 页面配置 =================
st.set_page_config(page_title="MindfulFocus AI", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; max-width: 800px; }
.stTextArea textarea { font-size: 16px !important; border-radius: 12px !important; border: 1px solid #e2e8f0 !important; background: #f8fafc !important; }
.stButton > button { border-radius: 12px !important; height: 52px !important; font-weight: 600 !important; background: linear-gradient(135deg, #14b8a6 0%, #10b981 100%) !important; border: none !important; color: white !important; }
.stButton > button:disabled { background: #cbd5e1 !important; color: #94a3b8 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: white; padding: 6px; border-radius: 14px; border: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #f0fdfa !important; color: #0d9488 !important; }
/* 隐藏侧边栏 */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
button[kind="headerNoPadding"] { display: none !important; }
.st-emotion-cache-1dp5vir { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ================= 3. Cookie 管理 =================
@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()

def get_secret_key():
    """获取加密密钥"""
    return st.secrets.get("COOKIE_SECRET", "mindfocus_default_secret_key_2024")

def generate_auth_token(username, daily_limit, days_valid=7):
    """生成加密的认证token"""
    secret = get_secret_key()
    expire_ts = int((datetime.datetime.now() + datetime.timedelta(days=days_valid)).timestamp())
    payload = f"{username}:{daily_limit}:{expire_ts}"
    signature = hashlib.sha256(f"{payload}:{secret}".encode()).hexdigest()[:16]
    token = base64.b64encode(f"{payload}:{signature}".encode()).decode()
    return token

def verify_auth_token(token):
    """验证token并返回用户信息"""
    try:
        secret = get_secret_key()
        decoded = base64.b64decode(token.encode()).decode()
        parts = decoded.rsplit(':', 1)
        if len(parts) != 2:
            return None
        payload, signature = parts
        expected_sig = hashlib.sha256(f"{payload}:{secret}".encode()).hexdigest()[:16]
        if signature != expected_sig:
            return None
        username, daily_limit, expire_ts = payload.split(':')
        if int(expire_ts) < datetime.datetime.now().timestamp():
            return None
        return {"username": username, "daily_limit": int(daily_limit)}
    except:
        return None

# ================= 4. 数据库连接 =================
@st.cache_resource
def init_supabase():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except:
        return None

# ================= 5. 用户认证系统 =================
def verify_login(username, password):
    sb = init_supabase()
    if not sb:
        return False, "数据库未连接", None
    try:
        res = sb.table("test_accounts").select("*").eq("username", username).eq("password", password).execute()
        if res.data and len(res.data) > 0:
            user = res.data[0]
            if not user.get('is_active', True):
                return False, "账号已被禁用", None
            expires = pd.to_datetime(user['expires_at'])
            if expires.tz_localize(None) < datetime.datetime.now():
                return False, "账号已过期", None
            return True, "登录成功", user
        return False, "用户名或密码错误", None
    except Exception as e:
        return False, f"验证失败: {e}", None

def get_today_usage(username):
    sb = init_supabase()
    if not sb:
        return 0
    try:
        today = datetime.date.today().isoformat()
        res = sb.table("test_accounts").select("daily_usage").eq("username", username).execute()
        if res.data and res.data[0].get('daily_usage'):
            usage = res.data[0]['daily_usage']
            return usage.get(today, 0)
        return 0
    except:
        return 0

def increment_usage(username):
    sb = init_supabase()
    if not sb:
        return
    try:
        today = datetime.date.today().isoformat()
        res = sb.table("test_accounts").select("daily_usage, total_usage").eq("username", username).execute()
        if res.data:
            usage = res.data[0].get('daily_usage') or {}
            total = res.data[0].get('total_usage') or 0
            usage[today] = usage.get(today, 0) + 1
            sb.table("test_accounts").update({
                "daily_usage": usage,
                "total_usage": total + 1
            }).eq("username", username).execute()
    except:
        pass

def check_quota(username, daily_limit):
    used = get_today_usage(username)
    return used < daily_limit, daily_limit - used, used

# ================= 6. 数据存储 =================
def save_to_db(user_id, text, json_result):
    sb = init_supabase()
    if sb:
        try:
            if isinstance(json_result, dict):
                ai_result_str = json.dumps(json_result, ensure_ascii=False)
            else:
                ai_result_str = json_result
            sb.table("emotion_logs").insert({
                "user_id": user_id, 
                "user_input": text, 
                "ai_result": ai_result_str
            }).execute()
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
            return False
    return False

def get_history(user_id, limit=200):
    sb = init_supabase()
    if sb:
        try:
            res = sb.table("emotion_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return res.data or []
        except: pass
    return []

# ================= 7. AI 逻辑 =================
def clean_json_string(s):
    if not s:
        return "{}"
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    match = re.search(r'\{[\s\S]*\}', s)
    if match:
        s = match.group()
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    s = re.sub(r':\s*\+(\d)', r': \1', s)
    return s.strip()

def analyze_emotion(text, api_key):
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": STRICT_SYSTEM_PROMPT}, {"role": "user", "content": text}],
            temperature=0.4
        )
        content = response.choices[0].message.content
        cleaned = clean_json_string(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {str(e)}", "raw": content[:500]}
    except Exception as e:
        return {"error": str(e)}

# ================= 8. 工具函数 =================
def safe_text(text):
    """安全处理文本，防止HTML注入"""
    if not text:
        return ""
    text = html.escape(str(text))
    text = re.sub(r'&lt;/?div&gt;', '', text)
    text = re.sub(r'&lt;/?p&gt;', '', text)
    text = re.sub(r'&lt;[^&]*&gt;', '', text)
    return text.strip()

def get_recommendation(result):
    """兼容新旧两种字段名"""
    recs = result.get('recommendations', {})
    if '身心灵调适建议' in recs:
        return recs['身心灵调适建议']
    if 'action_guide' in recs:
        return recs['action_guide']
    return ''

def should_show_risk_alert(scores, risk_alert):
    """判断是否需要显示风险提示"""
    peace = scores.get('平静度', 0)
    try:
        peace = int(peace)
    except:
        peace = 0
    if peace <= -3 and risk_alert and risk_alert != "null" and risk_alert.lower() != "null":
        return True
    return False

# ================= 9. UI 组件 =================
def render_header(username, daily_limit):
    used = get_today_usage(username)
    remaining = daily_limit - used
    color = "#10b981" if remaining > 10 else "#f59e0b" if remaining > 3 else "#ef4444"
    
    st.markdown(f"""
    <div style="background: white; border-bottom: 1px solid #e2e8f0; padding: 12px 20px; margin: -6rem -1rem 1rem -1rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #14b8a6 0%, #3b82f6 100%); color: white; padding: 8px; border-radius: 10px; font-size: 20px;">🧠</div>
            <span style="font-weight: 700; font-size: 18px; color: #1e293b;">MindfulFocus AI</span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #64748b;">今日剩余</div>
                <div style="font-size: 14px; font-weight: 600; color: {color};">{remaining}/{daily_limit}</div>
            </div>
            <div style="background: #f1f5f9; padding: 6px 12px; border-radius: 20px;">
                <span style="font-size: 13px; font-weight: 500; color: #475569;">👤 {safe_text(username)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_gauge_card(scores):
    """渲染温度计卡片"""
    def gauge(label, score, icon, theme):
        try:
            score = int(score)
        except:
            score = 0
        score = max(-5, min(5, score))
        
        percent = (score + 5) * 10
        colors = {"peace": ("#11998e", "#38ef7d", "#0d9488"), "awareness": ("#8E2DE2", "#4A00E0", "#7c3aed"), "energy": ("#f97316", "#fbbf24", "#ea580c")}
        c = colors.get(theme)
        badge = f"+{score}" if score > 0 else str(score)
        return f"""<div style="display: flex; flex-direction: column; align-items: center; width: 100px;">
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="position: relative; height: 140px; width: 44px; background: #f1f5f9; border-radius: 22px; overflow: hidden; border: 1px solid #e2e8f0;">
                    <div style="position: absolute; bottom: 0; width: 100%; height: {percent}%; background: linear-gradient(to top, {c[0]}, {c[1]}); opacity: 0.85;"></div>
                    <div style="position: absolute; bottom: {percent}%; left: 50%; transform: translate(-50%, 50%); background: white; color: {c[2]}; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 8px; border: 2px solid {c[2]}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">{badge}</div>
                </div>
                <div style="display: flex; flex-direction: column; justify-content: space-between; height: 140px; padding: 4px 0;">
                    <span style="font-size: 9px; color: #94a3b8;">+5</span>
                    <span style="font-size: 9px; color: #94a3b8;">0</span>
                    <span style="font-size: 9px; color: #94a3b8;">-5</span>
                </div>
            </div>
            <div style="margin-top: 12px; text-align: center;"><div style="font-size: 20px;">{icon}</div><div style="font-size: 11px; font-weight: 600; color: #64748b;">{safe_text(label)}</div></div>
        </div>"""
    
    st.markdown(f"""<div style="background: white; padding: 24px 20px; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-around; align-items: flex-end;">
            {gauge("平静度", scores.get("平静度", 0), "🕊️", "peace")}
            {gauge("觉察度", scores.get("觉察度", 0), "👁️", "awareness")}
            {gauge("能量值", scores.get("能量水平", 0), "🔋", "energy")}
        </div>
    </div>""", unsafe_allow_html=True)

def render_insights(insights, recommendation, risk_alert, scores):
    """渲染洞察和行动指南"""
    safe_insights = []
    if isinstance(insights, list):
        for i in insights:
            safe_insights.append(safe_text(i))
    
    items = "".join([f'<li style="margin-bottom: 6px; color: #581c87; font-size: 14px; line-height: 1.5;">• {i}</li>' for i in safe_insights])
    
    show_risk = should_show_risk_alert(scores, risk_alert)
    
    if not show_risk:
        action_content = f'<p style="margin: 0; color: #166534; font-size: 14px; line-height: 1.6;">{safe_text(recommendation)}</p>'
    else:
        action_content = f'''<div style="background: #fffbeb; padding: 12px 16px; border-radius: 10px; border: 1px solid #fde68a;">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                <span style="font-size: 14px;">⚠️</span>
                <span style="font-size: 13px; font-weight: 600; color: #f59e0b;">温馨提示</span>
            </div>
            <p style="margin: 0; color: #292524; font-size: 14px; line-height: 1.6;">{safe_text(risk_alert)}</p>
        </div>'''
    
    st.markdown(f"""<div style="background: #faf5ff; padding: 16px; border-radius: 16px; border: 1px solid #e9d5ff; margin-bottom: 10px;">
        <h4 style="margin: 0 0 10px; font-size: 14px; color: #7c3aed;">💡 深度洞察</h4>
        <ul style="margin: 0; padding: 0; list-style: none;">{items}</ul>
    </div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div style="background: #f0fdf4; padding: 16px; border-radius: 16px; border: 1px solid #bbf7d0; margin-bottom: 12px;">
        <h4 style="margin: 0 0 10px; font-size: 14px; color: #16a34a;">❤️ 行动指南</h4>
        {action_content}
    </div>""", unsafe_allow_html=True)

def parse_to_beijing(t_str):
    try:
        dt = pd.to_datetime(t_str)
        if dt.tzinfo: dt = dt.tz_convert('Asia/Shanghai').tz_localize(None)
        else: dt = dt + pd.Timedelta(hours=8)
        return dt
    except: return datetime.datetime.now()

def render_trend(data_list):
    """渲染情绪波动图 - 去掉框，压缩高度，更平滑曲线"""
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    today_str, start_dt, end_dt = now.strftime('%Y-%m-%d'), now.replace(hour=0, minute=0, second=0), now.replace(hour=23, minute=59, second=59)
    
    df_list = []
    for item in (data_list or []):
        try:
            dt = parse_to_beijing(item['created_at'])
            if dt.strftime('%Y-%m-%d') == today_str:
                res = item['ai_result'] if isinstance(item['ai_result'], dict) else json.loads(item['ai_result'])
                df_list.append({"Time": dt, "Score": res['scores'].get('平静度', 0)})
        except: continue
    
    # 去掉框，压缩标题行高度
    st.markdown("""<div style="padding: 8px 0 4px 0;">
        <span style="font-size: 14px; font-weight: 600; color: #334155;">🌊 情绪波动 (近24小时)</span>
    </div>""", unsafe_allow_html=True)
    
    df = pd.DataFrame(df_list) if df_list else pd.DataFrame({'Time': [start_dt, end_dt], 'Score': [0, 0]})
    
    # 使用 basis 插值让曲线更平滑
    chart = alt.Chart(df).mark_area(
        interpolate='basis',
        line={'color': '#0d9488', 'strokeWidth': 2},
        color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='rgba(20,184,166,0.3)', offset=0), alt.GradientStop(color='rgba(20,184,166,0)', offset=1)], x1=1, x2=1, y1=1, y2=0)
    ).encode(
        x=alt.X('Time:T', scale=alt.Scale(domain=[start_dt, end_dt]), axis=alt.Axis(format='%H:%M', title='')),
        y=alt.Y('Score:Q', scale=alt.Scale(domain=[-5, 5]), axis=alt.Axis(title='', values=[-5, 0, 5])),
        opacity=alt.value(1 if df_list else 0)
    ).properties(height=120).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)

def render_focus_map(data_list):
    """渲染注意力地图 - 去掉框，压缩高度"""
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    today_str, start_dt, end_dt = now.strftime('%Y-%m-%d'), now.replace(hour=0, minute=0, second=0), now.replace(hour=23, minute=59, second=59)
    
    points = []
    for item in (data_list or []):
        try:
            dt = parse_to_beijing(item['created_at'])
            if dt.strftime('%Y-%m-%d') == today_str:
                res = item['ai_result'] if isinstance(item['ai_result'], dict) else json.loads(item['ai_result'])
                focus = res.get('focus_analysis', {})
                y_map = {"Past": 1, "Present": 2, "Future": 3}
                color = "#f97316" if "external" in str(focus.get('focus_target', '')).lower() else "#8b5cf6"
                points.append({"Time": dt, "Y": y_map.get(focus.get('time_orientation', 'Present'), 2), "Color": color})
        except: continue
    
    # 去掉框，压缩标题行高度
    st.markdown("""<div style="padding: 8px 0 4px 0; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 14px; font-weight: 600; color: #334155;">🗺️ 注意力地图</span>
        <span style="font-size: 11px; color: #64748b;"><span style="color: #8b5cf6;">●</span> 内在 <span style="color: #f97316;">●</span> 外在</span>
    </div>""", unsafe_allow_html=True)
    
    df = pd.DataFrame(points) if points else pd.DataFrame({'Time': [start_dt], 'Y': [2], 'Color': ['#fff']})
    chart = alt.Chart(df).mark_circle(size=150 if points else 0, opacity=0.85).encode(
        x=alt.X('Time:T', scale=alt.Scale(domain=[start_dt, end_dt]), axis=alt.Axis(format='%H:%M', title='')),
        y=alt.Y('Y:Q', scale=alt.Scale(domain=[0.5, 3.5]), axis=alt.Axis(title='', labelExpr="datum.value==1?'Past':datum.value==2?'Present':'Future'", values=[1,2,3])),
        color=alt.Color('Color:N', scale=None)
    ).properties(height=150).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)

# ================= 10. 登录页面 =================
def render_login(cookie_manager):
    st.markdown("""<div style="text-align: center; margin-top: 60px;">
        <div style="background: linear-gradient(135deg, #14b8a6, #3b82f6); color: white; padding: 16px; border-radius: 16px; display: inline-block; margin-bottom: 20px; font-size: 32px;">🧠</div>
        <h1 style="font-size: 28px; font-weight: 700; color: #1e293b;">MindfulFocus AI</h1>
        <p style="color: #64748b;">情绪资产管理专家</p>
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            if st.form_submit_button("登 录", use_container_width=True):
                if username and password:
                    ok, msg, user = verify_login(username, password)
                    if ok:
                        # 登录成功后设置Cookie
                        token = generate_auth_token(username, user['daily_limit'])
                        cookie_manager.set("auth_token", token, expires_at=datetime.datetime.now() + datetime.timedelta(days=7))
                        
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.daily_limit = user['daily_limit']
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("请输入用户名和密码")

# ================= 11. 主程序 =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

api_key = st.secrets.get("OPENAI_API_KEY", "")

# 初始化Cookie管理器
cookie_manager = get_cookie_manager()

# 尝试从Cookie自动登录
if not st.session_state.logged_in:
    auth_token = cookie_manager.get("auth_token")
    if auth_token:
        user_info = verify_auth_token(auth_token)
        if user_info:
            st.session_state.logged_in = True
            st.session_state.username = user_info["username"]
            st.session_state.daily_limit = user_info["daily_limit"]

if not st.session_state.logged_in:
    render_login(cookie_manager)
else:
    username = st.session_state.username
    daily_limit = st.session_state.daily_limit
    
    render_header(username, daily_limit)
    history = get_history(username)
    
    tab1, tab2 = st.tabs(["✨ 情绪资产记录", "🗺️ 注意力地图"])
    
    with tab1:
        # 情绪波动图在最顶部
        render_trend(history)
        
        if history:
            latest = history[0]['ai_result']
            if isinstance(latest, str): 
                try:
                    latest = json.loads(latest)
                except:
                    latest = {}
            
            scores = latest.get('scores', {})
            render_gauge_card(scores)
            render_insights(
                latest.get('key_insights', []), 
                get_recommendation(latest),
                latest.get('risk_alert'),
                scores
            )
        
        # 去掉引导语卡片，保留文字
        st.markdown("""<div style="padding: 8px 0 4px 0;">
            <span style="font-size: 14px; font-weight: 600; color: #334155;">此刻你的感受如何？</span>
        </div>""", unsafe_allow_html=True)
        
        user_input = st.text_area("", height=120, placeholder="描述此刻的身体感受、念头或所处情境...", label_visibility="collapsed")
        
        has_quota, remaining, used = check_quota(username, daily_limit)
        
        # 按钮和加载状态
        col_btn, col_status = st.columns([1, 2])
        
        with col_btn:
            is_disabled = not has_quota or st.session_state.is_analyzing
            if st.button("提交", disabled=is_disabled, use_container_width=True):
                if not user_input:
                    st.warning("请先输入内容")
                elif not api_key:
                    st.error("API Key 未配置")
                else:
                    st.session_state.is_analyzing = True
                    st.rerun()
        
        with col_status:
            if st.session_state.is_analyzing:
                st.markdown("""<div style="display: flex; align-items: center; height: 52px; padding-left: 12px;">
                    <span style="font-size: 14px; color: #0d9488;">🧠 AI分析中...</span>
                </div>""", unsafe_allow_html=True)
        
        # 执行分析
        if st.session_state.is_analyzing and user_input:
            result = analyze_emotion(user_input, api_key)
            if "error" not in result:
                result['date'] = datetime.date.today().isoformat()
                save_to_db(username, user_input, result)
                increment_usage(username)
                st.session_state.is_analyzing = False
                st.toast("✅ 提交成功！")
                st.rerun()
            else:
                st.session_state.is_analyzing = False
                st.error(f"分析失败: {result['error']}")
        
        if not has_quota:
            st.warning(f"⚠️ 今日配额已用完 ({daily_limit}/{daily_limit})")
    
    with tab2:
        render_focus_map(history)
        
        if history:
            latest = history[0]['ai_result']
            if isinstance(latest, str): 
                try:
                    latest = json.loads(latest)
                except:
                    latest = {}
            focus = latest.get('focus_analysis', {})
            time_ori = focus.get('time_orientation', 'Present')
            target = focus.get('focus_target', 'Internal')
            
            time_labels = {"Past": "过去", "Present": "当下", "Future": "未来"}
            target_labels = {"Internal": "内在感受", "External": "外在事件"}
            
            st.markdown(f"""<div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0; margin-top: 12px;">
                <h4 style="margin: 0 0 12px; font-size: 15px; color: #334155;">🎯 最近一次注意力焦点</h4>
                <div style="display: flex; gap: 12px;">
                    <div style="flex: 1; padding: 14px; background: #f0fdf4; border-radius: 12px; text-align: center;">
                        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">时间维度</div>
                        <div style="font-size: 18px; font-weight: 600; color: #16a34a;">{time_labels.get(time_ori, time_ori)}</div>
                    </div>
                    <div style="flex: 1; padding: 14px; background: #faf5ff; border-radius: 12px; text-align: center;">
                        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">关注对象</div>
                        <div style="font-size: 18px; font-weight: 600; color: #7c3aed;">{target_labels.get(target, target)}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("暂无数据，请先在「情绪资产记录」页面记录。")
