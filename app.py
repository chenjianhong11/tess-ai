import streamlit as st
import numpy as np
import pandas as pd
import time
from openai import OpenAI

st.set_page_config(page_title="天文专家问答平台", layout="wide")

# 1. 密码锁 (给所有人用的访问码)
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 欢迎访问天文问答平台")
        st.markdown("请输入访问码（默认密码：**123456**）")
        pwd = st.text_input("访问码", type="password")
        if st.button("进入平台"):
            if pwd == "123456":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("密码错误！")
        return False
    return True

if check_password():
    st.title("🌌 TESS 食双星天文专家问答与诊断")
    
    # 🔑 这里填你申请的智谱 API Key！必须填在这里，所有人才能用！
    # 把下面引号里的汉字换成你的真Key，比如 "ebe2cfe9..."
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是天文专家。你可以问我任何天文知识，或者直接问我：“TESS短周期食双星周期判定的可靠性评估与误判诊断” 或 “不同去趋势方法对TESS食双星光变参数测量的影响”。"}]
    if "depth_ratio" not in st.session_state:
        st.session_state.depth_ratio = 0.5
    if "window_hours" not in st.session_state:
        st.session_state.window_hours = 3.0

    # 2. 接入大模型回答任何问题
    def get_ai_response(prompt):
        try:
            client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
            sys_prompt = """
            你是一个资深天文学家，精通所有天文学知识。
            如果用户问关于TESS双星，请专业解释半周期误判和去趋势的影响。
            请用专业、易懂的语言回答用户的所有问题。
            """
            res = client.chat.completions.create(
                model="glm-4-flash", 
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                temperature=0.3
            )
            reply_content = res.choices[0].message.content
            
            # 智能联动图表参数
            new_depth = st.session_state.depth_ratio
            new_window = st.session_state.window_hours
            if "周期" in prompt or "误判" in prompt: new_depth = 0.95
            if "去趋势" in prompt or "窗口" in prompt: new_window = 1.0
                
            return reply_content, new_depth, new_window
        except Exception as e:
            return f"API 报错: {e}", st.session_state.depth_ratio, st.session_state.window_hours

    # 3. 页面排版
    col_chat, col_plot = st.columns([1, 1.2])

    with col_chat:
        st.subheader("💬 专家问答")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("输入你的天文问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("AI 正在查阅文献..."):
                    reply, new_depth, new_window = get_ai_response(prompt)
                    st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.depth_ratio = new_depth
            st.session_state.window_hours = new_window
            st.rerun()

    # 右侧图表区
    with col_plot:
        st.subheader("📊 实时光变数据与诊断")
        depth_ratio = st.session_state.depth_ratio
        window_hours = st.session_state.window_hours
        time_arr = np.linspace(0, 10, 1000) 
        phase = (time_arr % 0.4) / 0.4
        flux = np.ones_like(time_arr)
        flux[(phase < 0.05) | (phase > 0.95)] -= 0.1 
        flux[np.abs(phase - 0.5) < 0.05] -= 0.1 * depth_ratio 
        flux += np.random.normal(0, 0.002, len(time_arr)) 
        window = max(1, int(window_hours * 20)) 
        trend = pd.Series(flux).rolling(window=window, center=True, min_periods=1).mean().values
        flux_detrended = flux - trend + 1.0
        mask = time_arr < 2
        df = pd.DataFrame({"原始观测": flux[mask], "去趋势后": flux_detrended[mask]}, index=time_arr[mask])
        st.line_chart(df)

        if depth_ratio > 0.8:
            st.error("⚠️ **周期判定风险：高！** 主次食深度接近，极易发生半周期误判。")
        else:
            st.success("✅ **周期判定风险：低。**")
        if window_hours < 2.0:
            st.error("⚠️ **去趋势影响：严重失真！** 窗口太小，真实信号被扣除。")
        else:
            st.info("ℹ️ **去趋势影响：可控。**")
