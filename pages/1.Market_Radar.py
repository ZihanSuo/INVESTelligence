import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. 页面配置 (必须是第一行)
# ==========================================
st.set_page_config(
    page_title="Market Radar - INVESTelligence",
    page_icon="🚀",
    layout="wide", # 关键：开启宽屏
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 审美调整 (CSS 注入)
# ==========================================
# 这里我们微调一下顶部边距，让数据尽可能靠上，利用好屏幕空间
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 让指标卡片有点立体感 */
    div[data-testid="metric-container"] {
        background-color: #f9f9f9;
        border: 1px solid #e6e6e6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 数据加载逻辑 (Data Pipeline)
# ==========================================

# 🔴 开关：True = 使用假数据测试排版; False = 从 GitHub 读取
USE_MOCK_DATA = True 

@st.cache_data
def load_data_from_github(date_str):
    """
    实际逻辑：从 GitHub 读取 CSV
    """
    if USE_MOCK_DATA:
        return generate_mock_data(date_str)
        
    # TODO: 替换为你的真实 GitHub Raw URL
    # url = f"https://raw.githubusercontent.com/ZihanSuo/INVESTelligence/main/data/{date_str}/alpha.csv"
    # try:
    #     return pd.read_csv(url)
    # except:
    #     return None
    return None

def generate_mock_data(date_str):
    """生成用于测试的假数据，包含 5 个核心资产"""
    np.random.seed(int(date_str.replace("-", ""))) # 保证同一天生成的假数据一样
    
    keywords = ["Bitcoin", "Tesla", "Nvidia", "Rare Earth", "Gold"]
    data = []
    for k in keywords:
        data.append({
            "keyword": k,
            "title": f"{k} market update for {date_str}",
            "url": "https://google.com",
            # 生成一些随机分值
            "source_credibility": np.random.uniform(0.4, 1.0),
            "materiality_score": np.random.uniform(0.3, 0.9),
            "sentiment_score": np.random.uniform(-1.0, 1.0),
            "pickup_count": np.random.randint(10, 100),
            "final_score": np.random.randint(50, 95)
        })
    return pd.DataFrame(data)

# ==========================================
# 4. 侧边栏控制区
# ==========================================
with st.sidebar:
    st.header("🎛️ 这里的控制台")
    # 默认选今天
    selected_date = st.date_input("选择日期", datetime.now())
    
    st.divider()
    st.caption(f"Backend Status: {'🟢 Mock Mode' if USE_MOCK_DATA else '🟠 Live GitHub'}")

# ==========================================
# 5. 核心逻辑：加载今日 vs 昨日数据
# ==========================================
date_today_str = selected_date.strftime("%Y-%m-%d")
date_yesterday_str = (selected_date - timedelta(days=1)).strftime("%Y-%m-%d")

# 加载数据
current_df = load_data_from_github(date_today_str)
prev_df = load_data_from_github(date_yesterday_str)

# ==========================================
# 6. 页面头部渲染 (方案 B：终端风格)
# ==========================================
col_header_1, col_header_2 = st.columns([3, 1])

with col_header_1:
    st.title(f"🚀 Market Radar")
    st.caption(f"Intelligent Financial Surveillance System | Date: {date_today_str}")

with col_header_2:
    # 右上角显示数据状态
    if current_df is not None and not current_df.empty:
        st.success(f"✅ Data Synced ({len(current_df)} assets)")
    else:
        st.error("❌ No Data Found")

st.markdown("---")

# ==========================================
# 7. (预留位置) 下一步要做的东西
# ==========================================
st.info("🚧 这里的区域即将放置：Step 2 - Sentiment Ticker (Sparklines)")
st.info("🚧 这里的区域即将放置：Step 3 - Alpha Quadrant Chart")

# 临时展示一下读取到的数据，方便调试
if current_df is not None:
    with st.expander("🔍 调试：查看原始数据 (Raw Data)"):
        st.dataframe(current_df)
