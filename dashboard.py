import streamlit as st

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="INVESTelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Custom CSS (Typography & Aesthetics)
# ==========================================
st.markdown("""
<style>
    /* 1. Import Google Fonts (Inter for a clean, modern look) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* 2. Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333333;
    }

    /* 3. Hero Section Styles */
    .main-title {
        font-size: 3.5rem; /* Large title */
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6); /* Gradient Blue Text */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 300;
        color: #64748B;
        margin-bottom: 2.5rem;
    }

    /* 4. Feature Card Styles */
    .feature-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #E2E8F0; /* Subtle border */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); /* Soft shadow */
        height: 100%;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .icon-box {
        font-size: 2rem;
        margin-bottom: 12px;
        background-color: #F1F5F9;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 8px;
        color: #0F172A;
    }
    .card-text {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
    }
    
    /* 5. Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Main Content
# ==========================================

# Hero Section
st.markdown('<div class="main-title">INVESTelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Driven Financial Surveillance & Alpha Discovery Terminal</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🗺️ System Modules")
st.caption("Select a module from the sidebar to begin.")

# Card Layout using Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="icon-box">🚀</div>
        <div class="card-title">Market Radar</div>
        <div class="card-text">
            <strong>The Daily Visualization.</strong><br>
            Visualize market sentiment, spot anomalies, and identify high-materiality events using the Alpha Quadrant.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="icon-box">🕰️</div>
        <div class="card-title">Time Machine</div>
        <div class="card-text">
            <strong>Historical Analysis.</strong><br>
            <em>(In Development)</em><br>
            Search past trends and replay market sentiment evolution for specific assets.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="icon-box">📰</div>
        <div class="card-title">Briefing Archive</div>
        <div class="card-text">
            <strong>Executive Reports.</strong><br>
            <em>(In Development)</em><br>
            Access the curated, AI-generated HTML newsletters delivered daily.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="icon-box">🧠</div>
        <div class="card-title">Methodology</div>
        <div class="card-text">
            <strong>Transparent AI.</strong><br>
            <em>(In Development)</em><br>
            Understand the scoring algorithms, source weighting, and n8n pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.info("Zihan Suo, Dec 2025**")
