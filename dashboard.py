import streamlit as st

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
        padding: 24px 24px 10px 24px; /* Reduced bottom padding for button integration */
        border-radius: 12px 12px 0 0; /* Rounded top only */
        border: 1px solid #E2E8F0;
        border-bottom: none; /* Merge with button area */
        height: 220px; /* Fixed height for alignment */
    }
    
    /* 5. Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
    
    /* Custom Button Styling to match cards */
    .stButton button {
        width: 100%;
        border-radius: 0 0 12px 12px;
        border: 1px solid #E2E8F0;
        border-top: none;
        background-color: #F1F5F9;
        color: #1E3A8A;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #E2E8F0;
        color: #0F172A;
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
st.caption("Select a module to initiate analysis.")

# Card Layout using Columns
col1, col2, col3 = st.columns(3)

# --- Module 1: Market Radar ---
with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 12px; background-color: #F1F5F9; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">🚀</div>
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; color: #0F172A;">Market Radar</div>
        <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
            <strong>The Daily Visualization.</strong><br>
            Visualize market sentiment, spot anomalies, and identify high-materiality events using the Alpha Quadrant.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Radar ➜", key="btn_radar"):
        st.switch_page("pages/1_Market_Radar.py")


# --- Module 2: Briefing Archive ---
with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 12px; background-color: #F1F5F9; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">📰</div>
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; color: #0F172A;">Briefing Archive</div>
        <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
            <strong>Executive Reports.</strong><br>
            Access the curated, AI-generated HTML newsletters delivered daily.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Archive ➜", key="btn_archive"):
        st.switch_page("pages/2_Briefing_Archive.py")

# --- Module 3: Methodology ---
with col3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 12px; background-color: #F1F5F9; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">🧠</div>
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; color: #0F172A;">Methodology</div>
        <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
            <strong>Transparent AI.</strong><br>
            Understand the scoring algorithms, source weighting, and n8n pipelines.
        </div>
        <a href="/3_Methodology" target="_self" style="display: inline-block; margin-top: 12px; padding: 6px 12px; background-color: #2563eb; color: white; text-decoration: none; border-radius: 6px; font-weight: 600;">Learn More ➜</a>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.info("Designed by Zihan Suo, Dec 2025")
