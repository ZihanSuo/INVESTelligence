import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Financial News Dashboard", layout="wide")

# 2. Load Data Function
# Using caching to optimize performance during re-runs
@st.cache_data
def load_data():
    # Define paths based on project structure
    metrics_path = '/mnt/data/metrics.json'
    csv_path = '/mnt/data/ranked_news.csv'
    
    # Mocking data loading for local testing if file is missing
    # In production, ensure these paths exist
    if not os.path.exists(metrics_path):
        return None, None

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    news_df = pd.read_csv(csv_path)
    return metrics_data, news_df

# Load the data
metrics, df = load_data()

# 3. Dashboard Layout
st.title("AI Financial News Agent | Daily Briefing")

if metrics and df is not None:
    
    # --- TIER 1: KPI OVERVIEW ---
    st.markdown("### 1. Market Snapshot")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric(
            label="Total Articles Scanned",
            value=metrics.get('total_articles', 0)
        )
        
    with kpi2:
        avg_align = metrics.get('avg_alignment', 0)
        st.metric(
            label="Avg Semantic Alignment",
            value=f"{avg_align:.2f}"
        )
        
    with kpi3:
        sent_score = metrics.get('sentiment_score', 0)
        st.metric(
            label="Aggregate Sentiment",
            value=f"{sent_score:+.2f}",
            delta="Bullish" if sent_score > 0 else "Bearish"
        )
    
    st.divider()

    # --- TIER 2: VISUAL ANALYTICS ---
    st.markdown("### 2. Impact & Source Analysis")
    col_charts_1, col_charts_2 = st.columns([2, 1])
    
    with col_charts_1:
        st.subheader("Impact Dimension Distribution")
        # Convert dictionary to DataFrame for charting
        impact_data = metrics.get('impact_distribution', {})
        impact_df = pd.DataFrame(list(impact_data.items()), columns=['Category', 'Count'])
        
        # Create Bar Chart
        st.bar_chart(impact_df.set_index('Category'))

    with col_charts_2:
        st.subheader("Source Distribution")
        # Simple processing to get source counts from the main dataframe
        if 'source' in df.columns:
            source_counts = df['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            
            # Using Plotly for a better Pie chart than st.pyplot
            fig = px.pie(source_counts, values='Count', names='Source', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- TIER 3: DETAILED NEWS TABLE ---
    st.markdown("### 3. Top Ranked News")
    # Display specific columns as per requirements
    display_cols = ['title', 'source', 'score', 'impact_dimension', 'url']
    # Ensure columns exist before displaying
    valid_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[valid_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

    # --- TIER 4: AI REFLECTION (EXPLAINABILITY) ---
    st.markdown("### 4. AI Reasoning Trace")
    with st.expander("View AI Analysis Logic (Why these articles?)", expanded=True):
        # iterate through the top 3 rows to show reasoning
        if 'reason_trace' in df.columns:
            top_reasons = df[['title', 'reason_trace']].dropna().head(3)
            for index, row in top_reasons.iterrows():
                st.markdown(f"**Article:** {row['title']}")
                st.info(f"**AI Reasoning:** {row['reason_trace']}")
                st.write("---")
else:
    st.error("Data files not found in /mnt/data/. Please run the n8n workflow first.")
