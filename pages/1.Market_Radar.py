import streamlit as st
import pandas as pd
import os
import glob

# 1. Page Configuration
st.set_page_config(page_title="AI Market Radar", layout="wide")

# 2. Date Selection System (Handling the new folder structure)
def get_available_dates():
    # Looking for folders inside 'data' directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    # Get all subdirectories that look like dates
    dates = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # Sort descending (newest first)
    dates.sort(reverse=True)
    return dates

# Sidebar for navigation
with st.sidebar:
    st.header("🗄️ Archive Navigation")
    available_dates = get_available_dates()
    
    if available_dates:
        selected_date = st.selectbox("Select Analysis Date", available_dates)
        # Construct dynamic path: data/2025-12-10/
        current_path = os.path.join("data", selected_date)
    else:
        st.error("No data folders found in /data/")
        st.stop()

# 3. Load Data Function (Updated for new CSVs)
@st.cache_data
def load_daily_data(folder_path):
    # Defining paths to specific files in the date folder
    # Assuming 'alpha.csv' contains the top ranked news
    news_path = os.path.join(folder_path, 'alpha.csv')
    # Assuming 'sentiment_statistics.csv' contains aggregate metrics
    metrics_path = os.path.join(folder_path, 'sentiment_statistics.csv')
    
    news_df = None
    metrics_df = None

    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path)
    
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        
    return news_df, metrics_df

# Load data for the selected date
news_df, metrics_df = load_daily_data(current_path)

# 4. Main Dashboard Layout
st.title(f"AI Market Radar | {selected_date}")

if news_df is not None:
    
    # --- SECTION A: KPI METRICS ---
    # Try to extract metrics from sentiment_statistics.csv or calculate from news_df
    st.markdown("### 1. Daily Market Pulse")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    # KPI 1: Total Volume
    total_articles = len(news_df)
    kpi1.metric("Total Alpha Signals", total_articles)
    
    # KPI 2 & 3: Sentiment (If metrics file exists, use it, else calculate)
    avg_score = 0
    if metrics_df is not None and not metrics_df.empty:
        # Assuming column 0 is the label and column 1 is value, adjust as needed
        # Example logic:
        avg_score = metrics_df.iloc[0, 1] if len(metrics_df.columns) > 1 else 0
    elif 'score' in news_df.columns:
        avg_score = news_df['score'].mean()

    kpi2.metric("Market Sentiment Score", f"{avg_score:.2f}")
    kpi3.metric("Data Freshness", "Verified") # Placeholder or check timestamp

    st.divider()

    # --- SECTION B: IMPACT ANALYSIS (Removed Source Stats) ---
    st.markdown("### 2. Impact Sector Analysis")
    
    # If you have an 'industry' or 'category' column, we visualize that
    # Mapping likely columns based on common naming conventions
    cat_col = next((col for col in ['category', 'industry', 'sector', 'tag'] if col in news_df.columns), None)
    
    if cat_col:
        # Use full width since we removed the Source Pie Chart
        impact_counts = news_df[cat_col].value_counts()
        st.bar_chart(impact_counts)
    else:
        st.info("No category/industry column found for visualization.")

    st.divider()

    # --- SECTION C: RANKED NEWS TABLE ---
    st.markdown("### 3. Alpha News List")
    
    # select columns to display (adjust based on your actual CSV headers)
    # Trying to be smart about picking columns that likely exist
    all_cols = news_df.columns.tolist()
    target_cols = ['title', 'summary', 'score', 'url', 'reasoning']
    display_cols = [c for c in target_cols if c in all_cols]
    
    # If specific columns aren't found, just show the first 5 columns
    if not display_cols:
        display_cols = all_cols[:5]

    st.dataframe(
        news_df[display_cols].head(15),
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Source Link")
        }
    )

    # --- SECTION D: AI EXPLAINABILITY ---
    # Only show if a reasoning column exists
    reason_col = next((col for col in ['reason_trace', 'reasoning', 'explanation'] if col in news_df.columns), None)
    
    if reason_col:
        st.markdown("### 4. AI Deep Dive")
        with st.expander("View Top Analysis Logic", expanded=True):
            top_news = news_df.head(3)
            for i, row in top_news.iterrows():
                st.markdown(f"**Signal:** {row.get('title', 'N/A')}")
                st.info(f"**Logic:** {row[reason_col]}")

else:
    st.warning(f"No 'alpha.csv' found in {current_path}. Please check the data generation pipeline.")
