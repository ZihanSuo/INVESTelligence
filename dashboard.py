import streamlit as st
import pandas as pd
import os

# --- 1. Page Setup ---
st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

# --- 2. Find Latest Data ---
# Get current folder path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, 'data')

# Check if data folder exists
if not os.path.exists(data_root):
    st.error("Data folder not found!")
    st.stop()

# Get all date subfolders inside 'data'
subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]

if not subfolders:
    st.warning("No data folders found.")
    st.stop()

# Sort by name to get the latest date (e.g., 2025-11-30)
latest_folder = sorted(subfolders)[-1]
date_label = os.path.basename(latest_folder)

st.success(f"📅 Displaying data for: **{date_label}**")

# Define file paths
scores_file = os.path.join(latest_folder, 'scores.csv')
words_file = os.path.join(latest_folder, 'word_count.csv')

# --- 3. Display Scores (News Analysis) ---
if os.path.exists(scores_file):
    df_scores = pd.read_csv(scores_file)

    # Layout: Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(df_scores))
    
    # Calculate average sentiment if column exists
    if 'sentiment_polarization_score' in df_scores.columns:
        avg_score = df_scores['sentiment_polarization_score'].mean()
        col2.metric("Avg Sentiment", f"{avg_score:.2f}")
    
    st.divider()

    # Layout: Charts and Insights
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📈 Sentiment Chart")
        if 'sentiment_polarization_score' in df_scores.columns:
            # Use title as index for the chart
            chart_data = df_scores.set_index('title')['sentiment_polarization_score']
            st.bar_chart(chart_data)

    with c2:
        st.subheader("💡 AI Insights")
        # Show reasons if available
        if 'sentiment_polarization_reason' in df_scores.columns:
            for reason in df_scores['sentiment_polarization_reason'].dropna().head(3):
                st.info(reason)

    # Show Data Table
    with st.expander("View Raw Data"):
        st.dataframe(df_scores)

else:
    st.warning("scores.csv not found.")

# --- 4. Display Word Cloud (Keywords) ---
if os.path.exists(words_file):
    st.divider()
    st.subheader("☁️ Trending Keywords")
    df_words = pd.read_csv(words_file)
    
    # Simple bar chart for keywords
    if not df_words.empty:
        # Assuming first column is word, second is count
        st.bar_chart(df_words.set_index(df_words.columns[0]))
