import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from datetime import datetime

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Financial News Agent",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------
# Load data from today's folder
# -------------------------------------------------------

# Auto-detect today's folder
today = datetime.today().strftime("%Y-%m-%d")
data_path = f"data/2025-12-10"  #####################################################3

# Load main files
scores_file = os.path.join(data_path, "scores.csv")
alpha_file = os.path.join(data_path, "alpha.csv")

# Read scores.csv
scores = pd.read_csv(scores_file)

# Check if alpha.csv exists (for pickup_count)
has_alpha = os.path.exists(alpha_file)
if has_alpha:
    alpha = pd.read_csv(alpha_file)
else:
    alpha = None



# -------------------------------------------------------
# Prepare marker size
# If alpha.csv exists → use pickup_count
# Else → use constant marker size
# -------------------------------------------------------

# Merge alpha into scores if available
if has_alpha:
    # Only merge pickup_count and final_score/sentiment_score must match
    scores = scores.merge(
        alpha[["title", "pickup_count"]],
        on="title",
        how="left"
    )

# If pickup_count exists and is not all zero
if "pickup_count" in scores.columns and scores["pickup_count"].fillna(0).max() > 0:
    
    # Min-max scaling into visible range
    pc = scores["pickup_count"].fillna(0)
    scores["marker_size"] = 10 + 20 * (pc - pc.min()) / (pc.max() - pc.min())
else:
    # Fall back to constant
    scores["marker_size"] = 12

# -------------------------------------------------------
# Page title and Level 1: The Snapshot
# -------------------------------------------------------

st.title("Market Radar")

st.markdown("### A. The Snapshot (Daily Market Pulse)")

# Total Articles
total_articles = len(scores)

# Weighted market sentiment using final_score as weight
if "final_score" in scores.columns:
    weighted_senti = (
        (scores["sentiment_score"] * scores["final_score"]).sum()
        / scores["final_score"].sum()
    )
else:
    weighted_senti = scores["sentiment_score"].mean()

# Label for sentiment direction
if weighted_senti > 0.05:
    senti_label = "Bullish"
elif weighted_senti < -0.05:
    senti_label = "Bearish"
else:
    senti_label = "Neutral"

# Top keyword by article count
top_keyword = (
    scores["keyword"].value_counts().idxmax()
    if "keyword" in scores.columns
    else "N/A"
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Articles", total_articles)
col2.metric("Market Sentiment", f"{weighted_senti:+.2f}", senti_label)
col3.metric("Key Theme", top_keyword)

# -------------------------------------------------------
# Alpha Matrix Scatter Plot
# -------------------------------------------------------

st.subheader("Alpha Matrix: Importance vs Sentiment")

fig_scatter = px.scatter(
    scores,
    x="final_score",
    y="sentiment_score",
    color="keyword",
    size="marker_size",
    hover_data=["title", "final_score", "sentiment_score", "url"],
    title="Core Signals: Impact vs Market Sentiment"
)

# Optional quadrant lines for readability
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
fig_scatter.add_vline(x=scores["final_score"].median(), line_dash="dash", line_color="gray")

st.plotly_chart(fig_scatter, use_container_width=True)

