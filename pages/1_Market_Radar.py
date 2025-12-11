import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
if weighted_senti > 0.1:
    senti_label = "Bullish"
elif weighted_senti < -0.1:
    senti_label = "Bearish"
else:
    senti_label = "Neutral"

# Top keyword by article count
if "keyword" in scores.columns:
    unique_keywords = (
        scores["keyword"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # Join all keywords into one display string
    key_themes_display = ", ".join(sorted(unique_keywords))
else:
    key_themes_display = "N/A"

col1, col2, col3 = st.columns(3)

col1.metric("Total Articles", total_articles)
col2.metric("Market Sentiment", f"{weighted_senti:+.2f}", senti_label)
col3.metric("Key Themes", key_themes_display)

# -------------------------------------------------------
# B. Alpha Matrix: Importance vs Sentiment
# -------------------------------------------------------

st.markdown("### B. Alpha Matrix (Core Signals)")

# Merge pickup_count from alpha.csv if available
if has_alpha:
    scores = scores.merge(
        alpha[["title", "pickup_count"]],
        on="title",
        how="left",
        suffixes=("", "_alpha")
    )

# Prepare marker size
if "pickup_count" in scores.columns and scores["pickup_count"].fillna(0).max() > 0:
    pc = scores["pickup_count"].fillna(0)
    scores["marker_size"] = 10 + 20 * (pc - pc.min()) / (pc.max() - pc.min())
else:
    scores["marker_size"] = 14  # constant size if no pickup_count

fig_scatter = px.scatter(
    scores,
    x="final_score",
    y="sentiment_score",
    color="keyword",
    size="marker_size",
    hover_data=["title", "final_score", "sentiment_score", "url"],
    title="Core Signals: Impact vs Market Sentiment"
)

# Horizontal line at sentiment = 0
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")

# Vertical line at median final_score (can be changed to a fixed threshold)
fig_scatter.add_vline(
    x=scores["final_score"].median(),
    line_dash="dash",
    line_color="gray"
)

st.plotly_chart(fig_scatter, use_container_width=True)


# -------------------------------------------------------
# C. Word Cloud per Keyword (color matched to scatter)
# -------------------------------------------------------

st.markdown("### C. Keyword Word Cloud (per theme)")

# Load today's word_count file
word_count_file = os.path.join(data_path, "word_count.csv")
wc = pd.read_csv(word_count_file)   # expected columns: keyword, word, count

# -------- Step 1: Extract keyword-color mapping from Scatter --------
color_map = {}
for trace in fig_scatter.data:
    kw = trace.name
    # Plotly sometimes stores category colors as arrays, sometimes as strings
    if hasattr(trace.marker, "color"):
        color_map[kw] = trace.marker.color
    else:
        color_map[kw] = "#888888"  # fallback

# -------- Step 2: Build word clouds per keyword --------
unique_keywords = wc["keyword"].unique()

# Layout: 3 images per row
cols_per_row = 3

for i in range(0, len(unique_keywords), cols_per_row):
    row_keywords = unique_keywords[i:i + cols_per_row]
    cols = st.columns(len(row_keywords))

    for col, kw in zip(cols, row_keywords):
        subset = wc[wc["keyword"] == kw]

        # Build frequency dict: {word: count}
        freq = dict(zip(subset["word"], subset["count"]))

        # Custom color function for this keyword
        def color_func(word, *args, **kwargs):
            return color_map.get(kw, "#999999")

        # Generate word cloud
        wc_img = WordCloud(
            width=500,
            height=350,
            background_color="white",
            collocations=False
        ).generate_from_frequencies(freq)

        wc_img = wc_img.recolor(color_func=color_func)

        # Plot into Streamlit
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wc_img, interpolation='bilinear')
        ax.axis("off")
        col.markdown(f"**{kw.capitalize()}**")
        col.pyplot(fig, use_container_width=True, key=f"wc_{kw}")


