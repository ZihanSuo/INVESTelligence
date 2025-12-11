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
# C. Word Cloud (Color-matched to Scatter Plot)
# -------------------------------------------------------

st.markdown("### C. Keyword Word Cloud")

# Load word count file
word_count_file = os.path.join(data_path, "word_count.csv")
word_df = pd.read_csv(word_count_file)

# -------- Step 1: Extract color mapping from Scatter --------
color_map = {}
for trace in fig_scatter.data:
    if "marker" in trace and "color" in trace.marker:
        kw = trace.name
        color_map[kw] = trace.marker.color

# -------- Step 2: Build a single frequency dict --------
# expected columns: keyword, word, word_count
freq_dict = {}

if "word" in word_df.columns:
    # Use actual words
    for _, row in word_df.iterrows():
        key = f"{row['keyword']}_{row['word']}"
        freq_dict[key] = row["word_count"]
else:
    # If no 'word' column, use keyword repeated as 'fake words'
    for _, row in word_df.iterrows():
        freq_dict[row["keyword"]] = row["word_count"]

# -------- Step 3: Custom color function --------
def keyword_color_func(word, font_size, position, orientation, font_path, random_state):
    # Extract keyword part before '_' (if exists)
    kw = word.split("_")[0]
    return color_map.get(kw, "#999999")  # fallback gray

# -------- Step 4: Generate Word Cloud --------
wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    prefer_horizontal=0.9,
    collocations=False
).generate_from_frequencies(freq_dict)

# Apply color mapping
colored_wc = wc.recolor(color_func=keyword_color_func)

# -------- Step 5: Display in Streamlit --------
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(colored_wc, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig, use_container_width=True, key="wordcloud_chart")


