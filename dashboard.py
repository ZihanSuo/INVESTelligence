import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Page Setup 
st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

# Find Latest Data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, 'data')

if not os.path.exists(data_root):
    st.error("Data folder not found!")
    st.stop()

subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]

if not subfolders:
    st.warning("No data folders found.")
    st.stop()

latest_folder = sorted(subfolders)[-1]
date_label = os.path.basename(latest_folder)

st.success(f"📅 Displaying data for: **{date_label}**")

# File paths
scores_file = os.path.join(latest_folder, 'scores.csv')
words_file = os.path.join(latest_folder, 'word_count.csv')


# ============================================================
#   PART A — MIXED SCATTER PLOT (TOP 10 ARTICLES PER KEYWORD)
# ============================================================

def plot_mixed_scatter(df_scores):

    required_cols = ['keyword', 'title', 'final_score', 'sentiment_score', 'url']
    for col in required_cols:
        if col not in df_scores.columns:
            st.error(f"Missing column in scores.csv: {col}")
            return None

    df_top10 = (
        df_scores.sort_values("final_score", ascending=False)
        .groupby("keyword")
        .head(10)
    )

    df_top10["hover_text"] = (
        "<b>" + df_top10["title"] + "</b><br>" +
        "Final Score: " + df_top10["final_score"].astype(str) + "<br>" +
        "Sentiment: " + df_top10["sentiment_score"].astype(str) + "<br>" +
        "<a href='" + df_top10["url"] + "'>Open Source</a>"
    )

    fig = px.scatter(
        df_top10,
        x="final_score",
        y="sentiment_score",
        color="keyword",
        hover_name="title",
        custom_data=["hover_text"],
        size_max=12,
        opacity=0.85,
        height=600
    )

    fig.update_traces(hovertemplate="%{customdata[0]}")

    fig.update_layout(
        title="Mixed Scatter Plot (Top 10 Articles per Keyword)",
        xaxis_title="Final Score",
        yaxis_title="Sentiment Score",
        legend_title="Keyword"
    )

    return fig



# Load scores.csv and display section
if os.path.exists(scores_file):
    df_scores = pd.read_csv(scores_file)

    st.header("📌 Article Score Insights")
    st.write(f"Total Articles: **{len(df_scores)}**")

    fig = plot_mixed_scatter(df_scores)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Scores Data"):
        st.dataframe(df_scores)

else:
    st.warning("scores.csv not found.")

# ============================
# PART B — Sentiment Distribution (100% stacked bar)
# ============================

import matplotlib.pyplot as plt
import numpy as np

sentiment_file = os.path.join(latest_folder, "sentiment_statistics.csv")

if os.path.exists(sentiment_file):
    st.header("📊 Sentiment Distribution by Keyword (100% Stacked)")

    df_sent = pd.read_csv(sentiment_file)

    required_cols = [
        "keyword",
        "strong_neg", "weak_neg",
        "neutral",
        "weak_pos", "strong_pos"
    ]
    for col in required_cols:
        if col not in df_sent.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    df_plot = df_sent.copy()
    df_plot = df_plot.sort_values("keyword")

    categories = [
        "strong_neg", "weak_neg",
        "neutral",
        "weak_pos", "strong_pos"
    ]

    x = np.arange(len(df_plot["keyword"]))
    bottom = np.zeros(len(df_plot))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot layers
    for cat in categories:
        values = df_plot[cat].astype(float)
        ax.bar(x, values, bottom=bottom, label=cat)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["keyword"], rotation=30, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment Distribution Across Keywords")

    ax.legend(loc="upper right", fontsize=8)

    st.pyplot(fig)

else:
    st.warning("sentiment_dist.csv not found.")



# ============================
# PART C — WORD CLOUD (2 per row with captions)
# ============================

from wordcloud import WordCloud
import matplotlib.pyplot as plt

if os.path.exists(words_file):
    st.header("☁️ Word Cloud")

    df_words = pd.read_csv(words_file)
    required_cols = ["keyword", "word", "count"]
    for col in required_cols:
        if col not in df_words.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    keywords = df_words["keyword"].unique().tolist()

    # Build wordclouds for each keyword
    wc_images = []
    for kw in keywords:
        df_kw = df_words[df_words["keyword"] == kw]
        freq_dict = dict(zip(df_kw["word"], df_kw["count"]))

        wc = WordCloud(
            width=500,
            height=300,
            background_color="white",
            max_words=150
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        wc_images.append((kw, fig))

    # Display 3 per row
    for i in range(0, len(wc_images), 3):
        cols = st.columns(3)

        for j in range(3):
            if i + j < len(wc_images):
                kw, fig = wc_images[i + j]
                with cols[j]:
                    st.pyplot(fig)
                    st.markdown(f"<div style='text-align:center; color:#555;'>{kw}</div>", unsafe_allow_html=True)

else:
    st.warning("word_count.csv not found.")
