import streamlit as st

########## Import Packages #############
# 1. Global Setup & Data Loading
import pandas as pd
import os
from datetime import datetime

# 2. Alpha Matrix & Visualization

import plotly.express as px

# 3. Word Cloud Generation 

from wordcloud import WordCloud
import colorsys
import random


# 4. Sentiment Statistics 

import matplotlib.pyplot as plt
import numpy as np

# 5. Entity Network

import json
from collections import defaultdict
from pyvis.network import Network
from streamlit.components.v1 import html

# 6. Trend Analysis 

import glob
import plotly.graph_objects as go
import plotly.io as pio


# -------------------------------------------------------
# Load data from today's folder
# -------------------------------------------------------


# Auto-detect today's folder
today = datetime.today().strftime("%Y-%m-%d")
data_path = f"data/{today}"

# Load main files
scores_file = os.path.join(data_path, "scores.csv")
alpha_file = os.path.join(data_path, "alpha.csv")
entities_file = os.path.join(data_path, "entities.json")
wordcount_file = os.path.join(data_path, "word_count.csv")
sentiment_file = os.path.join(data_path, "sentiment_statistics.csv")
# Read scores.csv
scores = pd.read_csv(scores_file)

# Check if alpha.csv exists (for pickup_count)
has_alpha = os.path.exists(alpha_file)
if has_alpha:
    alpha = pd.read_csv(alpha_file)
else:
    alpha = None


# -------------------------------------------------------
# 1. Snapshot
# -------------------------------------------------------

st.title("Market Radar")

st.markdown("### 1. Daily Market Pulse")

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
# 2. Alpha Matrix: Core signals (graph_objects 版本)
# -------------------------------------------------------
st.markdown("### 2. Alpha Matrix (Core Signals)")

# -------------------------------------------------------
# 2.1 Impact vs Market Sentiment (graph_objects)
# -------------------------------------------------------

st.markdown("#### 2.1 Impact vs Market Sentiment")

df_imp = scores.copy()

# ----- keyword 专属颜色 -----
unique_keywords = sorted(df_imp["keyword"].unique())
palette = px.colors.qualitative.D3  # 可扩展
color_map = {kw: palette[i % len(palette)] for i, kw in enumerate(unique_keywords)}

# ----- 固定 marker 大小 -----
marker_size = 14

# ----- Build Scatter Figure -----
fig_imp = go.Figure()

for kw in unique_keywords:
    sub = df_imp[df_imp["keyword"] == kw]

    fig_imp.add_trace(
        go.Scatter(
            x=sub["final_score"],
            y=sub["sentiment_score"],
            mode="markers",
            name=kw,
            marker=dict(
                size=marker_size,
                color=color_map[kw],
                opacity=0.85,
                line=dict(width=0.5, color="black")
            ),
            text=sub["title"],
            hovertemplate="<b>%{text}</b><br>"
                          "Impact: %{x}<br>"
                          "Sentiment: %{y}<extra></extra>",
        )
    )

# ----- 中线 -----
fig_imp.add_vline(
    x=df_imp["final_score"].median(),
    line_dash="dash", line_color="gray"
)
fig_imp.add_hline(
    y=0,
    line_dash="dash", line_color="gray"
)

# ----- Layout -----
fig_imp.update_layout(
    height=500,
    template="plotly_white",
    xaxis_title="Impact Score (final_score)",
    yaxis_title="Market Sentiment (sentiment_score)",
    legend_title="Keyword",
)

st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------
# 2.2 Alpha Quadrant — NO pickup_count version
# -------------------------------------------------------
st.markdown("#### 2.2 Alpha Quadrant: Credibility vs Materiality")

if not has_alpha:
    st.info("No alpha.csv for today.")
else:
    df_q = alpha.copy()

    # sentiment norm for color mapping
    df_q["sentiment_norm"] = df_q["sentiment_score"].clip(-1, 1)

    # quadrant thresholds
    x_mid = df_q["source_credibility"].median()
    y_mid = df_q["materiality_score"].median()

    # ----- color scale -----
    color_scale = [[0, "red"], [0.5, "white"], [1, "green"]]

    fig_q = go.Figure()

    fig_q.add_trace(
        go.Scatter(
            x=df_q["source_credibility"],
            y=df_q["materiality_score"],
            mode="markers",
            marker=dict(
                size=14,
                color=df_q["sentiment_norm"],
                colorscale=color_scale,
                showscale=True
            ),
            text=df_q["title"],
            hovertemplate="<b>%{text}</b><br>"
                          "Credibility: %{x}<br>"
                          "Materiality: %{y}<extra></extra>",
        )
    )

    # ----- quadrant lines -----
    fig_q.add_vline(x=x_mid, line_dash="dash", line_color="gray")
    fig_q.add_hline(y=y_mid, line_dash="dash", line_color="gray")

    # ----- quadrant labels -----
    fig_q.add_annotation(x=x_mid + 0.05, y=y_mid + 0.05, text="Q1: Critical Movers", showarrow=False)
    fig_q.add_annotation(x=x_mid - 0.05, y=y_mid + 0.05, text="Q2: Rumor Mill", showarrow=False)
    fig_q.add_annotation(x=x_mid - 0.05, y=y_mid - 0.05, text="Q3: Low Value", showarrow=False)
    fig_q.add_annotation(x=x_mid + 0.05, y=y_mid - 0.05, text="Q4: Market Noise", showarrow=False)

    fig_q.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Source Credibility",
        yaxis_title="Materiality Score",
    )

    st.plotly_chart(fig_q, use_container_width=True)
