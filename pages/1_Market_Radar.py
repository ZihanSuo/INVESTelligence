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

df = scores.copy()

# 没有 keyword 的过滤掉
df = df.dropna(subset=["keyword"])

fig = go.Figure()

# 为每个 keyword 添加一个 trace（散点）
for kw, sub in df.groupby("keyword"):
    fig.add_trace(
        go.Scatter(
            x=sub["final_score"],
            y=sub["sentiment_score"],
            mode="markers",
            name=kw,
            text=sub["title"],               # hover 显示 title
            hovertemplate="<b>%{text}</b><br>" +
                          "Impact Score: %{x:.2f}<br>" +
                          "Sentiment: %{y:.2f}<extra></extra>",
            marker=dict(size=10)
        )
    )
    ##调试
st.write("DEBUG — number of rows in scores:", len(scores))
st.write(scores.head())
st.write(scores.dtypes)

# 添加横线 sentiment = 0
fig.add_hline(y=0, line_dash="dash", line_color="gray")

# 添加竖线 final_score 的中位数
median_final = df["final_score"].median()
fig.add_vline(x=median_final, line_dash="dash", line_color="gray")

fig.update_layout(
    height=450,
    xaxis_title="Impact Score (final_score)",
    yaxis_title="Market Sentiment (sentiment_score)",
    legend_title="Keyword",
    template="simple_white"
)

st.plotly_chart(fig, use_container_width=True)





