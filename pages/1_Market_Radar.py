import streamlit as st
# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Financial News Agent",
    page_icon="📈",
    layout="wide"
)

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
# 2.1 Impact vs Market Sentiment - graph_objects 版
# -------------------------------------------------------

st.markdown("#### 2.1 Impact vs Market Sentiment")

# 创建干净副本
df_viz = scores.copy()

# 移除重复列
if df_viz.columns.duplicated().any():
    df_viz = df_viz.loc[:, ~df_viz.columns.duplicated()]

# 重置索引并清理
df_viz = df_viz.reset_index(drop=True)
df_viz = df_viz.dropna(subset=['final_score', 'sentiment_score', 'keyword'])

if len(df_viz) == 0:
    st.warning("No valid data to display")
else:
    # 准备 marker size
    if "pickup_count" in df_viz.columns:
        pc = df_viz["pickup_count"].fillna(0)
        if pc.max() > pc.min():
            df_viz["marker_size"] = 10 + 20 * (pc - pc.min()) / (pc.max() - pc.min())
        else:
            df_viz["marker_size"] = 14
    else:
        df_viz["marker_size"] = 14
    
    # 创建图表 - 使用 graph_objects
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # 为每个 keyword 添加一个 trace
    keywords = df_viz['keyword'].unique()
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
    
    for i, kw in enumerate(keywords):
        subset = df_viz[df_viz['keyword'] == kw]
        
        # 构建 hover text
        hover_text = []
        for _, row in subset.iterrows():
            text = f"<b>{row['title']}</b><br>"
            text += f"Impact: {row['final_score']:.1f}<br>"
            text += f"Sentiment: {row['sentiment_score']:.2f}<br>"
            text += f"URL: {row['url']}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=subset['final_score'],
            y=subset['sentiment_score'],
            mode='markers',
            name=kw.title(),
            marker=dict(
                size=subset['marker_size'],
                color=colors[i % len(colors)],
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # 添加参考线
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray", 
        line_width=1.5,
        annotation_text="Neutral",
        annotation_position="right"
    )
    
    fig.add_vline(
        x=df_viz['final_score'].median(),
        line_dash="dash",
        line_color="gray",
        line_width=1.5,
        annotation_text="Median",
        annotation_position="top"
    )
    
    # 更新布局
    fig.update_layout(
        title="Impact vs Market Sentiment",
        xaxis_title="Impact Score (final_score)",
        yaxis_title="Market Sentiment",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title="Keywords",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(df_viz))
    with col2:
        st.metric("Avg Impact", f"{df_viz['final_score'].mean():.1f}")
    with col3:
        st.metric("Avg Sentiment", f"{df_viz['sentiment_score'].mean():.2f}")







