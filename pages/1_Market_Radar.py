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


# 1. 检查 data 文件夹是否存在
if not os.path.exists('data'):
    st.error("❌ 'data' directory not found. Please ensure you have uploaded the data folder.")
    st.stop()

# 2. 找到 data 目录下所有的日期文件夹
all_subdirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
# 按日期排序，找最近的一个
all_subdirs.sort(reverse=True)

if not all_subdirs:
    st.error("❌ No data folders found in 'data/' directory.")
    st.stop()

# 3. 锁定最新文件夹
latest_date_folder = all_subdirs[0]
data_path = os.path.join("data", latest_date_folder)

# 在侧边栏显示当前使用的数据日期，方便调试
st.sidebar.success(f"📅 Data Date: {latest_date_folder}")

# 4. Load main files
scores_file = os.path.join(data_path, "scores.csv")
alpha_file = os.path.join(data_path, "alpha.csv")
entities_file = os.path.join(data_path, "entities.json")
wordcount_file = os.path.join(data_path, "word_count.csv")
sentiment_file = os.path.join(data_path, "sentiment_statistics.csv") # 确保你也用了这个文件

# 5. Safely read scores
if os.path.exists(scores_file):
    scores = pd.read_csv(scores_file)
else:
    st.error(f"❌ scores.csv not found in {data_path}")
    st.stop()


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

df = scores.copy()

# Prepare colors
keywords = sorted(df["keyword"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(keywords)))
color_map = dict(zip(keywords, colors))

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter per keyword
for kw in keywords:
    d = df[df["keyword"] == kw]
    ax.scatter(
        d["final_score"],
        d["sentiment_score"],
        color=color_map[kw],
        label=kw,
        s=55,
        alpha=0.85
    )

# Lines
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
median_x = df["final_score"].median()
ax.axvline(median_x, color="gray", linestyle="--", linewidth=1)

# Labels
ax.set_xlabel("Impact Score (final_score)", fontsize=12)
ax.set_ylabel("Market Sentiment (sentiment_score)", fontsize=12)
ax.set_title("Impact vs Market Sentiment", fontsize=14)

# Legend inside top-left, with transparent box
legend = ax.legend(
    title="Keyword",
    loc="upper left",
    framealpha=0.6,        # semi-transparent background
    facecolor="white",
    edgecolor="gray"
)

st.pyplot(fig)


# -------------------------------------------------------
# 2.2 Alpha Quadrant - graph_objects 版
# -------------------------------------------------------

st.markdown("#### 2.2 Alpha Quadrant: Credibility vs Materiality")

if not os.path.exists(alpha_file):
    st.info("No alpha.csv found.")
else:
    df_alpha = pd.read_csv(alpha_file).copy()

    # Midpoints
    x_mid = df_alpha["source_credibility"].median()
    y_mid = df_alpha["materiality_score"].median()

    # Normalize sentiment for color map
    senti = df_alpha["sentiment_score"].clip(-1, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    sc = ax.scatter(
        df_alpha["source_credibility"],
        df_alpha["materiality_score"],
        c=senti,
        cmap="RdYlGn",        # red → yellow → green
        s=70,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3
    )

    # Quadrant lines
    ax.axvline(x_mid, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="gray", linestyle="--", linewidth=1)

    # Labels
    ax.set_xlabel("Source Credibility", fontsize=12)
    ax.set_ylabel("Materiality Score", fontsize=12)
    ax.set_title("Alpha Quadrant", fontsize=14)

    # Quadrant annotations
    ax.text(x_mid + 0.01, y_mid + 0.01, "Q1: Critical Movers", fontsize=10, color="black")
    ax.text(x_mid - 0.29, y_mid + 0.01, "Q2: Rumor Mill", fontsize=10, color="black")
    ax.text(x_mid - 0.29, y_mid - 0.03, "Q3: Low Value", fontsize=10, color="black")
    ax.text(x_mid + 0.01, y_mid - 0.03, "Q4: Market Noise", fontsize=10, color="black")

    # Colorbar as "legend", placed inside top-left
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sentiment Score", fontsize=11)

    # Move colorbar inside + style
    cbar.ax.set_position([0.13, 0.75, 0.02, 0.15])  # left, bottom, width, height
    cbar.ax.set_facecolor((1,1,1,0.6))             # semi-transparent

    st.pyplot(fig)

                
# -------------------------------------------------------
# 2.3 Word Cloud 
# -------------------------------------------------------

wc = pd.read_csv(wordcount_file)

unique_keywords = wc["keyword"].unique()
k = len(unique_keywords)

# ----- Step 1: generate base colors evenly across the hue wheel -----
def generate_base_colors(num):
    colors = []
    for i in range(num):
        h = i / num                     # evenly spaced hues [0,1)
        s = 0.55                         # moderate saturation
        l = 0.55                         # mid lightness
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
    return colors

base_colors = generate_base_colors(k)

# ----- Step 2: generate 5-shade harmonious palettes from base color -----
def make_palette(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    shades = []
    for factor in [0.6, 0.75, 0.9, 1.05, 1.2]:
        new_l = min(1, max(0, l * factor))
        rr, gg, bb = colorsys.hls_to_rgb(h, new_l, s)
        shades.append('#%02x%02x%02x' % (int(rr*255), int(gg*255), int(bb*255)))
    return shades

keyword_palettes = {kw: make_palette(base_colors[i]) for i, kw in enumerate(unique_keywords)}
st.markdown("#### 2.3 Word Cloud")

# ----- Render word clouds -----
cols_per_row = 3

for i in range(0, len(unique_keywords), cols_per_row):
    row_keywords = unique_keywords[i:i+cols_per_row]
    cols = st.columns(len(row_keywords))

    for col, kw in zip(cols, row_keywords):
        subset = wc[wc["keyword"] == kw]
        freq = dict(zip(subset["word"], subset["count"]))

        palette = keyword_palettes[kw]

        def color_func(*args, **kwargs):
            return random.choice(palette)

        wc_img = WordCloud(
            width=500, height=350, background_color="white",
            collocations=False
        ).generate_from_frequencies(freq)

        wc_img = wc_img.recolor(color_func=color_func)
        col.markdown(f"**{kw.capitalize()}**")
        col.image(wc_img.to_array(), use_column_width=True)

# --------------------------------------------------
# 3. Sentiment Score
# --------------------------------------------------

df_sent = pd.read_csv(sentiment_file)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3. Sentiment Distribution") 
#######check here

categories = ["strong_neg", "weak_neg", "neutral", "weak_pos", "strong_pos"]
df_plot = df_sent.sort_values("keyword")

x = np.arange(len(df_plot["keyword"]))
bottom = np.zeros(len(df_plot))

fig_b, ax_b = plt.subplots(figsize=(10, 5))
for cat in categories:
    vals = df_plot[cat].astype(float)
    ax_b.bar(x, vals, bottom=bottom, label=cat)
    bottom += vals

ax_b.set_xticks(x)
ax_b.set_xticklabels(df_plot["keyword"], rotation=30, ha="right")
ax_b.legend()

st.pyplot(fig_b, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------
# 4. Entity Co-occurrence Network
# -------------------------------------------------------


# 1. Load entities.json safely
OUTPUT_DIR = "network_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(entities_file):
    with open(entities_file, "r", encoding="utf-8") as f:
        raw_entities = json.load(f)
else:
    raw_entities = []


# 2. Helpers: color mapping & data shaping

def sentiment_to_color(s: float) -> str:
    """
    Map sentiment score [-1,1] to a red–green color scale.
    Higher |sentiment| -> deeper color.
    """
    s = max(-1.0, min(1.0, float(s)))
    intensity = int(150 + abs(s) * 105)  # 150–255
    if s >= 0:
        return f"rgb(0,{intensity},0)"   # greenish
    else:
        return f"rgb({intensity},0,0)"   # reddish


def normalize_entities(raw):
    """
    Ensure we always have a list of dicts:
    [{"keyword": ..., "graph_data": [...]}, ...]
    """
    # Case 1: already list of blocks
    if isinstance(raw, list):
        items = []
        for e in raw:
            if isinstance(e, dict) and "keyword" in e and "graph_data" in e:
                items.append(e)
        return items

    # Case 2: dict keyed by keyword
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict) and "graph_data" in v:
                items.append({"keyword": k, "graph_data": v["graph_data"]})
            elif isinstance(v, list):
                items.append({"keyword": k, "graph_data": v})
        return items

    return []


def build_network_data(entry):
    """
    Build entity frequency, avg sentiment and co-occurrence
    for a single keyword block.
    """
    articles = entry.get("graph_data", [])

    entity_freq = defaultdict(int)
    entity_sent_sum = defaultdict(float)
    entity_sent_count = defaultdict(int)
    cooccur = defaultdict(lambda: defaultdict(int))

    for art in articles:
        ents = art.get("entities", [])
        sentiment = float(art.get("sentiment", 0))

        # entity-level stats
        for e in ents:
            entity_freq[e] += 1
            entity_sent_sum[e] += sentiment
            entity_sent_count[e] += 1

        # pairwise co-occurrence
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    entity_sent_avg = {
        e: entity_sent_sum[e] / entity_sent_count[e]
        for e in entity_freq
    }

    return entity_freq, entity_sent_avg, cooccur

# 3. Graph generator (PyVis)

def generate_pyvis_graph(keyword, entity_freq, entity_sent_avg, cooccur):
    """
    Create a PyVis graph for one keyword and save as HTML.
    """
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#222",
    )

    # Compatible physics
    net.barnes_hut()

    # Slightly more compact layout (JS-level options)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.25,
          "springLength": 90,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)

    # Keyword node (center)
    net.add_node(
        keyword,
        label=keyword,
        size=55,
        color="#4976f5",
        title=f"{keyword} — {len(entity_freq)} entities"
    )

    max_freq = max(entity_freq.values()) if entity_freq else 1

    # Entity nodes
    for ent, freq in entity_freq.items():
        sentiment = entity_sent_avg.get(ent, 0.0)
        size = 10 + (freq / max_freq) * 25

        net.add_node(
            ent,
            label=ent,
            size=size,
            color=sentiment_to_color(sentiment),
            title=f"{ent} | Count: {freq} | Sent: {sentiment:.2f}"
        )
        net.add_edge(keyword, ent, color="#999999", width=1)

    # Co-occurrence edges
    for a in cooccur:
        for b, count in cooccur[a].items():
            if count > 0:
                net.add_edge(
                    a, b,
                    width=1 + count * 0.7,
                    color="#BBBBBB",
                    title=f"Co-occurs: {count}"
                )

    file_path = os.path.join(OUTPUT_DIR, f"{keyword}_network.html")
    net.write_html(file_path)
    return file_path


# -------------------------------------------------------
# 4. Build graphs & layout in Streamlit
# -------------------------------------------------------

entity_blocks = normalize_entities(raw_entities)

if not entity_blocks:
    st.subheader("D. Entity Co-occurrence Network")
    st.info("No entity data available for this date.")
else:
    network_files = {}
    for entry in entity_blocks:
        keyword = entry.get("keyword", "unknown")
        entity_freq, entity_sent_avg, cooccur = build_network_data(entry)
        html_file = generate_pyvis_graph(keyword, entity_freq, entity_sent_avg, cooccur)
        network_files[keyword] = html_file

    st.subheader("4. Entity Co-occurrence Network")

    keywords = list(network_files.keys())

    for i in range(0, len(keywords), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(keywords):
                key = keywords[i + j]
                file_path = network_files[key]

                with cols[j]:
                    with st.expander(f"{key.title()} Entity Network", expanded=False):
                        with open(file_path, "r", encoding="utf-8") as f:
                            html(f.read(), height=600)




# -------------------------------------------------------
# 5. "Must-Read" Ticker
# -------------------------------------------------------


st.subheader("5. The 'Must-Read' Ticker")

if scores is None or len(scores) == 0:
    st.info("No score data available.")
else:

    # ---------- Helper Functions ----------
    def sentiment_dot(s):
        if s > 0.15:
            return "🟢"
        elif s < -0.15:
            return "🔴"
        else:
            return "⚪️"

    def consensus_label(avg_senti):
        if avg_senti > 0.15:
            return "Bullish Consensus"
        elif avg_senti < -0.15:
            return "Bearish Consensus"
        else:
            return "Mixed"

    def extract_domain(url):
        try:
            return url.split("//")[1].split("/")[0].replace("www.", "")
        except:
            return ""

    grouped = scores.groupby("keyword")

    # ---------- Render Each Keyword as a Card ----------
    for keyword, df in grouped:

        avg_sent = df["sentiment_score"].mean()
        consensus = consensus_label(avg_sent)

        # Card container
        with st.container():
            st.markdown(
                f"""
                <div style="
                    padding: 20px 25px;
                    background-color: #F7F9FC;
                    border-radius: 12px;
                    border: 1px solid #E4E9F1;
                    margin-bottom: 22px;
                ">
                    <h3 style="margin-top:0; color:#1A1D27; font-size:22px;">
                        {keyword.title()} 
                        <span style="font-size:16px; font-weight:400; color:#6A7280;">
                            ({consensus})
                        </span>
                    </h3>
                """,
                unsafe_allow_html=True,
            )

            # Top 3 articles
            top3 = df.sort_values("final_score", ascending=False).head(3)

            for rank, (_, row) in enumerate(top3.iterrows(), start=1):
                dot = sentiment_dot(row["sentiment_score"])
                title = row["title"]
                score = int(row["final_score"])
                url = row["url"]
                domain = extract_domain(url)

                st.markdown(
                    f"""
                    <div style="margin-bottom:18px;">
                        <div style="font-size:17px;">
                            <b>{rank}. {dot} <a href="{url}" target="_blank" style="text-decoration:none; color:#1A73E8;">
                            {title}</a></b>
                        </div>
                        <div style="font-size:14px; color:#68707C; margin-left: 26px;">
                            {domain} • Score: {score}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)





