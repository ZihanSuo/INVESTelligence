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
data_path = f"data/{today}"

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
# B2. The Alpha Quadrant (Four Quadrant Analysis)
# -------------------------------------------------------

st.markdown("### B2. The Alpha Quadrant (Source Credibility × Materiality)")

alpha_file = os.path.join(data_path, "alpha.csv")
alpha = pd.read_csv(alpha_file)

# Basic preparation
df = alpha.copy()
df["sentiment_norm"] = df["sentiment_score"].clip(-1, 1)

# Determine quadrant boundaries using medians
x_mid = df["source_credibility"].median()
y_mid = df["materiality_score"].median()

# Quadrant labeling function
def get_quadrant(row):
    if row["source_credibility"] >= x_mid and row["materiality_score"] >= y_mid:
        return "Critical Movers (Q1)"
    elif row["source_credibility"] < x_mid and row["materiality_score"] >= y_mid:
        return "Rumor Mill (Q2)"
    elif row["source_credibility"] < x_mid and row["materiality_score"] < y_mid:
        return "Low Value Noise (Q3)"
    else:
        return "Market Noise (Q4)"

df["quadrant"] = df.apply(get_quadrant, axis=1)

# Build scatter plot
fig_q = px.scatter(
    df,
    x="source_credibility",
    y="materiality_score",
    color="sentiment_norm",
    color_continuous_scale=["red", "white", "green"],
    hover_data=["title", "keyword", "url"],
    size=[12] * len(df),
    title="Alpha Quadrant: Credibility vs Materiality"
)

# Draw quadrant lines
fig_q.add_vline(x=x_mid, line_width=1, line_dash="dash", line_color="gray")
fig_q.add_hline(y=y_mid, line_width=1, line_dash="dash", line_color="gray")

# Add quadrant text labels
fig_q.add_annotation(x=x_mid + 0.1, y=y_mid + 0.1, text="Q1: Critical Movers", showarrow=False)
fig_q.add_annotation(x=x_mid - 0.1, y=y_mid + 0.1, text="Q2: Rumor Mill", showarrow=False)
fig_q.add_annotation(x=x_mid - 0.1, y=y_mid - 0.1, text="Q3: Low Value", showarrow=False)
fig_q.add_annotation(x=x_mid + 0.1, y=y_mid - 0.1, text="Q4: Market Noise", showarrow=False)

st.plotly_chart(fig_q, use_container_width=True, key="alpha_quadrant")


# -------------------------------------------------------
# C. Word Cloud (scalable for unlimited keywords)
# -------------------------------------------------------

import colorsys
import random
from wordcloud import WordCloud

st.markdown("### C. Keyword Word Cloud")

word_count_file = os.path.join(data_path, "word_count.csv")
wc = pd.read_csv(word_count_file)

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
        col.image(wc_img.to_array(), use_container_width=True)




#----------------------ENTITIES

KEYWORD_COLORS = {
    "bitcoin":  {"main": "#4976f5", "light": "#AFC7FA"},
    "rare earth": {"main": "#2CB67D", "light": "#7FDDB1"},
    "tesla": {"main": "#D84D4D", "light": "#F2A7A7"},
}
import json
from collections import defaultdict

# -------------------------
# Load entity-level graph data
# -------------------------
with open("data/2025-12-10/entities.json", "r") as f:
    entities_data = json.load(f)

# [
#   {
#     "keyword": "...",
#     "graph_data": [
#         {"title": "...", "entities": [...], "sentiment": 0.3, ...}
#     ]
#   }
# ]
def build_network_data(entry):
    keyword = entry["keyword"]
    articles = entry["graph_data"]

    # 实体出现次数
    entity_freq = defaultdict(int)
    # 实体 sentiment 统计
    entity_sent_sum = defaultdict(float)
    entity_sent_count = defaultdict(int)

    # 共现关系计数
    cooccur = defaultdict(lambda: defaultdict(int))

    for art in articles:
        ents = art.get("entities", [])
        sentiment = art.get("sentiment", 0)

        # 统计实体出现频率 & sentiment
        for e in ents:
            entity_freq[e] += 1
            entity_sent_sum[e] += sentiment
            entity_sent_count[e] += 1

        # 构建共现对（entity pair）
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    # 计算实体平均 sentiment
    entity_sent_avg = {e: entity_sent_sum[e] / entity_sent_count[e] for e in entity_freq}

    return entity_freq, entity_sent_avg, cooccur
from pyvis.network import Network
import os

OUTPUT_DIR = "network_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pyvis_graph(keyword, entity_freq, entity_sent_avg, cooccur):
    net = Network(height="550px", width="100%", bgcolor="#FFFFFF", font_color="#1D1D1F")
    net.barnes_hut()

    colors = KEYWORD_COLORS.get(keyword, {"main": "#666", "light": "#CCC"})

    # -----------------------
    # 添加 Keyword 中心节点
    # -----------------------
    net.add_node(
        keyword,
        label=keyword,
        size=45,  
        color=colors["main"],
        title=f"<b>{keyword}</b><br>Total Entities: {len(entity_freq)}"
    )

    # -----------------------
    # 添加实体节点
    # -----------------------
    max_freq = max(entity_freq.values()) if entity_freq else 1

    for ent, freq in entity_freq.items():
        size = 10 + (freq / max_freq) * 25   # 中等大小（最小10 → 最大35）

        sentiment = entity_sent_avg.get(ent, 0)

        net.add_node(
            ent,
            label=ent,
            size=size,
            color=colors["light"],
            title=f"{ent}<br>Count: {freq}<br>Avg Sentiment: {sentiment:.2f}"
        )

        # keyword → entity
        net.add_edge(keyword, ent, color="#999", width=1)

    # -----------------------
    # 添加实体共现边
    # -----------------------
    for a in cooccur:
        for b, count in cooccur[a].items():
            if count > 0:
                net.add_edge(
                    a, b,
                    color="#666",
                    width=1 + count * 0.5,  # 共现次数越高 → 边越粗
                    title=f"Co-occurs {count} times"
                )

    file_path = f"{OUTPUT_DIR}/{keyword}_network.html"
    net.write_html(file_path)
    return file_path

# -------------------------
# 生成所有 keyword 的网络图
# -------------------------
network_files = {}

for entry in entities_data:
    keyword = entry["keyword"]

    entity_freq, entity_sent_avg, cooccur = build_network_data(entry)
    html_file = generate_pyvis_graph(keyword, entity_freq, entity_sent_avg, cooccur)

    network_files[keyword] = html_file

# -------------------------
# Streamlit 展示部分（缺了这个你就永远看不到图）
# -------------------------
import streamlit as st
from streamlit.components.v1 import html

st.subheader("C. Entity Co-occurrence Network")

keywords = list(network_files.keys())

# 一行 2 个 collapsible 图
for i in range(0, len(keywords), 2):
    cols = st.columns(2)

    for j in range(2):
        if i + j < len(keywords):
            key = keywords[i + j]
            file_path = network_files[key]

            with cols[j]:
                with st.expander(f"{key.title()} Entity Network", expanded=False):
                    html(open(file_path, "r", encoding="utf-8").read(), height=600)


