########## Import Packages #############
# 1. Global Setup & Data Loading

import streamlit as st
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
# Merge pickup_count (if alpha exists)
# -------------------------------------------------------
if has_alpha:
    scores = scores.merge(
        alpha[["title", "pickup_count"]],
        on="title",
        how="left"
    )


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
# 2. Alpha Matrix: Core signals
# -------------------------------------------------------
st.markdown("### 2. Alpha Matrix (Core Signals)")

# -------------------------------------------------------
# 2.1 Impact vs Market Sentiment (FIXED VERSION)
# -------------------------------------------------------

st.markdown("#### 2.1 Impact vs Market Sentiment")

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
# 2.2 The Alpha Quadrant (Four Quadrant Analysis)
# -------------------------------------------------------
st.markdown("#### 2.2 Alpha Quadrant: Credibility vs Materiality")

if not os.path.exists(alpha_file):
    st.info("No alpha.csv for today.")
else:
    df_alpha = pd.read_csv(alpha_file).copy()
    
    # 🔍 清理数据
    df_alpha = df_alpha.dropna(subset=['source_credibility', 'materiality_score', 'sentiment_score'])
    

if not os.path.exists(alpha_file):
    st.info("No alpha.csv for today.")
else:
    df_alpha = pd.read_csv(alpha_file).copy()

    # Normalize sentiment score for visualization
    df_alpha["sentiment_norm"] = df_alpha["sentiment_score"].clip(-1, 1)

    # Compute quadrant midpoints
    x_mid = df_alpha["source_credibility"].median()
    y_mid = df_alpha["materiality_score"].median()

    # Quadrant function
    def get_quadrant(row):
        if row["source_credibility"] >= x_mid and row["materiality_score"] >= y_mid:
            return "Critical Movers (Q1)"
        elif row["source_credibility"] < x_mid and row["materiality_score"] >= y_mid:
            return "Rumor Mill (Q2)"
        elif row["source_credibility"] < x_mid and row["materiality_score"] < y_mid:
            return "Low Value (Q3)"
        else:
            return "Market Noise (Q4)"

    df_alpha["quadrant"] = df_alpha.apply(get_quadrant, axis=1)

    # Build quadrants scatter plot
    fig_q = px.scatter(
        df_alpha,
        x="source_credibility",
        y="materiality_score",
        color="sentiment_norm",
        color_continuous_scale=["red", "white", "green"],
        hover_data=["title", "keyword", "url"],
        size=[12] * len(df_alpha)
    )

    # Add lines
    fig_q.add_vline(x=x_mid, line_dash="dash", line_color="gray")
    fig_q.add_hline(y=y_mid, line_dash="dash", line_color="gray")

    # Add quadrant labels
    fig_q.add_annotation(x=x_mid + 0.05, y=y_mid + 0.05, text="Q1: Critical Movers", showarrow=False)
    fig_q.add_annotation(x=x_mid - 0.05, y=y_mid + 0.05, text="Q2: Rumor Mill", showarrow=False)
    fig_q.add_annotation(x=x_mid - 0.05, y=y_mid - 0.05, text="Q3: Low Value", showarrow=False)
    fig_q.add_annotation(x=x_mid + 0.05, y=y_mid - 0.05, text="Q4: Market Noise", showarrow=False)

    fig_q.update_layout(height=500)

    st.plotly_chart(fig_q, use_container_width=True)


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
