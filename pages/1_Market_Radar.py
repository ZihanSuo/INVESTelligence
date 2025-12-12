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

# Prepare colors for each keyword
keywords = sorted(df["keyword"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(keywords)))
color_map = dict(zip(keywords, colors))

# Figure setup
fig, ax = plt.subplots(figsize=(10, 6))

# Draw each keyword's scatter
for kw in keywords:
    d = df[df["keyword"] == kw]
    ax.scatter(
        d["final_score"],
        d["sentiment_score"],
        color=color_map[kw],
        label=kw,
        s=50,        # marker size
        alpha=0.8
    )

# Horizontal line for sentiment = 0
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

# Vertical line for median(final_score)
median_x = df["final_score"].median()
ax.axvline(median_x, color="gray", linestyle="--", linewidth=1)

# Labels
ax.set_xlabel("Impact Score (final_score)", fontsize=12)
ax.set_ylabel("Market Sentiment (sentiment_score)", fontsize=12)
ax.set_title("Impact vs Market Sentiment (Matplotlib Version)", fontsize=14)

# Show legend
ax.legend(title="Keyword", bbox_to_anchor=(1.05, 1), loc='upper left')

# Push to Streamlit
st.pyplot(fig)



# -------------------------------------------------------
# 2.2 Alpha Quadrant - graph_objects 版
# -------------------------------------------------------

st.markdown("#### 2.2 Alpha Quadrant: Credibility vs Materiality")

if not os.path.exists(alpha_file):
    st.info("No alpha.csv for today.")
else:
    # 读取数据
    df_alpha = pd.read_csv(alpha_file)
    
    # 清理
    if df_alpha.columns.duplicated().any():
        df_alpha = df_alpha.loc[:, ~df_alpha.columns.duplicated()]
    
    df_alpha = df_alpha.reset_index(drop=True)
    df_alpha = df_alpha.dropna(subset=['source_credibility', 'materiality_score', 'sentiment_score'])
    
    if len(df_alpha) == 0:
        st.warning("No valid data in alpha.csv")
    else:
        # 计算中位数
        x_mid = df_alpha["source_credibility"].median()
        y_mid = df_alpha["materiality_score"].median()
        
        # 标准化 sentiment
        sentiment_norm = df_alpha["sentiment_score"].clip(-1, 1)
        
        # 创建颜色映射 (红→黄→绿)
        def sentiment_to_color(s):
            # s 范围 [-1, 1]
            # 红(负) → 黄(0) → 绿(正)
            if s < 0:
                # 红色到黄色
                r = 255
                g = int(255 * (1 + s))  # -1→0, 0→255
                b = 0
            else:
                # 黄色到绿色
                r = int(255 * (1 - s))  # 0→255, 1→0
                g = 255
                b = 0
            return f'rgb({r},{g},{b})'
        
        colors = [sentiment_to_color(s) for s in sentiment_norm]
        
        # 构建 hover text
        hover_text = []
        for _, row in df_alpha.iterrows():
            text = f"<b>{row['title']}</b><br>"
            text += f"Keyword: {row['keyword']}<br>"
            text += f"Credibility: {row['source_credibility']:.2f}<br>"
            text += f"Materiality: {row['materiality_score']:.2f}<br>"
            text += f"Sentiment: {row['sentiment_score']:.2f}<br>"
            text += f"URL: {row['url']}"
            hover_text.append(text)
        
        # 创建散点图
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=df_alpha['source_credibility'],
            y=df_alpha['materiality_score'],
            mode='markers',
            marker=dict(
                size=14,
                color=colors,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # 添加参考线
        fig2.add_vline(
            x=x_mid,
            line_dash="dash",
            line_color="gray",
            line_width=2
        )
        
        fig2.add_hline(
            y=y_mid,
            line_dash="dash",
            line_color="gray",
            line_width=2
        )
        
        # 计算偏移量用于标签
        x_range = df_alpha['source_credibility'].max() - df_alpha['source_credibility'].min()
        y_range = df_alpha['materiality_score'].max() - df_alpha['materiality_score'].min()
        offset_x = x_range * 0.05
        offset_y = y_range * 0.05
        
        # 添加象限标签
        fig2.add_annotation(
            x=x_mid + offset_x, 
            y=y_mid + offset_y,
            text="<b>Q1: Critical Movers</b><br>(High Cred + High Impact)",
            showarrow=False,
            font=dict(size=10, color="green"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=4
        )
        
        fig2.add_annotation(
            x=x_mid - offset_x,
            y=y_mid + offset_y,
            text="<b>Q2: Rumor Mill</b><br>(Low Cred + High Impact)",
            showarrow=False,
            font=dict(size=10, color="orange"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=4,
            xanchor="right"
        )
        
        fig2.add_annotation(
            x=x_mid - offset_x,
            y=y_mid - offset_y,
            text="<b>Q3: Low Value</b><br>(Low Cred + Low Impact)",
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=4,
            xanchor="right",
            yanchor="top"
        )
        
        fig2.add_annotation(
            x=x_mid + offset_x,
            y=y_mid - offset_y,
            text="<b>Q4: Market Noise</b><br>(High Cred + Low Impact)",
            showarrow=False,
            font=dict(size=10, color="blue"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=4,
            yanchor="top"
        )
        
        # 更新布局
        fig2.update_layout(
            title="Alpha Quadrant: Credibility vs Materiality",
            xaxis_title="Source Credibility",
            yaxis_title="Materiality Score",
            height=550,
            hovermode='closest',
            showlegend=False,
            plot_bgcolor='rgba(245, 245, 245, 0.5)'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 象限统计
        def get_quadrant(row):
            if row["source_credibility"] >= x_mid and row["materiality_score"] >= y_mid:
                return "Q1: Critical Movers"
            elif row["source_credibility"] < x_mid and row["materiality_score"] >= y_mid:
                return "Q2: Rumor Mill"
            elif row["source_credibility"] < x_mid and row["materiality_score"] < y_mid:
                return "Q3: Low Value"
            else:
                return "Q4: Market Noise"
        
        df_alpha["quadrant"] = df_alpha.apply(get_quadrant, axis=1)
        
        st.markdown("##### Quadrant Distribution")
        quad_counts = df_alpha["quadrant"].value_counts()
        
        cols = st.columns(4)
        quadrants = ["Q1: Critical Movers", "Q2: Rumor Mill", "Q3: Low Value", "Q4: Market Noise"]
        for i, quad in enumerate(quadrants):
            with cols[i]:
                count = quad_counts.get(quad, 0)
                st.metric(quad, count)
                
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





