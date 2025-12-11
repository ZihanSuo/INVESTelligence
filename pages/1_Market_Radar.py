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
# 2.1 Impact vs Market Sentiment - FIXED
# -------------------------------------------------------

st.markdown("#### 2.1 Impact vs Market Sentiment")

# 🔧 创建一个干净的副本，避免修改全局 scores
df_viz = scores.copy()

# 确保 pickup_count 存在（已经在前面 merge 过了）
if "pickup_count" in df_viz.columns:
    pc = df_viz["pickup_count"].fillna(0)
    if pc.max() > pc.min():
        df_viz["marker_size"] = 10 + 20 * (pc - pc.min()) / (pc.max() - pc.min())
    else:
        df_viz["marker_size"] = 14
else:
    df_viz["marker_size"] = 14

# 创建散点图
fig_scatter = px.scatter(
    df_viz,
    x="final_score",
    y="sentiment_score",
    color="keyword",
    size="marker_size",
    hover_data=["title", "final_score", "sentiment_score", "url"],
    color_discrete_sequence=px.colors.qualitative.Set2
)

# 添加参考线
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.5)
fig_scatter.add_vline(
    x=df_viz["final_score"].median(), 
    line_dash="dash", 
    line_color="gray",
    line_width=1.5
)

# 优化布局
fig_scatter.update_layout(
    height=500,
    xaxis_title="Impact Score (final_score)",
    yaxis_title="Market Sentiment",
    showlegend=True,
    hovermode='closest'
)

# 更新点的样式
fig_scatter.update_traces(
    marker=dict(
        line=dict(width=1, color='white'),
        opacity=0.8
    )
)

st.plotly_chart(fig_scatter, use_container_width=True)


# -------------------------------------------------------
# 2.2 Alpha Quadrant - FIXED
# -------------------------------------------------------

st.markdown("#### 2.2 Alpha Quadrant: Credibility vs Materiality")

if not os.path.exists(alpha_file):
    st.info("No alpha.csv for today.")
else:
    # 🔧 重新读取 alpha.csv，不使用之前的变量
    df_alpha = pd.read_csv(alpha_file)
    
    # 清理数据
    df_alpha = df_alpha.dropna(subset=['source_credibility', 'materiality_score', 'sentiment_score'])
    
    if len(df_alpha) == 0:
        st.warning("No valid data in alpha.csv")
    else:
        # 标准化 sentiment
        df_alpha["sentiment_norm"] = df_alpha["sentiment_score"].clip(-1, 1)
        
        # 计算中位数
        x_mid = df_alpha["source_credibility"].median()
        y_mid = df_alpha["materiality_score"].median()
        
        # 象限分类
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
        
        # 创建散点图
        fig_q = px.scatter(
            df_alpha,
            x="source_credibility",
            y="materiality_score",
            color="sentiment_norm",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data=["title", "keyword", "url"],
            size=[14] * len(df_alpha)
        )
        
        # 添加参考线
        fig_q.add_vline(x=x_mid, line_dash="dash", line_color="gray", line_width=2)
        fig_q.add_hline(y=y_mid, line_dash="dash", line_color="gray", line_width=2)
        
        # 添加象限标签
        fig_q.add_annotation(x=x_mid + 0.05, y=y_mid + 0.05, 
                            text="Q1: Critical Movers", showarrow=False, 
                            font=dict(size=11, color="green"))
        fig_q.add_annotation(x=x_mid - 0.05, y=y_mid + 0.05, 
                            text="Q2: Rumor Mill", showarrow=False, 
                            font=dict(size=11, color="orange"), xanchor="right")
        fig_q.add_annotation(x=x_mid - 0.05, y=y_mid - 0.05, 
                            text="Q3: Low Value", showarrow=False, 
                            font=dict(size=11, color="gray"), 
                            xanchor="right", yanchor="top")
        fig_q.add_annotation(x=x_mid + 0.05, y=y_mid - 0.05, 
                            text="Q4: Market Noise", showarrow=False, 
                            font=dict(size=11, color="blue"), yanchor="top")
        
        # 更新布局
        fig_q.update_layout(
            height=500,
            coloraxis_colorbar=dict(
                title="Sentiment",
                tickvals=[-1, 0, 1],
                ticktext=["Bearish", "Neutral", "Bullish"]
            )
        )
        
        # 更新点的样式
        fig_q.update_traces(
            marker=dict(
                line=dict(width=1, color='white'),
                opacity=0.8
            )
        )
        
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



            st.markdown("### 🔍 诊断模式")

# Step 1: 检查 scores 数据
st.write("**Step 1: 检查 scores 数据**")
st.write(f"scores 的形状: {scores.shape}")
st.write(f"scores 的列: {scores.columns.tolist()}")

# 检查是否有重复列
duplicate_cols = scores.columns[scores.columns.duplicated()].tolist()
if duplicate_cols:
    st.error(f"❌ scores 有重复列: {duplicate_cols}")
else:
    st.success("✅ scores 没有重复列")

# Step 2: 显示前5行数据
st.write("**Step 2: scores 前5行数据**")
st.dataframe(scores[['keyword', 'title', 'final_score', 'sentiment_score']].head())

# Step 3: 检查数据统计
st.write("**Step 3: 数据统计**")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("总行数", len(scores))
    st.metric("final_score 非空", scores['final_score'].notna().sum())
    st.metric("final_score 有无穷大", scores['final_score'].isin([float('inf'), float('-inf')]).sum())

with col2:
    st.metric("sentiment_score 非空", scores['sentiment_score'].notna().sum())
    st.metric("sentiment_score 有无穷大", scores['sentiment_score'].isin([float('inf'), float('-inf')]).sum())
    st.metric("keyword 非空", scores['keyword'].notna().sum())

with col3:
    st.metric("final_score 范围", f"{scores['final_score'].min():.1f} ~ {scores['final_score'].max():.1f}")
    st.metric("sentiment_score 范围", f"{scores['sentiment_score'].min():.2f} ~ {scores['sentiment_score'].max():.2f}")

# Step 4: 创建副本并清理
st.write("**Step 4: 创建干净副本**")
df_viz = scores.copy()

# 移除重复列（如果有）
if df_viz.columns.duplicated().any():
    st.warning("移除重复列...")
    df_viz = df_viz.loc[:, ~df_viz.columns.duplicated()]

# 重置索引
df_viz = df_viz.reset_index(drop=True)

st.write(f"清理后的形状: {df_viz.shape}")
st.write(f"清理后的列: {df_viz.columns.tolist()}")

# Step 5: 检查必需列
required_cols = ['final_score', 'sentiment_score', 'keyword']
missing = [col for col in required_cols if col not in df_viz.columns]

if missing:
    st.error(f"❌ 缺少必需列: {missing}")
    st.stop()
else:
    st.success("✅ 所有必需列存在")

# Step 6: 清理数据
st.write("**Step 6: 数据清理**")
before_count = len(df_viz)

# 移除 NaN
df_viz = df_viz.dropna(subset=['final_score', 'sentiment_score', 'keyword'])
after_nan = len(df_viz)

# 移除无穷大
df_viz = df_viz[~df_viz['final_score'].isin([float('inf'), float('-inf')])]
df_viz = df_viz[~df_viz['sentiment_score'].isin([float('inf'), float('-inf')])]
after_inf = len(df_viz)

st.write(f"原始行数: {before_count}")
st.write(f"移除 NaN 后: {after_nan} (移除了 {before_count - after_nan} 行)")
st.write(f"移除无穷大后: {after_inf} (移除了 {after_nan - after_inf} 行)")

if len(df_viz) == 0:
    st.error("❌ 清理后没有数据了！")
    st.stop()
else:
    st.success(f"✅ 清理后还有 {len(df_viz)} 行数据")

# Step 7: 显示清理后的数据
st.write("**Step 7: 清理后的数据样本**")
st.dataframe(df_viz[['keyword', 'title', 'final_score', 'sentiment_score']].head(10))

# Step 8: 准备 marker_size
st.write("**Step 8: 准备 marker_size**")
if "pickup_count" in df_viz.columns:
    pc = df_viz["pickup_count"].fillna(0)
    st.write(f"pickup_count 范围: {pc.min()} ~ {pc.max()}")
    
    if pc.max() > pc.min():
        df_viz["marker_size"] = 10 + 20 * (pc - pc.min()) / (pc.max() - pc.min())
    else:
        df_viz["marker_size"] = 14
else:
    df_viz["marker_size"] = 14
    st.info("没有 pickup_count 列，使用固定大小")

st.write(f"marker_size 范围: {df_viz['marker_size'].min()} ~ {df_viz['marker_size'].max()}")

# Step 9: 尝试创建最简单的图
st.write("**Step 9: 测试 Plotly 散点图**")

try:
    fig_test = px.scatter(
        df_viz,
        x="final_score",
        y="sentiment_score",
        color="keyword",
        size="marker_size",
        hover_data=["title"],
        title="Test Scatter Plot"
    )
    
    fig_test.update_layout(height=500)
    st.plotly_chart(fig_test, use_container_width=True)
    
    st.success("✅ Plotly 图表创建成功")
    
except Exception as e:
    st.error(f"❌ 创建图表时出错: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Step 10: 尝试用 matplotlib
st.write("**Step 10: 备用方案 - Matplotlib**")

try:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for kw in df_viz['keyword'].unique():
        subset = df_viz[df_viz['keyword'] == kw]
        ax.scatter(subset['final_score'], subset['sentiment_score'], 
                  label=kw, s=100, alpha=0.6, edgecolors='black')
    
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.axvline(x=df_viz['final_score'].median(), color='gray', linestyle='--')
    ax.set_xlabel('Final Score')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    st.success("✅ Matplotlib 图表创建成功")
    
except Exception as e:
    st.error(f"❌ Matplotlib 出错: {str(e)}")

