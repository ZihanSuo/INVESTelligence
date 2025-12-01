import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Page Setup 
st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

# Page layout styling
st.markdown(
    """
    <style>
    .main-block {
        max-width: 1300px;
        margin: 0 auto;
        padding: 10px 20px 40px 20px;
    }
    .card {
        padding: 18px 20px 20px 20px;
        border-radius: 14px;
        background-color: #FFF5F7;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 8px;
    }
    </style>
    <div class="main-block">
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Page Style
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",  # red–yellow–green–blue soft
    "#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA",  # classic pastel
    "#F4B6C2", "#F6E2B3", "#C7D8C6", "#D5E1DF", "#E4C1F9",  # purple variants
    "#F1C0E8", "#FDE2E4", "#FAD2E1", "#D8E2DC", "#ECE4DB",  # elegant earth pastels
]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PASTEL_COLORS)

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
        color_discrete_sequence=PASTEL_COLORS,
        size_max=12,
        opacity=0.85,
        height=600
    )

    fig.update_traces(hovertemplate="%{customdata[0]}")

    fig.update_layout(
        title="Mixed Scatter Plot (Top 10 Articles per Keyword)",
        xaxis_title="Final Score",
        yaxis_title="Sentiment Score",
        legend_title="Keyword",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[-1, 1])
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


# ============================================
# PART C — Emotion–Score Density Heatmap
# ============================================

st.header("🔥 Emotion–Score Density Heatmap")

required_cols_c = ["final_score", "sentiment_score", "keyword"]
if all(col in df_scores.columns for col in required_cols_c):

    df_density = df_scores.dropna(subset=["final_score", "sentiment_score"])

    keyword_list = sorted(df_density["keyword"].unique())
    selected_kw = st.multiselect(
        "Select keywords",
        options=keyword_list,
        default=keyword_list[: min(3, len(keyword_list))]
    )

    if selected_kw:
        df_density = df_density[df_density["keyword"].isin(selected_kw)]

    if df_density.empty:
        st.info("No data available.")
    else:
        fig_c = px.density_heatmap(
            df_density,
            x="final_score",
            y="sentiment_score",
            facet_col="keyword" if df_density["keyword"].nunique() > 1 else None,
            nbinsx=20,
            nbinsy=20,
            color_continuous_scale="Pinkyl",
            height=500
        )
        fig_c.update_layout(
            xaxis_title="Final Score",
            yaxis_title="Sentiment Score",
            coloraxis_colorbar_title="Density"
        )
        st.plotly_chart(fig_c, use_container_width=True)

else:
    st.warning("sentiment heatmap cannot be generated due to missing columns.")


# ============================
# PART D — WORD CLOUD
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
            colormap="Pastel1",
            max_words=50
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



# ===================== ROW 1 =====================
st.markdown("### Today’s News Landscape")

row1_col1, row1_col2 = st.columns([1.1, 1])

with row1_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("A. Article Score Scatter")
    # --- BEGIN: your current Part A block (scatter plot) ---
    # uses df_scores and plot_mixed_scatter(...)
    fig_a = plot_mixed_scatter(df_scores)
    if fig_a:
        st.plotly_chart(fig_a, use_container_width=True)
    with st.expander("View score table"):
        st.dataframe(df_scores)
    # --- END: Part A block ---
    st.markdown('</div>', unsafe_allow_html=True)

with row1_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("B. Sentiment Distribution")
    # --- BEGIN: your current Part B block (100% stacked bar) ---
    # assumes df_sent / sentiment_file already loaded above
    if os.path.exists(sentiment_file):
        df_sent = pd.read_csv(sentiment_file)
        # ... your stacked bar code ...
        st.pyplot(fig_b, use_container_width=True)
    else:
        st.warning("sentiment_statistics.csv not found.")
    # --- END: Part B block ---
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ===================== ROW 2 =====================
st.markdown("### Sentiment Structure & Keywords")

row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("C. Emotion–Score Density Heatmap")
    # --- BEGIN: your current Part C heatmap block ---
    # uses df_scores and px.density_heatmap(...)
    fig_c = px.density_heatmap(
        df_density,
        x="final_score",
        y="sentiment_score",
        facet_col="keyword" if df_density["keyword"].nunique() > 1 else None,
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Pinkyl",
        height=450,
    )
    fig_c.update_layout(
        xaxis_title="Final Score",
        yaxis_title="Sentiment Score",
        coloraxis_colorbar_title="Density",
    )
    st.plotly_chart(fig_c, use_container_width=True)
    # --- END: Part C block ---
    st.markdown('</div>', unsafe_allow_html=True)

with row2_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("D. Keyword Word Clouds")
    # --- BEGIN: your current Part D word cloud block ---
    if os.path.exists(words_file):
        # your word_count.csv loading + loops
        # showing 2 clouds per row with captions
        ...
    else:
        st.warning("word_count.csv not found.")
    # --- END: Part D block ---
    st.markdown('</div>', unsafe_allow_html=True)
