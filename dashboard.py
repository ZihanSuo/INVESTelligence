import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from wordcloud import WordCloud

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

# --------------------------------------------------
# CSS Styling (no main-block wrapper)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        padding: 18px 20px 20px 20px;
        border-radius: 14px;
        background-color: #FFFFFF;   /* FIX: remove pink block */
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Color Palette
# --------------------------------------------------
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA",
    "#F4B6C2", "#F6E2B3", "#C7D8C6", "#D5E1DF", "#E4C1F9",
    "#F1C0E8", "#FDE2E4", "#FAD2E1", "#D8E2DC", "#ECE4DB"
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PASTEL_COLORS)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, "data")

subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]
latest_folder = sorted(subfolders)[-1]
date_label = os.path.basename(latest_folder)

st.success(f"📅 Displaying data for: **{date_label}**")

scores_file = os.path.join(latest_folder, "scores.csv")
words_file = os.path.join(latest_folder, "word_count.csv")
sentiment_file = os.path.join(latest_folder, "sentiment_statistics.csv")

df_scores = pd.read_csv(scores_file)

# --------------------------------------------------
# PART A — Scatter Plot (with filter)
# --------------------------------------------------
def plot_mixed_scatter(df):
    df_top10 = df.sort_values("final_score", ascending=False).groupby("keyword").head(10)

    df_top10["hover_text"] = (
        "<b>" + df_top10["title"] + "</b><br>"
        + "Final Score: " + df_top10["final_score"].astype(str) + "<br>"
        + "Sentiment: " + df_top10["sentiment_score"].astype(str) + "<br>"
        + "<a href='" + df_top10["url"] + "'>Open Source</a>"
    )

    fig = px.scatter(
        df_top10,
        x="final_score",
        y="sentiment_score",
        color="keyword",
        custom_data=["hover_text"],
        hover_name="title",
        opacity=0.85,
        color_discrete_sequence=PASTEL_COLORS,
        height=520,
    )

    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_layout(
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[-1, 1])
    )
    return fig

# UI for row 1
row1_col1, row1_col2 = st.columns([1.1, 1])

with row1_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("A. Article Score Scatter")

    keywords = sorted(df_scores["keyword"].unique())
    selected_kw_A = st.multiselect(
        "Select keywords:",
        options=keywords,
        default=keywords
    )
    df_A = df_scores[df_scores["keyword"].isin(selected_kw_A)]
    st.plotly_chart(plot_mixed_scatter(df_A), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PART B — Stacked Bar
# --------------------------------------------------
df_sent = pd.read_csv(sentiment_file)

with row1_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("B. Sentiment Distribution (100% Stacked)")

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


# --------------------------------------------------
# PART C — Density Heatmap
# --------------------------------------------------
df_density = df_scores.dropna(subset=["final_score", "sentiment_score"])

row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("C. Emotion–Score Density Heatmap")

    selected_kw_C = st.multiselect(
        "Select keywords:",
        options=keywords,
        default=keywords,
        key="heatmap_kw"
    )
    df_C = df_density[df_density["keyword"].isin(selected_kw_C)]

    fig_c = px.density_heatmap(
        df_C,
        x="final_score",
        y="sentiment_score",
        facet_col="keyword" if df_C["keyword"].nunique() > 1 else None,
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Pinkyl",
        height=450,
    )
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# PART D — Word Cloud
# --------------------------------------------------
df_words = pd.read_csv(words_file)

with row2_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("D. Keyword Word Clouds")

    wc_figs = []
    for kw in df_words["keyword"].unique():
        freq = dict(zip(df_words[df_words["keyword"] == kw]["word"],
                        df_words[df_words["keyword"] == kw]["count"]))

        wc = WordCloud(width=500, height=300, background_color="white",
                       colormap="Pastel1", max_words=80).generate_from_frequencies(freq)

        fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        wc_figs.append((kw, fig_wc))

    for i in range(0, len(wc_figs), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(wc_figs):
                kw, fig_wc = wc_figs[i+j]
                cols[j].pyplot(fig_wc)
                cols[j].markdown(f"<div style='text-align:center;'>{kw}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
