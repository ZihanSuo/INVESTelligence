import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from wordcloud import WordCloud

# Page Setup
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
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 20px;
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

# Pastel color palette
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA",
    "#F4B6C2", "#F6E2B3", "#C7D8C6", "#D5E1DF", "#E4C1F9",
    "#F1C0E8", "#FDE2E4", "#FAD2E1", "#D8E2DC", "#ECE4DB",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PASTEL_COLORS)

# ====================================================
# Load Latest Data
# ====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, "data")

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

scores_file = os.path.join(latest_folder, "scores.csv")
words_file = os.path.join(latest_folder, "word_count.csv")
sentiment_file = os.path.join(latest_folder, "sentiment_statistics.csv")

# ====================================================
# Prepare PART A — Mixed Scatter Plot
# ====================================================
def plot_mixed_scatter(df_scores):
    df_top10 = (
        df_scores.sort_values("final_score", ascending=False)
        .groupby("keyword").head(10)
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
        opacity=0.85,
        height=520,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_layout(
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[-1, 1]),
        title="Mixed Scatter Plot (Top 10 Articles per Keyword)",
    )
    return fig


# Load scores
df_scores = None
if os.path.exists(scores_file):
    df_scores = pd.read_csv(scores_file)

    # ------------------------------
    # FILTER INSERTED HERE (Part A)
    # ------------------------------
    keywords_available = sorted(df_scores["keyword"].unique())
    selected_keywords_a = st.multiselect(
        "Filter keywords for Scatter Plot",
        options=keywords_available,
        default=keywords_available,
        key="filter_scatter"
    )
    df_scores_filtered_a = df_scores[df_scores["keyword"].isin(selected_keywords_a)]

    fig_a = plot_mixed_scatter(df_scores_filtered_a)

else:
    st.error("scores.csv not found.")
    st.stop()


# ====================================================
# Prepare PART B — 100% Stacked Bar
# ====================================================
fig_b = None
if os.path.exists(sentiment_file):
    df_sent = pd.read_csv(sentiment_file)

    categories = ["strong_neg", "weak_neg", "neutral", "weak_pos", "strong_pos"]
    df_plot = df_sent.sort_values("keyword")
    x = np.arange(len(df_plot["keyword"]))
    bottom = np.zeros(len(df_plot))

    fig_b, ax_b = plt.subplots(figsize=(10, 5))
    for cat in categories:
        values = df_plot[cat].astype(float)
        ax_b.bar(x, values, bottom=bottom, label=cat)
        bottom += values

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(df_plot["keyword"], rotation=30, ha="right")
    ax_b.set_title("Sentiment Distribution Across Keywords")
    ax_b.set_ylabel("Percentage")
    ax_b.legend(fontsize=8)
else:
    st.warning("sentiment_statistics.csv not found.")

# ====================================================
# Prepare PART C — Density Heatmap
# ====================================================
df_density = df_scores.dropna(subset=["final_score", "sentiment_score"])

# ------------------------------
# FILTER INSERTED HERE (Part C)
# ------------------------------
keywords_density = sorted(df_density["keyword"].unique())
selected_keywords_c = st.multiselect(
    "Filter keywords for Density Heatmap",
    options=keywords_density,
    default=keywords_density,
    key="filter_density"
)
df_density_filtered = df_density[df_density["keyword"].isin(selected_keywords_c)]

fig_c = px.density_heatmap(
    df_density_filtered,
    x="final_score",
    y="sentiment_score",
    facet_col="keyword" if df_density_filtered["keyword"].nunique() > 1 else None,
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

# ====================================================
# Prepare PART D — Word Clouds
# ====================================================
wc_figs = []
if os.path.exists(words_file):
    df_words = pd.read_csv(words_file)
    keywords_wc = df_words["keyword"].unique().tolist()

    for kw in keywords_wc:
        df_kw = df_words[df_words["keyword"] == kw]
        freq = dict(zip(df_kw["word"], df_kw["count"]))

        wc = WordCloud(
            width=500, height=300, background_color="white",
            colormap="Pastel1", max_words=80
        ).generate_from_frequencies(freq)

        fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        wc_figs.append((kw, fig_wc))


# ====================================================
# ROW 1 — A + B
# ====================================================
st.markdown("### Today’s News Landscape")
row1_col1, row1_col2 = st.columns([1.1, 1])

with row1_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("A. Article Score Scatter")
    st.plotly_chart(fig_a, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row1_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("B. Sentiment Distribution")
    if fig_b:
        st.pyplot(fig_b, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ====================================================
# ROW 2 — C + D
# ====================================================
st.markdown("### Sentiment Structure & Keywords")
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("C. Emotion–Score Density Heatmap")
    st.plotly_chart(fig_c, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("D. Keyword Word Clouds")

    if wc_figs:
        for i in range(0, len(wc_figs), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(wc_figs):
                    kw, fig_wc = wc_figs[i + j]
                    cols[j].pyplot(fig_wc)
                    cols[j].markdown(
                        f"<div style='text-align:center; color:#666;'>{kw}</div>",
                        unsafe_allow_html=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
