import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="INVESTelligence", layout="wide")
st.title("📊 INVESTelligence Dashboard")

# Main container styling (width + margins)
st.markdown(
    """
    <style>
    .main-block {
        max-width: 1300px;
        margin: 0 auto;
        padding: 10px 20px 40px 20px;
    }
    </style>
    <div class="main-block">
    """,
    unsafe_allow_html=True,
)

# Global pastel palette for matplotlib
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#FF9AA2", "#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA",
    "#F4B6C2", "#F6E2B3", "#C7D8C6", "#D5E1DF", "#E4C1F9",
    "#F1C0E8", "#FDE2E4", "#FAD2E1", "#D8E2DC", "#ECE4DB",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PASTEL_COLORS)

# -------------------------
# Find latest data folder
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, "data")

if not os.path.exists(data_root):
    st.error("Data folder not found!")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]
if not subfolders:
    st.warning("No data folders found.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

latest_folder = sorted(subfolders)[-1]
date_label = os.path.basename(latest_folder)
st.success(f"📅 Displaying data for: **{date_label}**")

scores_file = os.path.join(latest_folder, "scores.csv")
words_file = os.path.join(latest_folder, "word_count.csv")
sentiment_file = os.path.join(latest_folder, "sentiment_statistics.csv")

# ============================================================
# PART A — Mixed scatter (Top 10 per keyword)
# ============================================================

def plot_mixed_scatter(df_scores: pd.DataFrame):
    required_cols = ["keyword", "title", "final_score", "sentiment_score", "url"]
    for col in required_cols:
        if col not in df_scores.columns:
            st.error(f"Missing column in scores.csv: {col}")
            return None

    df_top10 = (
        df_scores.sort_values("final_score", ascending=False)
        .groupby("keyword")
        .head(10)
        .copy()
    )

    # Scale final score to 0–100
    df_top10["final_score_pct"] = df_top10["final_score"] * 100

    df_top10["hover_text"] = (
        "<b>" + df_top10["title"] + "</b><br>"
        + "Final Score: " + df_top10["final_score_pct"].round(1).astype(str) + "<br>"
        + "Sentiment: " + df_top10["sentiment_score"].round(2).astype(str) + "<br>"
        + "<a href='" + df_top10["url"] + "'>Open Source</a>"
    )

    fig = px.scatter(
        df_top10,
        x="final_score_pct",
        y="sentiment_score",
        color="keyword",
        hover_name="title",
        custom_data=["hover_text"],
        color_discrete_sequence=PASTEL_COLORS,
        size_max=12,
        opacity=0.85,
        height=500,
    )

    fig.update_traces(hovertemplate="%{customdata[0]}")

    fig.update_xaxes(title="Final Score (0–100)", range=[0, 100])
    fig.update_yaxes(title="Sentiment Score (−1 to 1)", range=[-1, 1])

    fig.update_layout(
        title="A. Article Score Scatter (Top 10 per Keyword)",
        legend_title="Keyword",
    )

    return fig


if os.path.exists(scores_file):
    df_scores = pd.read_csv(scores_file)

    st.header("A. Article Score Insights")
    st.write(f"Total articles: **{len(df_scores)}**")

    fig_a = plot_mixed_scatter(df_scores)
    if fig_a is not None:
        st.plotly_chart(fig_a, use_container_width=True)

    with st.expander("View raw scores data"):
        st.dataframe(df_scores)
else:
    df_scores = None
    st.warning("scores.csv not found.")

# ============================================================
# PART B — Sentiment distribution (100% stacked bar)
# ============================================================

st.header("B. Sentiment Distribution by Keyword (100% stacked)")

if os.path.exists(sentiment_file):
    df_sent = pd.read_csv(sentiment_file)

    required_cols = [
        "keyword",
        "strong_neg",
        "weak_neg",
        "neutral",
        "weak_pos",
        "strong_pos",
    ]
    for col in required_cols:
        if col not in df_sent.columns:
            st.error(f"Missing column in sentiment_statistics.csv: {col}")
            break
    else:
        df_plot = df_sent.copy().sort_values("keyword")

        categories = [
            "strong_neg",
            "weak_neg",
            "neutral",
            "weak_pos",
            "strong_pos",
        ]

        x = np.arange(len(df_plot["keyword"]))
        bottom = np.zeros(len(df_plot))

        fig_b, ax = plt.subplots(figsize=(10, 4))
        for cat in categories:
            values = df_plot[cat].astype(float)
            ax.bar(x, values, bottom=bottom, label=cat)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["keyword"], rotation=30, ha="right")
        ax.set_ylabel("Percentage")
        ax.set_title("Sentiment share per keyword")
        ax.legend(loc="upper right", fontsize=8)

        st.pyplot(fig_b)
else:
    st.warning("sentiment_statistics.csv not found.")

# ============================================================
# PART C — Emotion–Score density heatmap
# ============================================================

st.header("C. Emotion–Score Density Heatmap")

if df_scores is not None:
    needed_cols = ["final_score", "sentiment_score", "keyword"]
    if all(col in df_scores.columns for col in needed_cols):
        df_density = df_scores.dropna(subset=["final_score", "sentiment_score"]).copy()
        df_density["final_score_pct"] = df_density["final_score"] * 100

        keyword_list = sorted(df_density["keyword"].unique())
        selected_kw = st.multiselect(
            "Select keywords",
            options=keyword_list,
            default=keyword_list[: min(3, len(keyword_list))],
        )

        if selected_kw:
            df_density = df_density[df_density["keyword"].isin(selected_kw)]

        if df_density.empty:
            st.info("No data available for the selected keywords.")
        else:
            fig_c = px.density_heatmap(
                df_density,
                x="final_score_pct",
                y="sentiment_score",
                facet_col="keyword"
                if df_density["keyword"].nunique() > 1
                else None,
                nbinsx=20,
                nbinsy=20,
                color_continuous_scale="Pinkyl",
                height=500,
            )

            fig_c.update_xaxes(title="Final Score (0–100)", range=[0, 100])
            fig_c.update_yaxes(title="Sentiment Score (−1 to 1)", range=[-1, 1])
            fig_c.update_layout(coloraxis_colorbar_title="Density")

            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.warning("Heatmap cannot be generated: missing columns in scores.csv.")
else:
    st.info("Heatmap not available because scores.csv is missing.")

# ============================================================
# PART D — Word clouds
# ============================================================

st.header("D. Keyword Word Clouds")

if os.path.exists(words_file):
    df_words = pd.read_csv(words_file)

    required_cols_wc = ["keyword", "word", "count"]
    for col in required_cols_wc:
        if col not in df_words.columns:
            st.error(f"Missing column in word_count.csv: {col}")
            df_words = None
            break

    if df_words is not None:
        keywords = df_words["keyword"].unique().tolist()
        wc_images = []

        for kw in keywords:
            df_kw = df_words[df_words["keyword"] == kw]
            freq_dict = dict(zip(df_kw["word"], df_kw["count"]))

            wc = WordCloud(
                width=500,
                height=300,
                background_color="white",
                colormap="Pastel1",
                max_words=50,
            ).generate_from_frequencies(freq_dict)

            fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            wc_images.append((kw, fig_wc))

        # Show 2 per row
        for i in range(0, len(wc_images), 2):
            cols = st.columns(2)
            for j in range(23):
                if i + j < len(wc_images):
                    kw, fig_wc = wc_images[i + j]
                    with cols[j]:
                        st.pyplot(fig_wc)
                        st.markdown(
                            f"<div style='text-align:center; color:#555;'>{kw}</div>",
                            unsafe_allow_html=True,
                        )
else:
    st.warning("word_count.csv not found.")

# Close main-block div
st.markdown("</div>", unsafe_allow_html=True)
