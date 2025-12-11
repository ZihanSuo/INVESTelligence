import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from datetime import datetime

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Financial News Agent",
    page_icon="📈",
    layout="wide"
)

DATA_ROOT = "./data"  # Assuming your folder structure starts here

@st.cache_data
def load_all_data(target_date=None):
    """
    Loads all DataFrames from the specified date folder (or the latest one).
    Also aggregates historical data for trend analysis.
    """
    
    # 1. Find the target folder (Default: Latest Date)
    # -----------------------------------------------------
    all_dates = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    if not all_dates:
        st.error("No data folders found in ./data!")
        return None, None, None, None, None, None

    # Use the latest date if none specified
    current_date = target_date if target_date else all_dates[-1]
    day_folder = os.path.join(DATA_ROOT, current_date)
    
    # 2. Load Daily DataFrames (Exactly matching your filenames)
    # -----------------------------------------------------
    try:
        # A. alpha.csv -> High priority signals (Level 2 Scatter)
        alpha_df = pd.read_csv(os.path.join(day_folder, "alpha.csv"))
        
        # B. dedupted_news.csv -> Full raw news list (Level 5 Table)
        news_df = pd.read_csv(os.path.join(day_folder, "dedupted_news.csv"))
        
        # C. scores.csv -> Detailed scoring breakdown (Maybe useful for deep dive)
        scores_df = pd.read_csv(os.path.join(day_folder, "scores.csv"))
        
        # D. word_count.csv -> Keyword distribution (Level 2 Treemap)
        word_count_df = pd.read_csv(os.path.join(day_folder, "word_count.csv"))
        
        # E. sentiment_statistics.csv -> Daily metrics (Level 1 Pulse)
        stats_df = pd.read_csv(os.path.join(day_folder, "sentiment_statistics.csv"))
        
        # F. entities.json -> Knowledge Graph (Level 4 Network)
        # Note: Using json.load because graph data is often nested, not tabular
        with open(os.path.join(day_folder, "entities.json")) as f:
            entities_data = json.load(f)

    except FileNotFoundError as e:
        st.error(f"Missing critical file in {day_folder}: {e}")
        return None
        
    # 3. Load History (Loop through previous folders for Level 3)
    # -----------------------------------------------------
    history_list = []
    for d in all_dates:
        stat_path = os.path.join(DATA_ROOT, d, "sentiment_statistics.csv")
        if os.path.exists(stat_path):
            try:
                # Assuming this CSV has columns like [date, avg_sentiment, total_articles]
                daily_stat = pd.read_csv(stat_path)
                daily_stat['date'] = d # Ensure date column exists
                history_list.append(daily_stat)
            except:
                continue
    
    if history_list:
        history_df = pd.concat(history_list, ignore_index=True)
    else:
        history_df = pd.DataFrame() # Empty fallback

    # Return everything as a dictionary or individual variables
    return {
        "current_date": current_date,
        "alpha_df": alpha_df,
        "news_df": news_df,
        "scores_df": scores_df,
        "word_count_df": word_count_df,
        "stats_df": stats_df,
        "entities_data": entities_data,
        "history_df": history_df
    }

# Execution
data_pack = load_all_data()
# ---------------------------------------------------------
# Level 1: Macro Snapshot (The Dashboard Header)
# ---------------------------------------------------------
st.title("💰 AI Financial News Dashboard")
st.markdown("---")

# CSS to style the metric containers slightly for better readability
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f0f2f6;
    border: 1px solid #e0e0e0;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Layout: Three columns for key metrics
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="Total Articles Analyzed",
        value=metrics['total_articles'],
        delta="12 New"  # Example: Change since last run
    )

with c2:
    sentiment_val = metrics['avg_sentiment']
    # Dynamic coloring: Inverse usually means Red=Negative, Green=Positive
    delta_color = "normal" if abs(sentiment_val) < 0.1 else ("inverse" if sentiment_val < 0 else "normal") 
    st.metric(
        label="Market Sentiment Pulse",
        value=f"{sentiment_val:+.2f}",
        delta="Bullish" if sentiment_val > 0.2 else ("Bearish" if sentiment_val < -0.2 else "Neutral"),
        delta_color=delta_color
    )

with c3:
    st.metric(
        label="Top Trending Theme",
        value=metrics['top_keyword'],
        delta="High Conviction"
    )

st.markdown("---")

# ---------------------------------------------------------
# Level 2: Core Signal Discovery (The Alpha Layer)
# ---------------------------------------------------------
st.subheader("🎯 Alpha Discovery Layer")

# Layout: Left 2/3 (Scatter Plot), Right 1/3 (Treemap)
col_left, col_right = st.columns([2, 1])

# --- Left Column: The Alpha Matrix (Scatter Plot) ---
with col_left:
    st.markdown("#### 1. Signal-to-Noise Matrix (Alpha Quadrant)")
    st.caption("X-Axis: Importance (Score) | Y-Axis: Sentiment (Direction)")
    
    # Construct Custom Hover Text
    df["hover_text"] = (
        "<b>" + df["title"] + "</b><br>" +
        "Source: " + df["source"] + "<br>" +
        "Score: " + df["final_score"].astype(str)
    )

    # Plotly Scatter Plot
    fig_scatter = px.scatter(
        df,
        x="final_score",
        y="sentiment_score",
        color="keyword",
        size="word_count",   # Bubble size represents article length/depth
        hover_name="title",
        custom_data=["source", "final_score"], # Data available for hover template
        color_discrete_sequence=px.colors.qualitative.G10,
        height=500
    )

    # Add Reference Lines for Decision Making
    # Horizontal line at 0 (Neutral Sentiment)
    fig_scatter.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    # Vertical line at 75 (High Importance Threshold)
    fig_scatter.add_vline(x=75, line_width=1, line_dash="dash", line_color="red", opacity=0.5, annotation_text="High Priority")

    # Optimize Axes
    fig_scatter.update_layout(
        xaxis_title="Relevance / Materiality Score",
        yaxis_title="Sentiment Score (-1 to +1)",
        xaxis=dict(range=[40, 105]), # Focusing on mid-to-high relevance
        yaxis=dict(range=[-1.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Right Column: Market Attention Map (Treemap) ---
with col_right:
    st.markdown("#### 2. Market Attention Map")
    st.caption("Size: Content Volume | Color: Sentiment")

    # Plotly Treemap
    fig_tree = px.treemap(
        df,
        path=[px.Constant("Market"), 'keyword', 'title'], # Hierarchy: All -> Keyword -> Article
        values='word_count',      # Tile Area = Information Depth
        color='sentiment_score',  # Tile Color = Sentiment
        color_continuous_scale='RdBu', # Red-Blue Diverging Scale
        color_continuous_midpoint=0,
        height=500
    )
    
    # Clean up UI for the Treemap
    fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig_tree.data[0].textinfo = 'label+text' # Show text labels
    
    st.plotly_chart(fig_tree, use_container_width=True)
