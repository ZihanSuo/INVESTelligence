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

# -------------------------------------------------------
# Load data from today's folder
# -------------------------------------------------------

# Auto-detect today's folder
today = datetime.today().strftime("%Y-%m-%d")
data_path = f"data/2025-12-10"  #####################################################3

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
# Prepare marker size for scatter plot
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
