import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")

with open("readme.md", "r", encoding="utf-8") as f:
    md_content = f.read()

st.markdown(md_content, unsafe_allow_html=True)
