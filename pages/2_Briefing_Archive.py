import streamlit as st
import streamlit.components.v1 as components
import os

# --- CONFIGURATION: REPLACE THIS PATH ---
ARCHIVE_FOLDER = 'newsletter_archive' 

def render_html_archive():
    st.header("📚 Briefing Archive")

    # 1. Validation: Check if folder exists
    if not os.path.exists(ARCHIVE_FOLDER):
        st.error(f"Error: The folder '{ARCHIVE_FOLDER}' does not exist.")
        return

    # 2. Get File List (Filtered for .html)
    try:
        # Filter strictly for .html files
        files = [f for f in os.listdir(ARCHIVE_FOLDER) if f.endswith('.html')]
    except Exception as e:
        st.error(f"Error reading directory: {e}")
        return

    # 3. Sort Files (Newest first)
    # Assumes filenames follow a date pattern like YYYY-MM-DD.html
    files.sort(reverse=True)

    if not files:
        st.warning(f"No .html briefing files found in '{ARCHIVE_FOLDER}'.")
        return

    # 4. Dropdown Menu (Sidebar)
    # This creates the index/navigation in the sidebar
    selected_file = st.sidebar.selectbox(
        "🗄️ Select Past Briefing:",
        options=files,
        index=0
    )

    # 5. Load and Render HTML
    file_path = os.path.join(ARCHIVE_FOLDER, selected_file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        st.subheader(f"📄 Viewing: {selected_file}")
        st.divider()
        
        # Using streamlit components to render raw HTML safely
        # height=800 sets the scrollable area height
        # scrolling=True ensures you can read long newsletters
        components.html(html_content, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    render_html_archive()
