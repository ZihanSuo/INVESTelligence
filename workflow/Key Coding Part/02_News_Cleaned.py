import re


# --- Helper Functions ---

def strip_urls(line: str) -> str:
    if not line: return ""
    line = re.sub(r'https?://\S+', '', line)
    return re.sub(r'www\.\S+', '', line)


def clean_text_content(text):
    if not text: return ""

    lines = text.split('\n')
    valid_lines = []

    # Noise phrases to filter out
    noise = [
        "futures", "/ ozt", "/ bbl", "brent crude", "sign in", "subscribe",
        "read more", "click here", "all rights reserved", "advertisement",
        "cookie policy", "skip navigation", "markets data"
    ]

    for line in lines:
        line = strip_urls(line)
        stripped = line.strip()

        if not stripped: continue

        # Filter noise
        if any(x in stripped.lower() for x in noise):
            continue

        # Filter short fragments (unless it ends with punctuation)
        if len(stripped) < 60 and not stripped.endswith(('.', '!', '?', '"', '”')):
            continue

        valid_lines.append(stripped)

    return "\n\n".join(valid_lines)


# --- Main Logic ---

items = _input.all()
output_list = []

# No seen_urls set here (Deduplication disabled as requested)

for item in items:
    # 1. Safe extraction (Fixes the AttributeError)
    source_data = item.get("json", {})
    if not source_data:
        continue

    # Convert to standard dict to allow modification
    news = dict(source_data)

    # 2. Clean Content
    # Try multiple fields in case one is missing
    raw_text = news.get("content") or news.get("raw_content") or ""
    cleaned_text = clean_text_content(raw_text)

    # 3. Quality Gate (Remove only if text is practically empty)
    # Still useful to remove pure ad pages, but won't remove duplicates
    if len(cleaned_text) < 30:
        continue

    # Update content in place
    news["content"] = cleaned_text

    output_list.append({"json": news})

return output_list