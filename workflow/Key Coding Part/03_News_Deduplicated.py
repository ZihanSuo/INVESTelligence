from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

items = _input.all()
articles = [item["json"] for item in items]

unique_news = []

for i in range(len(articles)):
    current = articles[i]

    if current.get("is_duplicate"):
        continue

    current["pickup_count"] = 0

    title1 = current.get("title","").lower()

    for j in range(i + 1, len(articles)):
        compare = articles[j]

        if compare.get("is_duplicate"):
            continue

        title2 = compare.get("title","").lower()

        if similarity(title1, title2) > 0.55:
            current["pickup_count"] += 1
            compare["is_duplicate"] = True


    current.pop("is_duplicate", None)

    unique_news.append(current)

return [{"json": item} for item in unique_news]