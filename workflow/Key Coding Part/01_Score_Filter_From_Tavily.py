import json
import re

# Access n8n input data
items = _input.all()
output_items = []

# Filter threshold
MIN_SCORE_THRESHOLD = 0.1  ####For Broad Search, the threshold is 0.3.

for item in items:
    try:
        # Access payload
        node_data = item.get("json", {})
        parent_query = node_data.get("query") or node_data.get("seed", "unknown_query")

        # Get raw results
        results_list = node_data.get("results", [])

        # Handle single item structure
        if not results_list and "url" in node_data:
            results_list = [node_data]

        filtered_results = []

        for result in results_list:
            score = float(result.get("score", 0))

            # Skip low quality data
            if score < MIN_SCORE_THRESHOLD:
                continue

            if parent_query:
                result["query"] = parent_query

            filtered_results.append(result)

        # Output flattened valid results
        if filtered_results:
            for valid_res in filtered_results:
                output_items.append({"json": valid_res})

    except Exception:
        continue

return output_items