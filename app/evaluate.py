import pandas as pd

from search import search

queries = pd.read_csv("data/query.csv", sep="\t")
labels = pd.read_csv("data/label.csv", sep="\t")

print("Query columns:", list(queries.columns))

# Fast lookup: (query_id, product_id) -> label
label_lookup = {
    (row.query_id, row.product_id): row.label for row in labels.itertuples()
}

K = 5
strict_scores = []
lenient_scores = []

for row in queries.itertuples():
    results = search(row.query, top_k=K)
    found = [label_lookup.get((row.query_id, pid)) for pid in results["product_id"]]

    strict_scores.append(sum(label == "Exact" for label in found) / K)
    lenient_scores.append(sum(label in ("Exact", "Partial") for label in found) / K)

print(f"Queries evaluated: {len(queries)}")
print(f"Precision@{K} (Exact only):       {sum(strict_scores) / len(strict_scores):.3f}")
print(f"Precision@{K} (Exact + Partial):  {sum(lenient_scores) / len(lenient_scores):.3f}")