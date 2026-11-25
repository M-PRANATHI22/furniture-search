import pandas as pd
from sentence_transformers import SentenceTransformer

from clean import load_products, build_search_text

products = build_search_text(load_products())
queries = pd.read_csv("data/query.csv", sep="\t")
labels = pd.read_csv("data/label.csv", sep="\t")

# Shorter text: name + class + category only (no description or features)
short_text = (
    products["product_name"]
    + " "
    + products["product_class"]
    + " "
    + products["category hierarchy"]
).str.strip()

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 64

embeddings = model.encode(
    short_text.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

label_lookup = {
    (row.query_id, row.product_id): row.label for row in labels.itertuples()
}

K = 5
strict_scores = []
lenient_scores = []

for row in queries.itertuples():
    query_vec = model.encode([row.query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    top_ids = scores.argsort()[::-1][:K]
    product_ids = products.iloc[top_ids]["product_id"]

    found = [label_lookup.get((row.query_id, pid)) for pid in product_ids]
    strict_scores.append(sum(label == "Exact" for label in found) / K)
    lenient_scores.append(sum(label in ("Exact", "Partial") for label in found) / K)

print("--- Short text (name + class + category) ---")
print(f"Precision@{K} (Exact only):       {sum(strict_scores) / len(strict_scores):.3f}")
print(f"Precision@{K} (Exact + Partial):  {sum(lenient_scores) / len(lenient_scores):.3f}")