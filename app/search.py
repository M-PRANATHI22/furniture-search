import numpy as np
from sentence_transformers import SentenceTransformer

from clean import load_products, build_search_text

products = build_search_text(load_products())
embeddings = np.load("data/product_embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 128


def search(query, top_k=5):
    """Return the top_k products closest in meaning to the query."""
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    top_ids = np.argsort(-scores)[:top_k]

    results = products.iloc[top_ids][
        ["product_id", "product_name", "product_class"]
    ].copy()
    results["score"] = scores[top_ids].round(3)
    return results


if __name__ == "__main__":
    for query in ["small grey sofa", "comfortable wooden bed frame", "kids desk lamp"]:
        print(f"\nQuery: {query}")
        print(search(query).to_string(index=False))