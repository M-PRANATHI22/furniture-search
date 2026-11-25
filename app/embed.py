import numpy as np
from sentence_transformers import SentenceTransformer

from clean import load_products, build_search_text

products = build_search_text(load_products())

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 128

embeddings = model.encode(
    products["search_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

np.save("data/product_embeddings.npy", embeddings)
print(embeddings.shape)