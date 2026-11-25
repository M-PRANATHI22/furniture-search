from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "solid wood platform bed",
    "comfortable wooden bed frame",
    "stainless steel slow cooker",
]

embeddings = model.encode(sentences)
print(embeddings.shape)

scores = util.cos_sim(embeddings[0], embeddings[1:])
print(scores)