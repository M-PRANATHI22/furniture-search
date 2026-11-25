from fastapi import FastAPI

from search import search

app = FastAPI(title="Home Goods Semantic Search")


@app.get("/")
def home():
    return {"message": "Semantic search API is running. Try /search?q=wooden bed frame"}


@app.get("/search")
def search_products(q: str, top_k: int = 5):
    results = search(q, top_k=top_k)
    return results.to_dict(orient="records")