import pandas as pd


def load_products(path="data/product.csv"):
    """Load the product file (it is tab-separated)."""
    return pd.read_csv(path, sep="\t")


def clean_text(value):
    """Turn a value into clean lowercase text. Missing values become empty text."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def build_search_text(products):
    """Join the useful text columns into one searchable text per product."""
    text_columns = [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
    ]
    cleaned = products.copy()
    for column in text_columns:
        cleaned[column] = cleaned[column].apply(clean_text)

    cleaned["search_text"] = (
        cleaned[text_columns].agg(" ".join, axis=1).str.strip()
    )

    # Missing ratings mean the product has no reviews yet, so use 0
    for column in ["rating_count", "average_rating", "review_count"]:
        cleaned[column] = cleaned[column].fillna(0)

    return cleaned


if __name__ == "__main__":
    products = load_products()
    cleaned = build_search_text(products)
    print(cleaned[["product_id", "search_text"]].head())
    print(cleaned["search_text"].str.len().describe())