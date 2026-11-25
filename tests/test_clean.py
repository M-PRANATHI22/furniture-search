from clean import build_search_text, clean_text, load_products


def test_clean_text_lowercases_and_removes_extra_spaces():
    assert clean_text("  Solid   WOOD Bed ") == "solid wood bed"


def test_clean_text_turns_missing_values_into_empty_text():
    assert clean_text(float("nan")) == ""
    assert clean_text(None) == ""


def test_build_search_text_keeps_every_row():
    products = load_products()
    cleaned = build_search_text(products)
    assert len(cleaned) == len(products)


def test_every_product_has_search_text():
    cleaned = build_search_text(load_products())
    assert (cleaned["search_text"].str.len() > 0).all()


def test_missing_ratings_are_filled_with_zero():
    cleaned = build_search_text(load_products())
    ratings = cleaned[["rating_count", "average_rating", "review_count"]]
    assert ratings.isna().sum().sum() == 0