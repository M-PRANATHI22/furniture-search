from search import search


def test_returns_requested_number_of_results():
    results = search("wooden bed frame", top_k=3)
    assert len(results) == 3


def test_results_are_sorted_best_first():
    scores = list(search("wooden bed frame", top_k=5)["score"])
    assert scores == sorted(scores, reverse=True)


def test_bed_query_returns_bed_products():
    results = search("wooden bed frame", top_k=5)
    assert results["product_name"].str.contains("bed").any()