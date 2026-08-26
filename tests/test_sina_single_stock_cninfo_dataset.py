import pandas as pd

from scripts.build_sina_single_stock_cninfo_dataset import build_single_stock_dataset


def test_build_single_stock_dataset_filters_and_reindexes_in_panel_order():
    clean = pd.DataFrame({
        "article_id": ["multi", "b", "a"],
        "stock_id": ["000003", "000002", "000001"],
        "is_multi_stock_article": [True, False, False],
        "text": ["m", "b", "a"],
    })
    panel = pd.DataFrame({
        "row_index": [30, 20, 10],
        "article_id": ["multi", "b", "a"],
        "stock_id": ["000003", "000002", "000001"],
        "is_multi_stock_article": [True, False, False],
        "next_day_return": [0.0, 0.1, -0.1],
        "event_return_3d": [0.0, 0.2, -0.2],
    })

    result_clean, result_panel = build_single_stock_dataset(clean, panel)

    assert result_panel["article_id"].tolist() == ["a", "b"]
    assert result_clean["article_id"].tolist() == ["a", "b"]
    assert result_panel["row_index"].tolist() == [1, 2]
    assert result_panel["source_row_index"].tolist() == [10, 20]
    assert not result_panel["is_multi_stock_article"].any()


def test_build_single_stock_dataset_rejects_different_article_sets():
    clean = pd.DataFrame({
        "article_id": ["a"], "stock_id": ["000001"],
        "is_multi_stock_article": [False],
    })
    panel = pd.DataFrame({
        "article_id": ["b"], "stock_id": ["000002"],
        "is_multi_stock_article": [False],
    })

    try:
        build_single_stock_dataset(clean, panel)
    except ValueError as error:
        assert "article sets differ" in str(error)
    else:
        raise AssertionError("different article sets should fail")
