import json

from src.data.gdelt_news import build_query_url, normalize_articles


def test_build_gdelt_url_and_normalize_metadata() -> None:
    url = build_query_url("中国 股票", max_records=10)
    assert "mode=artlist" in url
    assert "maxrecords=10" in url
    frame = normalize_articles([{
        "url": "https://example.com/a",
        "title": "公司业绩增长",
        "seendate": "20240102120000",
        "domain": "example.com",
        "language": "Chinese",
        "sourcecountry": "China",
    }])
    assert list(frame["article_id"]) == ["https://example.com/a"]
    assert frame.loc[0, "body"] == ""
    assert frame.loc[0, "data_source"] == "gdelt_doc_api"
