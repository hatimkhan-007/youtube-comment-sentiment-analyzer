import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import analyze_comments, classify_sentiment, clean_text, extract_video_id


def test_extract_video_id_variants():
    assert extract_video_id("https://www.youtube.com/watch?v=abc123&t=5s") == "abc123"
    assert extract_video_id("https://youtu.be/abc123?si=x") == "abc123"
    assert extract_video_id("https://www.youtube.com/shorts/abc123") == "abc123"
    assert extract_video_id("abc123") == "abc123"


def test_classification_labels():
    assert classify_sentiment("This is absolutely wonderful, I love it!")[0] == "Positive"
    assert classify_sentiment("This is terrible and I hate it.")[0] == "Negative"
    assert classify_sentiment("The video is ten minutes long.")[0] == "Neutral"


def test_clean_text_removes_urls_and_entities():
    assert clean_text("Great &amp; useful https://example.com/x  stuff") == "Great & useful stuff"


def test_analyze_drops_empty_and_duplicates():
    rows = [
        {"comment": "Great video!"},
        {"comment": "Great video!"},
        {"comment": "   "},
        {"comment": "https://example.com"},
        {"comment": "Awful."},
    ]
    df = analyze_comments(rows)
    assert len(df) == 2
    assert set(df["sentiment"]) == {"Positive", "Negative"}
