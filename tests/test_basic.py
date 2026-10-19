import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch
import main
import sentiment


def test_extract_video_id():
    assert fetch.extract_video_id("https://www.youtube.com/watch?v=abc123&t=5") == "abc123"
    assert fetch.extract_video_id("https://youtu.be/abc123?si=x") == "abc123"
    assert fetch.extract_video_id("https://www.youtube.com/shorts/abc123") == "abc123"
    assert fetch.extract_video_id("abc123") == "abc123"


def test_clean_text():
    assert main.clean_text("Great &amp; nice  https://x.com @bob") == "Great & nice"


def test_spam_detection():
    assert main.is_spam("Earn $500 daily with bitcoin")
    assert not main.is_spam("Bohat zabardast video hai")


def test_lexicon_roman_urdu():
    assert sentiment.lexicon_score("Bohat zabardast mashallah") > 0
    assert sentiment.lexicon_score("Bilkul bakwas hai") < 0
    assert sentiment.lexicon_score("acha nahi hai") < 0
    assert sentiment.lexicon_score("loooove it") > 0
    assert sentiment.lexicon_score("😍😍") > 0


def test_labels():
    assert sentiment.label_from_score(0.5) == "Positive"
    assert sentiment.label_from_score(-0.5) == "Negative"
    assert sentiment.label_from_score(0.0) == "Neutral"


def test_stance_rules():
    assert sentiment.apply_rules("Never buying Gillette again", 0.5) < -0.15
    assert sentiment.apply_rules("Will buy only Gillette from now on", 0.1) > 0.15
    assert sentiment.apply_rules("1:03 ITS THE FUNNY", 0.8) == 0.0
    assert sentiment.apply_rules("Great job lol", 0.6) == 0.0
    assert sentiment.apply_rules("🖕", 0.0) < 0


def test_llm_parse():
    raw = 'Here: [{"id": 1, "label": "negative", "confidence": 0.9}, {"id": 2, "label": "Neutral"}]'
    assert sentiment._parse_batch(raw, 2) == [("Negative", 0.9), ("Neutral", 1.0)]


def test_pipeline_vader_demo():
    from demo_data import generate_demo_comments

    df, spam = main.analyze_comments(generate_demo_comments(200), engine="vader")
    assert len(df) > 0
    assert set(df["sentiment"]) <= {"Positive", "Neutral", "Negative"}