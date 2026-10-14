import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def extract_video_id(video_url_or_id: str) -> str:
    """Accept a YouTube video ID or common YouTube URL."""
    value = video_url_or_id.strip()

    if "youtu.be/" in value:
        return value.split("youtu.be/", 1)[1].split("?", 1)[0].split("/", 1)[0]

    if "youtube.com" in value:
        if "v=" in value:
            return value.split("v=", 1)[1].split("&", 1)[0]
        for marker in ("/shorts/", "/embed/"):
            if marker in value:
                return value.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]

    return value.split("?", 1)[0].split("/", 1)[0]


def fetch_comments(video_id: str, max_comments: int = 500) -> list[dict]:
    """Fetch up to max_comments top-level comments from a public video."""
    if not API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is missing. Add it to a .env file in the project folder."
        )

    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="time",
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment": snippet.get("textDisplay", ""),
                "author": snippet.get("authorDisplayName", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def classify_sentiment(comment: str) -> tuple[str, float]:
    """Return a sentiment label and VADER compound score."""
    compound = analyzer.polarity_scores(comment)["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound


def analyze_comments(comments: list[dict]) -> pd.DataFrame:
    """Clean comment text and add sentiment labels and scores."""
    df = pd.DataFrame(comments)

    if df.empty:
        return pd.DataFrame(columns=[
            "comment", "author", "published_at", "like_count",
            "sentiment", "sentiment_score"
        ])

    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df = df[df["comment"] != ""].drop_duplicates(subset=["comment"]).copy()

    results = df["comment"].apply(classify_sentiment)
    df["sentiment"] = results.apply(lambda result: result[0])
    df["sentiment_score"] = results.apply(lambda result: result[1])

    return df.reset_index(drop=True)


def save_results(df: pd.DataFrame) -> None:
    """Export classified comments and a sentiment chart."""
    csv_path = OUTPUT_DIR / "classified_comments.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nClassified comments saved to: {csv_path}")

    if df.empty:
        print("No comments were available to chart.")
        return

    counts = df["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )

    ax = counts.plot(kind="bar", rot=0)
    ax.set_title("YouTube Comment Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of comments")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "sentiment_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Sentiment chart saved to: {chart_path}")

    print("\nSentiment summary:")
    for sentiment, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment}: {count} comments ({percentage:.1f}%)")


def main() -> None:
    print("=" * 50)
    print("YouTube Comment Sentiment Analyzer")
    print("=" * 50)

    video_input = input("\nEnter a YouTube video URL or ID: ").strip()
    if not video_input:
        print("No video URL or ID provided. Exiting.")
        return

    video_id = extract_video_id(video_input)

    try:
        max_input = input("Maximum comments to fetch (default 500): ").strip()
        max_comments = int(max_input) if max_input else 500

        if max_comments < 1:
            print("Please enter a number greater than zero.")
            return

        print("\nFetching comments...")
        comments = fetch_comments(video_id, max_comments)
        print(f"Fetched {len(comments)} comments.")

        if not comments:
            print("No comments returned. Comments may be disabled or unavailable.")
            return

        print("Analyzing sentiment...")
        results = analyze_comments(comments)
        save_results(results)
        print("\nDone!")

    except ValueError as error:
        print(f"\nConfiguration or input error: {error}")
    except HttpError as error:
        print(f"\nYouTube API error: {error}")
        print(
            "Check your API key, API quota, video ID, and whether comments "
            "are available for the video."
        )
    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
