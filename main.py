import argparse
import html
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

import eda
import sentiment

load_dotenv()

MIN_REQUIRED_COMMENTS = 2000
DEFAULT_TARGET = 2500
LABELS = ["Positive", "Neutral", "Negative"]
LABEL_COLORS = {"Positive": "#2e9e5b", "Neutral": "#8c8c8c", "Negative": "#d64545"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_video_id(video_url_or_id: str) -> str:
    """Accept a YouTube video ID or common YouTube URL."""
    value = video_url_or_id.strip()

    if "youtu.be/" in value:
        return value.split("youtu.be/", 1)[1].split("?", 1)[0].split("/", 1)[0]

    if "youtube.com" in value:
        if "v=" in value:
            return value.split("v=", 1)[1].split("&", 1)[0]
        for marker in ("/shorts/", "/embed/", "/live/"):
            if marker in value:
                return value.split(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]

    return value.split("?", 1)[0].split("/", 1)[0]


def fetch_comments(video_id: str, max_comments: int, order: str = "relevance") -> list[dict]:
    """Fetch up to max_comments top-level comments from a public video."""
    from googleapiclient.discovery import build

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY is missing. Add it to a .env file in the project folder."
        )

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order=order,
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment": snippet.get("textDisplay", ""),
                "author": snippet.get("authorDisplayName", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def collect_from_videos(video_ids: list[str], target: int, order: str) -> list[dict]:
    """Collect comments across one or more videos until the target is reached."""
    collected: list[dict] = []
    for video_id in video_ids:
        remaining = target - len(collected)
        if remaining <= 0:
            break
        print(f"Fetching from video {video_id} (need {remaining} more)...")
        batch = fetch_comments(video_id, remaining, order)
        print(f"  got {len(batch)} comments")
        collected.extend(batch)
    return collected


def clean_text(text: str) -> str:
    """Normalize a raw comment: unescape HTML, drop URLs, collapse whitespace."""
    text = html.unescape(str(text))
    text = URL_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def analyze_comments(comments: list[dict], engine: str = "xlmr") -> pd.DataFrame:
    """Clean comment text and add sentiment labels and scores."""
    columns = [
        "video_id", "comment", "clean_comment", "author", "published_at",
        "like_count", "reply_count", "sentiment", "sentiment_score",
    ]
    df = pd.DataFrame(comments)

    if df.empty:
        return pd.DataFrame(columns=columns)

    for column in ("video_id", "author", "published_at"):
        if column not in df:
            df[column] = ""
    for column in ("like_count", "reply_count"):
        if column not in df:
            df[column] = 0

    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["clean_comment"] = df["comment"].apply(clean_text)
    df = df[df["clean_comment"] != ""]
    df = df.drop_duplicates(subset=["clean_comment"]).copy()

    df["sentiment_score"] = sentiment.score_texts(df["clean_comment"].tolist(), engine)
    df["sentiment"] = df["sentiment_score"].apply(sentiment.label_from_score)

    return df[columns].reset_index(drop=True)


def plot_distribution(df: pd.DataFrame, path: Path) -> pd.Series:
    counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=[LABEL_COLORS[l] for l in counts.index])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count}\n({count / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title(f"YouTube Comment Sentiment Distribution (n={total})")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of comments")
    ax.set_ylim(0, max(counts.max(), 1) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return counts


def save_results(df: pd.DataFrame, output_dir: Path, extra_stopwords: set[str]) -> None:
    """Export classified comments, the sentiment chart and EDA outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classified_comments.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nClassified comments saved to: {csv_path}")

    if df.empty:
        print("No comments were available to chart.")
        return

    counts = plot_distribution(df, output_dir / "sentiment_distribution.png")
    print(f"Sentiment chart saved to: {output_dir / 'sentiment_distribution.png'}")

    eda.run_eda(df, output_dir, extra_stopwords)
    print(f"EDA outputs saved to: {output_dir}")

    print("\nSentiment summary:")
    for label, count in counts.items():
        print(f"{label}: {count} comments ({count / len(df) * 100:.1f}%)")

    if len(df) < MIN_REQUIRED_COMMENTS:
        print(
            f"\nWARNING: only {len(df)} unique comments after cleaning; "
            f"the project requires at least {MIN_REQUIRED_COMMENTS}. "
            "Add more videos with --videos or raise --max-comments."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube Comment Sentiment Analyzer")
    parser.add_argument("--videos", nargs="+", help="One or more YouTube video URLs or IDs")
    parser.add_argument(
        "--max-comments", type=int, default=DEFAULT_TARGET,
        help=f"Total comments to collect across all videos (default {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--order", choices=["relevance", "time"], default="relevance",
        help="Comment ordering requested from the API (default relevance)",
    )
    parser.add_argument(
        "--engine", choices=sentiment.ENGINES, default="xlmr",
        help="xlmr = multilingual transformer (English, Roman Urdu, Urdu, Hindi, etc.); vader = English only",
    )
    parser.add_argument(
        "--stopwords", nargs="*", default=[],
        help="Extra words to hide from the top-words chart (brand names, channel names)",
    )
    parser.add_argument("--output-dir", default="output", help="Folder for results")
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic offline dataset to test the pipeline (no API key needed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("YouTube Comment Sentiment Analyzer")
    print("=" * 50)

    output_dir = Path(args.output_dir)

    try:
        if args.demo:
            from demo_data import generate_demo_comments

            output_dir = Path("output_demo") if args.output_dir == "output" else output_dir
            print("\nDEMO MODE: using synthetic comments, not real YouTube data.")
            comments = generate_demo_comments(args.max_comments)
        else:
            videos = args.videos
            if not videos:
                raw = input("\nEnter one or more YouTube video URLs or IDs (space separated): ")
                videos = raw.split()
            if not videos:
                print("No video URL or ID provided. Exiting.")
                return
            if args.max_comments < 1:
                print("Please enter a number greater than zero.")
                return

            video_ids = [extract_video_id(v) for v in videos]
            print("\nFetching comments...")
            comments = collect_from_videos(video_ids, args.max_comments, args.order)

        print(f"Fetched {len(comments)} comments.")
        if not comments:
            print("No comments returned. Comments may be disabled or unavailable.")
            return

        print(f"Analyzing sentiment (engine: {args.engine})...")
        results = analyze_comments(comments, args.engine)
        print(f"{len(results)} unique comments remain after cleaning.")
        extra = {w.lower() for w in args.stopwords}
        save_results(results, output_dir, extra)
        print("\nDone!")

    except ValueError as error:
        print(f"\nConfiguration or input error: {error}")
    except ImportError as error:
        print(f"\nMissing dependency: {error}. Run: pip install -r requirements.txt")
    except Exception as error:
        try:
            from googleapiclient.errors import HttpError
        except ImportError:
            HttpError = ()
        if HttpError and isinstance(error, HttpError):
            print(f"\nYouTube API error: {error}")
            print(
                "Check your API key, API quota, video ID, and whether comments "
                "are available for the video."
            )
        else:
            print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()