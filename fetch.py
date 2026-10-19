import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TARGET = 2500
RAW_COLUMNS = ["comment_id", "video_id", "comment", "author", "published_at", "like_count", "reply_count"]


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
    """Fetch up to max_comments top-level comments; keep partial results on API errors."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is missing. Add it to a .env file in the project folder.")

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order=order,
            ).execute()
        except HttpError as error:
            print(f"  YouTube API error: {error}\n  Keeping the {len(comments)} comments fetched so far.")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
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


def collect_from_videos(video_ids: list[str], per_video: int, orders: list[str]) -> list[dict]:
    """Collect up to per_video comments per video and order, skipping duplicates."""
    collected: list[dict] = []
    seen: set[str] = set()
    for video_id in video_ids:
        for order in orders:
            print(f"Fetching up to {per_video} comments from {video_id} (order: {order})...")
            batch = fetch_comments(video_id, per_video, order)
            fresh = [c for c in batch if c["comment_id"] not in seen]
            seen.update(c["comment_id"] for c in fresh)
            print(f"  got {len(batch)} comments ({len(fresh)} new)")
            collected.extend(fresh)
    return collected


def save_raw(comments: list[dict], path: Path) -> pd.DataFrame:
    """Merge comments into the CSV so data accumulates across runs without duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(comments, columns=RAW_COLUMNS)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df = df.drop_duplicates(subset="comment_id").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch YouTube comments into a CSV (no sentiment analysis)")
    parser.add_argument("--videos", nargs="+", help="One or more YouTube video URLs or IDs")
    parser.add_argument(
        "--max-comments", type=int, default=DEFAULT_TARGET,
        help=f"Comments to fetch per video and per order (default {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--order", choices=["relevance", "time", "both"], default="both",
        help="Comment ordering requested from the API (default both, for more variety)",
    )
    parser.add_argument("--output", default="output/raw_comments.csv", help="CSV file to create or extend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("YouTube Comment Fetcher")
    print("=" * 50)

    videos = args.videos
    if not videos:
        videos = input("\nEnter one or more YouTube video URLs or IDs (space separated): ").split()
    if not videos:
        print("No video URL or ID provided. Exiting.")
        return
    if args.max_comments < 1:
        print("Please enter a number greater than zero.")
        return

    orders = ["relevance", "time"] if args.order == "both" else [args.order]
    video_ids = [extract_video_id(v) for v in videos]

    try:
        fetched = collect_from_videos(video_ids, args.max_comments, orders)
    except ValueError as error:
        print(f"\nConfiguration error: {error}")
        return
    except ImportError as error:
        print(f"\nMissing dependency: {error}. Run: pip install -r requirements.txt")
        return

    df = save_raw(fetched, Path(args.output))
    print(f"\nSaved {len(df)} total comments to: {args.output}")
    print(f"Per video: {df['video_id'].value_counts().to_dict()}")
    print(f"\nWhen you are ready, run:\npython main.py --input {args.output}")


if __name__ == "__main__":
    main()