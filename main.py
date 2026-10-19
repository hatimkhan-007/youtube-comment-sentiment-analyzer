import argparse
import html
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
ANALYSIS_COLUMNS = [
    "comment_id", "video_id", "comment", "clean_comment", "author", "published_at",
    "like_count", "reply_count", "sentiment", "sentiment_score", "confidence",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPAM_PATTERN = re.compile(
    r"bitcoin|crypto|forex|binance|telegram|whatsapp|\b\d{10,}\b|investment|earn\s*\$|"
    r"\bdm me\b|click (the )?link|giveaway|sub4sub|subscribe to my|check (out )?my channel",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize a raw comment: unescape HTML, drop URLs and @mentions, collapse whitespace."""
    text = html.unescape(str(text))
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_spam(text: str) -> bool:
    return bool(SPAM_PATTERN.search(text))


def analyze_comments(
    comments: list[dict], engine: str = "xlmr", target: str = sentiment.DEFAULT_TARGET,
    llm_model: str = sentiment.DEFAULT_LLM_MODEL, use_rules: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, de-duplicate, remove spam, then score. Returns (analyzed, spam)."""
    df = pd.DataFrame(comments)
    empty = pd.DataFrame(columns=ANALYSIS_COLUMNS)

    if df.empty:
        return empty, pd.DataFrame(columns=["comment_id", "video_id", "clean_comment"])

    if "comment_id" not in df:
        df["comment_id"] = [f"row{i}" for i in range(len(df))]
    for column in ("video_id", "author", "published_at"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("")
    for column in ("like_count", "reply_count"):
        if column not in df:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["clean_comment"] = df["comment"].apply(clean_text)
    df = df[df["clean_comment"] != ""]
    df = df.loc[~df["clean_comment"].str.lower().duplicated()].copy()

    spam_mask = df["clean_comment"].apply(is_spam)
    spam = df.loc[spam_mask, ["comment_id", "video_id", "clean_comment"]]
    df = df.loc[~spam_mask].copy()

    if df.empty:
        return empty, spam

    labels, scores, confidences = sentiment.classify(
        df["clean_comment"].tolist(), engine, target, llm_model, use_rules
    )
    df["sentiment"] = labels
    df["sentiment_score"] = scores
    df["confidence"] = confidences

    return df[ANALYSIS_COLUMNS].reset_index(drop=True), spam.reset_index(drop=True)


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


def export_review_sample(df: pd.DataFrame, path: Path, per_label: int = 35) -> None:
    """Stratified sample with an empty manual_label column to measure real accuracy."""
    parts = [group.sample(min(len(group), per_label), random_state=42) for _, group in df.groupby("sentiment")]
    sample = pd.concat(parts)[["comment_id", "clean_comment", "sentiment", "sentiment_score", "confidence"]]
    sample = sample.assign(manual_label="").sample(frac=1, random_state=1)
    sample.to_csv(path, index=False, encoding="utf-8-sig")


def evaluate(labels_path: Path, output_dir: Path) -> None:
    """Compare manual_label values with the labels in classified_comments.csv (matched by comment_id)."""
    labelled = pd.read_csv(labels_path)
    labelled = labelled[labelled["manual_label"].notna() & (labelled["manual_label"].astype(str).str.strip() != "")]
    if labelled.empty:
        print("No manual_label values found. Fill the manual_label column with Positive, Neutral or Negative.")
        return
    labelled = labelled.assign(manual_label=labelled["manual_label"].astype(str).str.strip().str.capitalize())

    classified_path = output_dir / "classified_comments.csv"
    if classified_path.exists() and "comment_id" in labelled:
        current = pd.read_csv(classified_path)[["comment_id", "sentiment"]]
        labelled = labelled.drop(columns=["sentiment"], errors="ignore").merge(current, on="comment_id")
    if labelled.empty:
        print("None of the labelled comments appear in classified_comments.csv.")
        return

    accuracy = (labelled["manual_label"] == labelled["sentiment"]).mean()
    print(f"Accuracy on {len(labelled)} manually labelled comments: {accuracy * 100:.1f}%")
    print("\nRows = your label, columns = model label:")
    print(pd.crosstab(labelled["manual_label"], labelled["sentiment"]))


def save_results(df: pd.DataFrame, spam: pd.DataFrame, output_dir: Path, extra_stopwords: set[str]) -> None:
    """Export classified comments, review sample, sentiment chart and EDA outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classified_comments.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nClassified comments saved to: {csv_path}")

    if not spam.empty:
        spam.to_csv(output_dir / "spam_comments.csv", index=False, encoding="utf-8-sig")
        print(f"{len(spam)} spam comments removed (saved to spam_comments.csv).")

    if df.empty:
        print("No comments were available to chart.")
        return

    export_review_sample(df, output_dir / "review_sample.csv")
    print("Review sample saved to review_sample.csv (fill manual_label, then run --evaluate).")

    counts = plot_distribution(df, output_dir / "sentiment_distribution.png")
    eda.run_eda(df, output_dir, extra_stopwords)
    print(f"Charts and EDA outputs saved to: {output_dir}")

    print("\nSentiment summary:")
    for label, count in counts.items():
        print(f"{label}: {count} comments ({count / len(df) * 100:.1f}%)")

    if len(df) < MIN_REQUIRED_COMMENTS:
        print(
            f"\nWARNING: only {len(df)} unique comments after cleaning; "
            f"the project requires at least {MIN_REQUIRED_COMMENTS}. "
            "Fetch more comments with fetch.py first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube Comment Sentiment Analyzer (run fetch.py first)")
    parser.add_argument("--input", default="output/raw_comments.csv", help="Comments CSV created by fetch.py")
    parser.add_argument("--evaluate", help="Labelled review CSV (manual_label filled); compares with classified_comments.csv")
    parser.add_argument(
        "--engine", choices=sentiment.ENGINES, default="xlmr",
        help="xlmr = local multilingual model + stance rules; claude = LLM, best for sarcasm "
             "(needs ANTHROPIC_API_KEY); vader = English only",
    )
    parser.add_argument(
        "--target", default=sentiment.DEFAULT_TARGET,
        help="What the sentiment is about, e.g. 'the Gillette ad and brand' (used by --engine claude)",
    )
    parser.add_argument("--llm-model", default=sentiment.DEFAULT_LLM_MODEL, help="Claude model for --engine claude")
    parser.add_argument("--no-rules", action="store_true", help="Disable the stance rules (local engines)")
    parser.add_argument(
        "--stopwords", nargs="*", default=[],
        help="Extra words to hide from the word charts (brand names, channel names)",
    )
    parser.add_argument("--output-dir", default="output", help="Folder for results")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic offline dataset")
    parser.add_argument("--demo-size", type=int, default=2500, help="Number of demo comments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("YouTube Comment Sentiment Analyzer")
    print("=" * 50)

    if args.evaluate:
        evaluate(Path(args.evaluate), Path(args.output_dir))
        return

    output_dir = Path(args.output_dir)

    try:
        if args.demo:
            from demo_data import generate_demo_comments

            output_dir = Path("output_demo") if args.output_dir == "output" else output_dir
            print("\nDEMO MODE: using synthetic comments, not real YouTube data.")
            comments = generate_demo_comments(args.demo_size)
        else:
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"\n{input_path} not found. Run fetch.py first to collect comments.")
                return
            comments = pd.read_csv(input_path).to_dict("records")

        print(f"Loaded {len(comments)} comments.")
        if not comments:
            print("The comments file is empty.")
            return

        print(f"Analyzing sentiment (engine: {args.engine})...")
        results, spam = analyze_comments(
            comments, args.engine, args.target, args.llm_model, not args.no_rules
        )
        print(f"{len(results)} unique comments remain after cleaning.")
        save_results(results, spam, output_dir, {w.lower() for w in args.stopwords})
        print("\nDone!")

    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}")
    except ImportError as error:
        print(f"\nMissing dependency: {error}. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()