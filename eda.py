import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#2e9e5b", "Neutral": "#8c8c8c", "Negative": "#d64545"}

STOPWORDS = set("""
a about after all also am an and any are as at be because been but by can could did do does
for from get got had has have he her him his how i if in into is it its just like me my no not
now of on one or our out so some than that the their them then there they this to too up us
was we were what when which who will with would you your im its dont thats ive youre video
""".split())

WORD_PATTERN = re.compile(r"[a-z']{3,}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["char_count"] = out["clean_comment"].str.len()
    out["word_count"] = out["clean_comment"].str.split().str.len()
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    return out


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    distribution = pd.DataFrame({
        "count": counts,
        "percent": (counts / len(df) * 100).round(2),
    })
    by_sentiment = df.groupby("sentiment").agg(
        mean_score=("sentiment_score", "mean"),
        mean_words=("word_count", "mean"),
        mean_likes=("like_count", "mean"),
        median_likes=("like_count", "median"),
    ).reindex(LABELS).round(3)
    overall = df[["sentiment_score", "char_count", "word_count", "like_count"]].describe().round(3)
    return {
        "sentiment_distribution": distribution,
        "stats_by_sentiment": by_sentiment,
        "overall_descriptive_stats": overall,
    }


def top_words(df: pd.DataFrame, label: str, n: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for text in df.loc[df["sentiment"] == label, "clean_comment"]:
        words = WORD_PATTERN.findall(text.lower())
        counter.update(w for w in words if w not in STOPWORDS)
    return counter.most_common(n)


def get_top_negative_comments(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Isolate and return the most relevant or most-liked negative comments."""
    negatives = df[df["sentiment"] == "Negative"]
    if negatives.empty:
        return pd.DataFrame()
    # Sort by like_count descending, then lowest sentiment score
    return negatives.sort_values(by=["like_count", "sentiment_score"], ascending=[False, True]).head(n)


def plot_score_histogram(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["sentiment_score"], bins=30, color="#4a78c2", edgecolor="white")
    ax.axvline(0.05, color="#2e9e5b", linestyle="--", label="Positive threshold (+0.05)")
    ax.axvline(-0.05, color="#d64545", linestyle="--", label="Negative threshold (-0.05)")
    ax.set_title("Distribution of VADER Compound Scores")
    ax.set_xlabel("Compound score")
    ax.set_ylabel("Number of comments")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_length_by_sentiment(df: pd.DataFrame, path: Path) -> None:
    data = [df.loc[df["sentiment"] == l, "word_count"] for l in LABELS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box = ax.boxplot(data, tick_labels=LABELS, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], LABELS):
        patch.set_facecolor(COLORS[label])
    ax.set_title("Comment Length (words) by Sentiment")
    ax.set_ylabel("Words per comment")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sentiment_over_time(df: pd.DataFrame, path: Path) -> bool:
    dated = df.dropna(subset=["published_at"])
    if dated["published_at"].nunique() < 2:
        return False
    span_days = (dated["published_at"].max() - dated["published_at"].min()).days
    freq = "D" if span_days <= 60 else "M"
    period = dated["published_at"].dt.tz_localize(None).dt.to_period(freq)
    table = (
        dated.assign(period=period)
        .groupby(["period", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[l] for l in LABELS], width=0.85)
    ax.set_title("Comments Over Time by Sentiment")
    ax.set_xlabel("Date" if freq == "D" else "Month")
    ax.set_ylabel("Number of comments")
    ax.tick_params(axis="x", rotation=60)
    if len(table) > 20:
        step = max(1, len(table) // 12)
        for i, tick in enumerate(ax.get_xticklabels()):
            tick.set_visible(i % step == 0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_top_words(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, label in zip(axes, LABELS):
        words = top_words(df, label)
        if words:
            names, values = zip(*words)
            ax.barh(names[::-1], values[::-1], color=COLORS[label])
        ax.set_title(f"Top words: {label}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Write EDA tables, a text summary and supporting charts to output_dir."""
    data = add_features(df)
    tables = summary_tables(data)

    lines = [f"Total unique comments analyzed: {len(data)}", ""]
    for name, table in tables.items():
        table.to_csv(output_dir / f"eda_{name}.csv")
        lines += [name.replace("_", " ").title(), table.to_string(), ""]
    
    for label in LABELS:
        lines.append(f"Top words ({label}): " + ", ".join(f"{w} ({c})" for w, c in top_words(data, label)))
    
    # Append top negative comments to summary
    top_neg = get_top_negative_comments(data, n=5)
    if not top_neg.empty:
        lines.append("\nTop Negative Comments Caught:")
        for _, row in top_neg.iterrows():
            lines.append(f"- [Score: {row['sentiment_score']:.2f}, Likes: {row['like_count']}] {row['comment']}")

    (output_dir / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    plot_score_histogram(data, output_dir / "eda_score_histogram.png")
    plot_length_by_sentiment(data, output_dir / "eda_length_by_sentiment.png")
    plot_top_words(data, output_dir / "eda_top_words.png")
    plot_sentiment_over_time(data, output_dir / "eda_sentiment_over_time.png")