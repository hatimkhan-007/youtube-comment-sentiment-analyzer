import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

LABELS = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#2e9e5b", "Neutral": "#8c8c8c", "Negative": "#d64545"}

ENGLISH_STOPWORDS = set("""
about after all also and any are because been but can could did does for from get got had has
have her him his how into its just like me my not now one our out some than that the their them
then there they this too was we were what when which who will with would you your dont thats
ive youre video very made add
""".split())

ROMAN_URDU_STOPWORDS = set("""
hai hain hy h ho hun hu tha thi the thay hota hoti hota hoga hogi ka ki ke ky ko se sy ne me
mein main mai pe par per ye yeh wo woh yahan wahan aur or bhi to toh kya kia kiya kon kaun
kyun kyu nahi nhi nahin na bhot bohat bohot bahut buhat kar kr karo kare karna karta karti
mera meri mere tera teri tere tum tu aap ap hum hamara apna apni apne liye lye wala wali wale
jo jis jab tab ab abhi ek ik is us in un bas sab sirf lekin magar agar phir kuch koi kisi
hona gaya gayi gaye raha rahi rahe diya di diye lag laga lagi
""".split())

STOPWORDS = ENGLISH_STOPWORDS | ROMAN_URDU_STOPWORDS
WORD_PATTERN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
REPEAT_PATTERN = re.compile(r"(.)\1{2,}")


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


def top_words(df: pd.DataFrame, label: str, n: int = 10, extra_stopwords: set[str] | None = None) -> list[tuple[str, int]]:
    blocked = STOPWORDS | (extra_stopwords or set())
    counter: Counter = Counter()
    for text in df.loc[df["sentiment"] == label, "clean_comment"]:
        words = (REPEAT_PATTERN.sub(r"\1\1", w) for w in WORD_PATTERN.findall(text.lower()))
        counter.update(w for w in words if w not in blocked)
    return counter.most_common(n)


def get_top_negative_comments(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the most-liked negative comments."""
    negatives = df[df["sentiment"] == "Negative"]
    if negatives.empty:
        return pd.DataFrame()
    return negatives.sort_values(by=["like_count", "sentiment_score"], ascending=[False, True]).head(n)


def plot_score_histogram(df: pd.DataFrame, path: Path) -> None:
    bins = np.linspace(-1, 1, 41)
    data = [df.loc[df["sentiment"] == l, "sentiment_score"] for l in LABELS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data, bins=bins, stacked=True, color=[COLORS[l] for l in LABELS], label=LABELS, edgecolor="white")
    ax.axvline(POSITIVE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.axvline(NEGATIVE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Sentiment Scores")
    ax.set_xlabel(f"Score (positive minus negative; thresholds at ±{POSITIVE_THRESHOLD})")
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
    for median in box["medians"]:
        median.set_color("black")
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
    if span_days <= 60:
        freq, xlabel = "D", "Date"
    elif span_days <= 730:
        freq, xlabel = "M", "Month"
    else:
        freq, xlabel = "Q", "Quarter"
    period = dated["published_at"].dt.tz_localize(None).dt.to_period(freq)
    table = (
        dated.assign(period=period)
        .groupby(["period", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    table = table.reindex(pd.period_range(table.index.min(), table.index.max(), freq=freq), fill_value=0)
    labels = table.index.astype(str)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(table)), table["Positive"], color=COLORS["Positive"], label="Positive", width=0.85)
    ax.bar(range(len(table)), table["Neutral"], bottom=table["Positive"], color=COLORS["Neutral"], label="Neutral", width=0.85)
    ax.bar(
        range(len(table)), table["Negative"], bottom=table["Positive"] + table["Neutral"],
        color=COLORS["Negative"], label="Negative", width=0.85,
    )
    step = max(1, len(table) // 12)
    ax.set_xticks(range(0, len(table), step))
    ax.set_xticklabels(labels[::step], rotation=60, ha="right")
    ax.set_title("Comments Over Time by Sentiment")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of comments")
    ax.legend(title="sentiment")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_top_words(df: pd.DataFrame, path: Path, extra_stopwords: set[str] | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, label in zip(axes, LABELS):
        words = top_words(df, label, extra_stopwords=extra_stopwords)
        if words:
            names, values = zip(*words)
            ax.barh(names[::-1], values[::-1], color=COLORS[label])
        ax.set_title(f"Top words: {label}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: Path, extra_stopwords: set[str] | None = None) -> None:
    """Write EDA tables, a text summary and supporting charts to output_dir."""
    data = add_features(df)
    tables = summary_tables(data)

    lines = [f"Total unique comments analyzed: {len(data)}", ""]
    for name, table in tables.items():
        table.to_csv(output_dir / f"eda_{name}.csv")
        lines += [name.replace("_", " ").title(), table.to_string(), ""]

    for label in LABELS:
        words = top_words(data, label, extra_stopwords=extra_stopwords)
        lines.append(f"Top words ({label}): " + ", ".join(f"{w} ({c})" for w, c in words))

    top_neg = get_top_negative_comments(data, n=5)
    if not top_neg.empty:
        lines.append("\nTop Negative Comments Caught:")
        for _, row in top_neg.iterrows():
            lines.append(f"- [Score: {row['sentiment_score']:.2f}, Likes: {row['like_count']}] {row['comment']}")

    (output_dir / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    plot_score_histogram(data, output_dir / "eda_score_histogram.png")
    plot_length_by_sentiment(data, output_dir / "eda_length_by_sentiment.png")
    plot_top_words(data, output_dir / "eda_top_words.png", extra_stopwords)
    plot_sentiment_over_time(data, output_dir / "eda_sentiment_over_time.png")