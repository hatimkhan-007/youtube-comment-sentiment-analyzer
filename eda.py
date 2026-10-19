import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = [
    "DejaVu Sans", "Nirmala UI", "Segoe UI", "Arial Unicode MS", "Noto Sans Arabic", "Noto Sans Devanagari",
]

LABELS = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#2e9e5b", "Neutral": "#8c8c8c", "Negative": "#d64545"}

ENGLISH_STOPWORDS = set("""
about after all also and any are because been but can could did does for from get got had has
have her him his how into its just like me my not now one our out some than that the their them
then there they this too was we were what when which who will with would you your dont thats
ive youre video very made add
dont didnt doesnt isnt wasnt arent wont cant couldnt wouldnt shouldnt havent hasnt
why where since being years still even only ever want make think people say said really
""".split())

ROMAN_URDU_STOPWORDS = set("""
hai hain hy h ho hun hu tha thi the thay hota hoti hoga hogi ka ki ke ky ko se sy ne me
mein main mai pe par per ye yeh wo woh yahan wahan aur or bhi to toh kya kia kiya kon kaun
kyun kyu nahi nhi nahin na bhot bohat bohot bahut buhat kar kr karo kare karna karta karti
mera meri mere tera teri tere tum tu aap ap hum hamara apna apni apne liye lye wala wali wale
jo jis jab tab ab abhi ek ik is us in un bas sab sirf lekin magar agar phir kuch koi kisi
hona gaya gayi gaye raha rahi rahe diya di diye lag laga lagi
""".split())

STOPWORDS = ENGLISH_STOPWORDS | ROMAN_URDU_STOPWORDS
WORD_PATTERN = re.compile(r"(?:[^\W\d_]|[\u0900-\u097F\u0600-\u06FF])+")
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
        mean_confidence=("confidence", "mean"),
        mean_words=("word_count", "mean"),
        mean_likes=("like_count", "mean"),
        median_likes=("like_count", "median"),
    ).reindex(LABELS).round(3)
    overall = df[["sentiment_score", "confidence", "char_count", "word_count", "like_count"]].describe().round(3)
    return {
        "sentiment_distribution": distribution,
        "stats_by_sentiment": by_sentiment,
        "overall_descriptive_stats": overall,
    }


def distinctive_words(
    df: pd.DataFrame, n: int = 10, extra_stopwords: set[str] | None = None, min_total: int = 5
) -> dict[str, list[tuple[str, int]]]:
    """Top words per sentiment ranked by count x log-lift, so words shared by every class drop out."""
    blocked = STOPWORDS | (extra_stopwords or set())
    counters: dict[str, Counter] = {label: Counter() for label in LABELS}
    for label in LABELS:
        for text in df.loc[df["sentiment"] == label, "clean_comment"]:
            text = text.lower().replace("'", "").replace("’", "").replace("`", "")
            words = (REPEAT_PATTERN.sub(r"\1\1", w) for w in WORD_PATTERN.findall(text))
            counters[label].update(w for w in words if w not in blocked and len(w) >= (3 if w.isascii() else 2))

    total: Counter = Counter()
    for counter in counters.values():
        total.update(counter)
    n_all = sum(total.values()) or 1

    result = {}
    for label in LABELS:
        counter = counters[label]
        n_class = sum(counter.values()) or 1
        scored = []
        for word, count in counter.items():
            if total[word] < min_total:
                continue
            lift = (count / n_class) / (total[word] / n_all)
            if lift > 1.0:
                scored.append((count * math.log(lift), word, count))
        scored.sort(reverse=True)
        picks = [(word, count) for _, word, count in scored[:n]]
        if len(picks) < n:
            used = {word for word, _ in picks}
            picks += [(w, c) for w, c in counter.most_common() if w not in used][: n - len(picks)]
        result[label] = picks
    return result


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
    x = range(len(table))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, table["Positive"], color=COLORS["Positive"], label="Positive", width=0.85)
    ax.bar(x, table["Neutral"], bottom=table["Positive"], color=COLORS["Neutral"], label="Neutral", width=0.85)
    ax.bar(
        x, table["Negative"], bottom=table["Positive"] + table["Neutral"],
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


def plot_top_words(words_by_label: dict[str, list[tuple[str, int]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, label in zip(axes, LABELS):
        words = words_by_label[label]
        if words:
            names, values = zip(*words)
            ax.barh(names[::-1], values[::-1], color=COLORS[label])
        ax.set_title(f"Distinctive words: {label}")
        ax.set_xlabel("Occurrences")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: Path, extra_stopwords: set[str] | None = None) -> None:
    """Write EDA tables, a text summary and supporting charts to output_dir."""
    data = add_features(df)
    tables = summary_tables(data)
    words_by_label = distinctive_words(data, extra_stopwords=extra_stopwords)

    lines = [f"Total unique comments analyzed: {len(data)}", ""]
    for name, table in tables.items():
        table.to_csv(output_dir / f"eda_{name}.csv")
        lines += [name.replace("_", " ").title(), table.to_string(), ""]

    for label in LABELS:
        lines.append(f"Distinctive words ({label}): " + ", ".join(f"{w} ({c})" for w, c in words_by_label[label]))

    top_neg = get_top_negative_comments(data, n=5)
    if not top_neg.empty:
        lines.append("\nTop Negative Comments Caught:")
        for _, row in top_neg.iterrows():
            lines.append(f"- [Score: {row['sentiment_score']:.2f}, Likes: {row['like_count']}] {row['comment']}")

    (output_dir / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    plot_score_histogram(data, output_dir / "eda_score_histogram.png")
    plot_length_by_sentiment(data, output_dir / "eda_length_by_sentiment.png")
    plot_top_words(words_by_label, output_dir / "eda_top_words.png")
    plot_sentiment_over_time(data, output_dir / "eda_sentiment_over_time.png")