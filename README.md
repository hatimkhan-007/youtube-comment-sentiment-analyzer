# YouTube Comment Sentiment Analyzer

Collect at least 2,000 comments from YouTube videos, classify each as **Positive**, **Neutral** or **Negative** with VADER, run exploratory data analysis (EDA), and export a CSV plus charts.

## What it does

1. Fetches top-level comments through the YouTube Data API v3 (one or more videos).
2. Cleans the text (HTML entities, URLs, whitespace) and removes empty and duplicate comments.
3. Scores every comment with VADER and labels it using the standard thresholds: compound >= 0.05 is Positive, <= -0.05 is Negative, otherwise Neutral.
4. Saves `output/classified_comments.csv` and `output/sentiment_distribution.png`.
5. Runs EDA and saves tables and charts to `output/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Create a YouTube Data API v3 key in Google Cloud Console, then copy `.env.example` to `.env` and paste the key:

```
YOUTUBE_API_KEY=your_key_here
```

## Run

```bash
python main.py --videos https://www.youtube.com/watch?v=VIDEO_ID_1 VIDEO_ID_2 --max-comments 2500
```

Running `python main.py` with no arguments prompts for the video URLs instead.

| Option | Meaning | Default |
|---|---|---|
| `--videos` | One or more video URLs or IDs | prompt |
| `--max-comments` | Total comments to collect across videos | 2500 |
| `--order` | `relevance` or `time` | relevance |
| `--output-dir` | Where results are written | output |
| `--demo` | Synthetic offline data to test the pipeline (writes to `output_demo/`) | off |

Pick videos with enough comments. If fewer than 2,000 unique comments remain after cleaning, the program prints a warning; add more videos.

## Output files

| File | Description |
|---|---|
| `output/classified_comments.csv` | video_id, comment, clean_comment, author, published_at, like_count, reply_count, sentiment, sentiment_score |
| `output/sentiment_distribution.png` | Bar chart of sentiment counts and percentages |
| `output/eda_summary.txt` | Text summary of the EDA tables and top words |
| `output/eda_*.csv` | Distribution, per-sentiment statistics, overall descriptive statistics |
| `output/eda_score_histogram.png` | Histogram of VADER compound scores |
| `output/eda_length_by_sentiment.png` | Comment length by sentiment |
| `output/eda_sentiment_over_time.png` | Sentiment counts over time |
| `output/eda_top_words.png` | Most frequent words per sentiment |

## Project structure

```text
youtube-comment-sentiment-analyzer/
├── main.py              # collection, cleaning, sentiment, CSV and chart
├── eda.py               # exploratory data analysis
├── demo_data.py         # synthetic data for offline testing
├── tests/
│   └── test_sentiment.py
├── docs/
│   └── Methodology.docx
├── output/              # results (generated)
├── CHANGELOG.md         # additions made to the original project
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Tests

```bash
python -m pytest tests
```

## Limitations

- VADER is lexicon-based and tuned for English; sarcasm, slang, mixed opinions and non-English comments can be misclassified.
- Results reflect the collected sample, not every viewer. Relevance ordering favours popular comments.
- Sentiment labels describe tone, not whether a comment is factually correct.

## Responsible use

- Follow YouTube's Terms of Service, API policies and quota limits.
- Do not publish personal information; author names are collected only for dataset completeness.
- Never commit `.env`; it is already in `.gitignore`.
