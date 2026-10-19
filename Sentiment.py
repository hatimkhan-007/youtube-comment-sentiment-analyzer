import re
from functools import lru_cache

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15
LEXICON_WEIGHT = 0.35
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ENGINES = ["xlmr", "vader"]

POSITIVE_WORDS = {
    "mashallah", "masha", "mashaallah", "subhanallah", "jazakallah", "ameen", "aameen",
    "mubarak", "zabardast", "zabrdast", "kamal", "kmaal", "behtareen", "behtar", "shandar",
    "lajawab", "umda", "acha", "achha", "accha", "achi", "acchi", "khoobsurat", "pyara",
    "pyari", "pasand", "mohabbat", "muhabbat", "shukriya", "shukria", "barkat", "waah", "wah",
    "dilchasp", "zabardasat", "mast", "kamaal", "love", "beautiful", "best", "great",
}
NEGATIVE_WORDS = {
    "bakwas", "bekar", "bekaar", "ghatiya", "faltu", "bura", "buri", "afsos", "nafrat",
    "jhoot", "jhoota", "sharam", "bewakoof", "kharab", "zulm", "boring", "flop", "waste",
    "worst", "hate", "bad",
}
NEGATORS = {"nahi", "nhi", "nahin", "nai", "not", "no", "mat", "never"}

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}")


def _normalize(token: str) -> str:
    """Collapse elongated letters (loooove -> love, zabardastttt -> zabardast)."""
    if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
        return token
    for candidate in (REPEAT_RE.sub(r"\1\1", token), REPEAT_RE.sub(r"\1", token), re.sub(r"(.)\1+", r"\1", token)):
        if candidate in POSITIVE_WORDS or candidate in NEGATIVE_WORDS:
            return candidate
    return token


def lexicon_score(text: str) -> float:
    """Roman Urdu / Hinglish lexicon signal in [-1, 1], with simple negation handling."""
    tokens = [_normalize(t) for t in WORD_RE.findall(text.lower())]
    total = 0
    for i, token in enumerate(tokens):
        sign = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if not sign:
            continue
        window = tokens[max(0, i - 1):i] + tokens[i + 1:i + 3]
        if any(w in NEGATORS for w in window):
            sign = -sign
        total += sign
    return max(-1.0, min(1.0, total / 2))


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, top_k=None)


def _model_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    outputs = _pipeline()(texts, batch_size=batch_size, truncation=True, max_length=256)
    fallback = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    scores = []
    for out in outputs:
        probs = {}
        for item in out:
            label = item["label"].lower()
            probs[fallback.get(label, label)] = item["score"]
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


def score_texts(texts: list[str], engine: str = "xlmr") -> list[float]:
    """Return a polarity score in [-1, 1] per text (model score blended with the lexicon)."""
    if engine == "vader":
        base = [_vader().polarity_scores(t)["compound"] for t in texts]
    else:
        base = _model_scores(texts)
    return [
        max(-1.0, min(1.0, b + LEXICON_WEIGHT * lexicon_score(t)))
        for b, t in zip(base, texts)
    ]


def label_from_score(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    if score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"