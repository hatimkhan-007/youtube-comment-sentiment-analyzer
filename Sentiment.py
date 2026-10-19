import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15
LEXICON_WEIGHT = 0.35
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ENGINES = ["xlmr", "vader", "claude"]
CHUNK_SIZE = 256

DEFAULT_TARGET = "the advertisement and the brand behind it"
DEFAULT_LLM_MODEL = "claude-haiku-5-5"
LLM_BATCH_SIZE = 40
LLM_MAX_CHARS = 600
LLM_RETRIES = 3
LLM_SCORES = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}

POSITIVE_WORDS = {
    "mashallah", "masha", "mashaallah", "subhanallah", "jazakallah", "ameen", "aameen",
    "mubarak", "zabardast", "zabrdast", "zabardasat", "kamal", "kamaal", "kmaal", "behtareen",
    "behtar", "shandar", "lajawab", "umda", "acha", "achha", "accha", "achi", "acchi",
    "khoobsurat", "khoob", "pyara", "pyari", "pyar", "pyaar", "pasand", "mohabbat", "muhabbat",
    "shukriya", "shukria", "barkat", "waah", "wah", "dilchasp", "mast", "badhiya", "bhadiya",
    "sahi", "jannat", "love", "beautiful", "best", "great", "awesome", "superb", "wow", "nice",
    "good", "amazing", "perfect", "touching", "excellent", "brilliant",
}
NEGATIVE_WORDS = {
    "bakwas", "bakwaas", "bekar", "bekaar", "ghatiya", "faltu", "fuzool", "bura", "buri",
    "ganda", "gandi", "afsos", "afsoos", "nafrat", "jhoot", "jhoota", "jhooti", "sharam",
    "bewakoof", "kharab", "zulm", "dhoka", "boring", "flop", "waste", "worst", "hate", "bad",
    "boycott", "cringe", "pathetic", "trash", "stupid", "disgusting", "terrible", "awful",
    "garbage", "toxic", "embarrassing", "embarassing", "shameful", "disappointing", "horrible",
}
NEGATORS = {"nahi", "nhi", "nahin", "nai", "not", "no", "mat", "never", "na"}

POSITIVE_EMOJI = set("❤😍🥰😘👍👏🙏🔥💖💕💗😊🤩😻💯✨")
NEGATIVE_EMOJI = set("👎😡😠🤮😤💩🤬😒🙄🖕🤡")

_VERBS = r"(?:buy|buying|bought|purchase|purchasing|purchased|using|use|used|support|supporting|touch|wasted|spent)"
_NEGS = (
    r"(?:no longer|not|never|won'?t|will not|wouldn'?t|stopped|stop|quit|haven'?t|hasn'?t|don'?t|do not|"
    r"doesn'?t|didn'?t|did not|can'?t|cannot|refuse to|wont|dont|nahi|nhi)"
)
NEGATIVE_STANCE = [
    r"\bboycott", r"\bmisandr", r"toxic femin", r"control/alt/delete", r"\bdelete (?:gill?ette|this|the)\b",
    rf"\b{_NEGS}\s+(?:ever\s+|even\s+|be\s+|still\s+|gonna\s+|going to\s+|a\s+penny\s+)?{_VERBS}\b",
    r"\bno more (?:buying|gill?ette|purchases?)\b", r"\b(?:stopped|quit)\s+(?:buying|using|purchasing)\b",
    r"\bavoid\b.{0,40}\b(?:products?|brand|gill?ette|p&g)\b",
    r"\bswitch(?:ed|ing)?\s+(?:brands?|to|from)\b", r"\bswitched\b",
    r"thumbs? down", r"dislike (?:button|it)", r"\bhit the dislike", r"\bdisliked\b", r"\bunsubscrib",
    r"\b(?:well done|great job|nice job|good job|smart move|brilliant move)\b.{0,80}"
    r"\b(?:billions?|lose|lost|customers?|sales|profits?|stock|shareholders?)\b",
    r"\b(?:lost|lose|losing|cost|wasted?)\b.{0,40}\b(?:billions?|\d+\s*b\b)",
    r"\bhalf (?:of )?(?:your|their|the) (?:paying |customer)", r"\b(?:didn'?t|did not) age well\b",
    r"\bpropaganda\b", r"\bgo woke go broke\b", r"\bhate (?:this|the) (?:ad|advert|commercial|video)\b",
    r"\b(?:didn'?t|did not|don'?t|do not|never)\s+(?:like|liked|love|loved)\s+(?:this|the|that)\b",
    r"\b(?:kabhi|kbhi)\s+(?:nahi|nhi)\b.{0,30}\b(?:khareed|kharid|lun|lena|use)",
]
POSITIVE_STANCE = [
    r"\bwill (?:only )?buy (?:only )?gill?ette\b",
    r"\b(?:love|loved|like|liked)\s+(?:this|the|that)\s+(?:ad|advert|commercial|video|message)\b",
    r"\b(?:great|good|powerful|brilliant|excellent|important|needed)\s+(?:ad|advert|commercial|message)\b",
    r"\bproud of (?:gill?ette|p&g|them)\b", r"\bsupport(?:ing)? gill?ette\b",
    r"\bdon'?t (?:really )?(?:see|understand) the (?:problem|anger|outrage)\b",
    r"\bnothing wrong with (?:this|the) (?:ad|advert|commercial)\b",
]
COMPETITORS = r"dollar ?shave|harry'?s|schick|\bbic\b|wilkinson|\bflamingo\b|king of shaves"
NEUTRAL_PATTERNS = [r"^\s*\d{1,2}:\d{2}\b", r"\bmemes?\b"]
SARCASM_PATTERN = re.compile(r"😂|🤣|😆|:D\b|\blol\b|\blmao\b|\brofl\b|\bhaha", re.IGNORECASE)

NEGATIVE_RE = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_STANCE]
POSITIVE_RE = [re.compile(p, re.IGNORECASE) for p in POSITIVE_STANCE]
NEUTRAL_RE = [re.compile(p, re.IGNORECASE) for p in NEUTRAL_PATTERNS]
COMPETITOR_RE = re.compile(COMPETITORS, re.IGNORECASE)

WORD_RE = re.compile(r"(?:[^\W\d_]|[\u0900-\u097F\u0600-\u06FF])+")
REPEAT_RE = re.compile(r"(.)\1{2,}")
ANY_REPEAT_RE = re.compile(r"(.)\1+")
HAS_WORD_RE = re.compile(r"[^\W_]")


def _normalize(token: str) -> str:
    """Collapse elongated letters (loooove -> love, zabardastttt -> zabardast)."""
    if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
        return token
    for candidate in (REPEAT_RE.sub(r"\1\1", token), REPEAT_RE.sub(r"\1", token), ANY_REPEAT_RE.sub(r"\1", token)):
        if candidate in POSITIVE_WORDS or candidate in NEGATIVE_WORDS:
            return candidate
    return token


def lexicon_score(text: str) -> float:
    """Roman Urdu / Hinglish / emoji signal in [-1, 1], with simple negation handling."""
    tokens = [_normalize(t) for t in WORD_RE.findall(text.lower().replace("'", "").replace("’", ""))]
    total = 0
    for i, token in enumerate(tokens):
        sign = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if not sign:
            continue
        window = tokens[max(0, i - 1):i] + tokens[i + 1:i + 3]
        if any(w in NEGATORS for w in window):
            sign = -sign
        total += sign
    total += sum(c in POSITIVE_EMOJI for c in text) - sum(c in NEGATIVE_EMOJI for c in text)
    return max(-1.0, min(1.0, total / 2))


def apply_rules(text: str, score: float) -> float:
    """Stance rules for ad/brand threads: boycott intent, sarcasm, memes, timestamps, emoji-only."""
    t = text.replace("’", "'").replace("`", "'")
    if not HAS_WORD_RE.search(t):
        return lexicon_score(t)
    if any(p.search(t) for p in NEGATIVE_RE):
        return min(score, -0.6)
    if any(p.search(t) for p in POSITIVE_RE):
        return max(score, 0.6)
    if COMPETITOR_RE.search(t) and score > NEGATIVE_THRESHOLD:
        return -0.4
    if any(p.search(t) for p in NEUTRAL_RE):
        return 0.0
    if score >= POSITIVE_THRESHOLD and SARCASM_PATTERN.search(t):
        return 0.0
    return score


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, top_k=None)


def _model_scores(texts: list[str], batch_size: int = 32) -> tuple[list[float], list[float]]:
    clf = _pipeline()
    fallback = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    scores, confidences = [], []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start:start + CHUNK_SIZE]
        for out in clf(chunk, batch_size=batch_size, truncation=True, max_length=256):
            probs = {fallback.get(i["label"].lower(), i["label"].lower()): i["score"] for i in out}
            scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
            confidences.append(max(probs.values()))
        print(f"  scored {min(start + CHUNK_SIZE, len(texts))}/{len(texts)}")
    return scores, confidences


def label_from_score(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    if score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def _cache_path() -> Path:
    return Path(os.getenv("LLM_CACHE", "output/llm_cache.json"))


def _load_cache() -> dict:
    path = _cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _cache_key(text: str, target: str, model: str) -> str:
    return hashlib.sha1(f"{model}|{target}|{text}".encode("utf-8")).hexdigest()


def _prompt(batch: list[str], target: str) -> str:
    lines = "\n".join(f"{i + 1}. {t[:LLM_MAX_CHARS]}" for i, t in enumerate(batch))
    return (
        f"Classify each YouTube comment by its stance toward {target}.\n"
        "Comments may be English, Roman Urdu, Hinglish, Urdu or Hindi script, with slang, emoji and sarcasm.\n\n"
        "Positive = praises, supports or defends it, or says they will buy or keep buying.\n"
        "Negative = criticizes, mocks (including sarcasm), opposes, insults, or says they will boycott, switch or stop buying.\n"
        "Neutral = unrelated, memes, timestamps, emoji only, questions, facts or news with no clear stance.\n\n"
        "Return ONLY a JSON array with one object per comment, in order, like "
        '[{"id": 1, "label": "Negative", "confidence": 0.9}]. '
        'label must be "Positive", "Neutral" or "Negative"; confidence is a number from 0 to 1.\n\n'
        f"Comments:\n{lines}"
    )


def _parse_batch(raw: str, size: int) -> list[tuple[str, float]]:
    data = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
    by_id = {int(item["id"]): item for item in data}
    results = []
    for i in range(1, size + 1):
        item = by_id[i]
        label = str(item["label"]).strip().capitalize()
        if label not in LLM_SCORES:
            raise ValueError(f"bad label {label!r}")
        results.append((label, max(0.0, min(1.0, float(item.get("confidence", 1.0))))))
    return results


def _llm_labels(texts: list[str], target: str, model: str) -> tuple[list[str], list[float]]:
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file to use --engine claude.")

    client = anthropic.Anthropic()
    cache = _load_cache()
    keys = [_cache_key(t, target, model) for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"  {len(texts) - len(todo)} cached, {len(todo)} to classify with {model}")

    for start in range(0, len(todo), LLM_BATCH_SIZE):
        idx = todo[start:start + LLM_BATCH_SIZE]
        batch = [texts[i] for i in idx]
        for attempt in range(1, LLM_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=model, max_tokens=2000, temperature=0,
                    messages=[{"role": "user", "content": _prompt(batch, target)}],
                )
                results = _parse_batch(response.content[0].text, len(batch))
                break
            except Exception as error:
                if attempt == LLM_RETRIES:
                    _save_cache(cache)
                    raise RuntimeError(f"LLM batch failed after {LLM_RETRIES} tries: {error}") from error
                time.sleep(2 * attempt)
        for i, (label, confidence) in zip(idx, results):
            cache[keys[i]] = [label, confidence]
        _save_cache(cache)
        print(f"  classified {min(start + LLM_BATCH_SIZE, len(todo))}/{len(todo)}")

    labels = [cache[k][0] for k in keys]
    confidences = [cache[k][1] for k in keys]
    return labels, confidences


def classify(
    texts: list[str], engine: str = "xlmr", target: str = DEFAULT_TARGET,
    llm_model: str = DEFAULT_LLM_MODEL, use_rules: bool = True,
) -> tuple[list[str], list[float], list[float]]:
    """Return (labels, scores, confidences) for each text."""
    if engine == "claude":
        labels, confidences = _llm_labels(texts, target, llm_model)
        scores = [LLM_SCORES[l] * c for l, c in zip(labels, confidences)]
        return labels, scores, confidences

    if engine == "vader":
        base = [_vader().polarity_scores(t)["compound"] for t in texts]
        confidences = [abs(b) for b in base]
    else:
        base, confidences = _model_scores(texts)

    scores = []
    for text, b in zip(texts, base):
        score = max(-1.0, min(1.0, b + LEXICON_WEIGHT * lexicon_score(text)))
        scores.append(apply_rules(text, score) if use_rules else score)
    return [label_from_score(s) for s in scores], scores, confidences