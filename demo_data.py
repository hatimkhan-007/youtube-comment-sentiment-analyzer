import random
from datetime import datetime, timedelta, timezone

POSITIVE = [
    "This was really helpful, thank you so much!", "Absolutely loved this video, great explanation.",
    "Best tutorial I have watched all year.", "Amazing work, subscribed instantly!",
    "You explain things so clearly, keep it up.", "This made my day, brilliant content.",
    "Fantastic video, learned a lot from it.", "Such a wonderful and inspiring presentation.",
]
NEGATIVE = [
    "This was a waste of time, really disappointing.", "Terrible audio and a confusing explanation.",
    "I hated this, completely useless.", "Worst video on the topic, do not recommend.",
    "So boring and badly edited, I gave up halfway.", "Misleading title, very annoying.",
    "This is awful, I want my time back.", "Poor quality and full of mistakes.",
]
NEUTRAL = [
    "The video is 10 minutes long.", "I watched this on Tuesday.",
    "Does anyone know the name of the tool used at 3:45?", "Uploaded in 2024 according to the page.",
    "He mentions the second chapter in the middle.", "I am watching from Lahore.",
    "What version of the software is shown here?", "The presenter uses a blue background.",
]


def generate_demo_comments(count: int, seed: int = 42) -> list[dict]:
    """Synthetic comments for offline pipeline testing only. Not real YouTube data."""
    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    comments = []
    for i in range(count):
        pool = rng.choices([POSITIVE, NEUTRAL, NEGATIVE], weights=[50, 30, 20])[0]
        text = f"{rng.choice(pool)} (#{i})"
        comments.append({
            "video_id": "DEMO",
            "comment": text,
            "author": f"demo_user_{i % 400}",
            "published_at": (start + timedelta(hours=rng.randint(0, 24 * 120))).isoformat(),
            "like_count": rng.randint(0, 50),
            "reply_count": rng.randint(0, 5),
        })
    return comments