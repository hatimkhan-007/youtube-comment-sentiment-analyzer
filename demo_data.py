import random
from datetime import datetime, timedelta, timezone

POSITIVE = [
    "This was really helpful, thank you so much!", "Absolutely loved this video, great explanation.",
    "Best tutorial I have watched all year.", "Amazing work, subscribed instantly!",
    "Bohat zabardast video hai, mashallah!", "Kamal ka content hai bhai, love it",
    "Bahut acha laga, Allah apko khush rakhe ameen", "Ye dil ko chu gaya, behtareen",
    "बहुत अच्छा वीडियो है", "بہت ہی خوبصورت اور شاندار",
]
NEGATIVE = [
    "This was a waste of time, really disappointing.", "Terrible audio and a confusing explanation.",
    "Worst video on the topic, do not recommend.", "So boring and badly edited, I gave up halfway.",
    "Bilkul bakwas video hai, time waste", "Ye bohat ghatiya hai, mujhe pasand nahi aya",
    "Bekar content, afsos hua dekh kar", "बकवास वीडियो",
]
NEUTRAL = [
    "The video is 10 minutes long.", "I watched this on Tuesday.",
    "Does anyone know the name of the tool used at 3:45?", "I am watching from Lahore.",
    "Ye video kab upload hui thi?", "Kon sa software use kiya hai is mein?",
    "Ramzan mubarak sab ko", "Main Lahore se dekh raha hun",
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