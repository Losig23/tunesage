"""Seed the DB with ~15 sample listens across genres so the demo works instantly."""

import db

SAMPLE_LISTENS = [
    # (title, artist, genre, tags, times_to_log)
    ("Blinding Lights", "The Weeknd", "pop", ["energetic", "night-drive"], 12),
    ("Levitating", "Dua Lipa", "pop", ["energetic", "dance"], 9),
    ("As It Was", "Harry Styles", "pop", ["catchy", "upbeat"], 7),
    ("HUMBLE.", "Kendrick Lamar", "hip-hop", ["energetic", "bold"], 11),
    ("SICKO MODE", "Travis Scott", "hip-hop", ["energetic", "hype"], 8),
    ("God's Plan", "Drake", "hip-hop", ["chill", "melodic"], 5),
    ("505", "Arctic Monkeys", "indie rock", ["moody", "romantic"], 10),
    ("Sweater Weather", "The Neighbourhood", "indie rock", ["chill", "moody"], 8),
    ("Fluorescent Adolescent", "Arctic Monkeys", "indie rock", ["energetic", "fun"], 4),
    ("Take Five", "Dave Brubeck Quartet", "jazz", ["chill", "sophisticated"], 6),
    ("So What", "Miles Davis", "jazz", ["chill", "cool"], 5),
    ("Strobe", "deadmau5", "electronic", ["chill", "epic"], 7),
    ("Midnight City", "M83", "electronic", ["dreamy", "night-drive"], 9),
    ("Heat Waves", "Glass Animals", "indie", ["chill", "dreamy"], 6),
    ("Bohemian Rhapsody", "Queen", "rock", ["epic", "classic"], 3),
]

if __name__ == "__main__":
    db.init_db()
    total = 0
    for title, artist, genre, tags, times in SAMPLE_LISTENS:
        for _ in range(times):
            db.log_listen(title, artist, genre=genre, tags=tags)
        total += times
    print(f"Seeded {len(SAMPLE_LISTENS)} songs, {total} total listens.")
    for s in db.top_songs(5):
        print(f"  {s['play_count']:>3}x  {s['title']} — {s['artist']}")
