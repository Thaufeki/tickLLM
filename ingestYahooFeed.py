import feedparser # type: ignore
import time
import hashlib
from datetime import datetime, timezone
import pykx as kx # type: ignore

RSS_URL = "https://finance.yahoo.com/rss/topstories"

# tickerplant connection
conn = kx.SyncQConnection(host="localhost", port=5010)

TABLE_NAME = "news"   # logical table name in kdb+

def normalize_entry(entry):
    published = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        published = datetime(
            *entry.published_parsed[:6],
            tzinfo=timezone.utc
        )

    raw_id = f"{entry.get('title','')}{entry.get('link','')}"
    event_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

    return {
        "time": datetime.now(tz=timezone.utc),
        "sym":"",
        "event_id": event_id,
        "source": "yahoo_finance",
        "title": entry.get("title"),
        "summary": entry.get("summary"),
        "link": entry.get("link"),
        "published_ts": published
    }


def fetch_feed():
    feed = feedparser.parse(RSS_URL)
    if feed.bozo:
        raise RuntimeError(feed.bozo_exception)
    return feed.entries


def publish_to_kdb(record):
    """
    Publish one record to the tickerplant.
    This maps cleanly to .u.upd[`news; enlist record]
    """
    conn('.u.upd', TABLE_NAME, record)


def ingest_loop(poll_interval_seconds=1):
    seen_ids = set()

    while True:
        try:
            for entry in fetch_feed():
                record = normalize_entry(entry)

                if record["event_id"] in seen_ids:
                    continue

                seen_ids.add(record["event_id"])
                publish_to_kdb(list(record.values()))

                print(f"Published: {record['title']}")

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    ingest_loop()