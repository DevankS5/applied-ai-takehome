"""Download the evaluation corpus listed in data/corpus/manifest.yaml.

Every URL in the manifest is immutable (arXiv version, Wikipedia revision id, GitHub commit
SHA). The lock file stores the sha256 of each document's EXTRACTED TEXT (what chunking and
the eval evidence quotes depend on), so cosmetic markup re-renders do not trip it but any
content change does.

    python scripts/fetch_corpus.py            # fetch missing files, add new lock entries
    python scripts/fetch_corpus.py --verify   # also fail (exit 1) on any mismatch or missing entry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.loaders import load_file  # noqa: E402

MANIFEST = ROOT / "data/corpus/manifest.yaml"
RAW = ROOT / "data/corpus/raw"
LOCK = ROOT / "data/corpus/corpus.lock.json"
EXT = {"pdf": ".pdf", "html": ".html", "md": ".md"}
# Wikimedia's robot policy asks for a contact URL in the User-Agent; shared cloud IPs get throttled without one.
HEADERS = {"User-Agent": "applied-ai-takehome-corpus-fetcher/0.3 (+https://github.com/DevankS5/applied-ai-takehome)"}


def get(client: httpx.Client, url: str, tries: int = 5) -> bytes:
    """GET with backoff on throttling (403/429) and server errors; arXiv and Wikipedia both throttle bursts."""
    for attempt in range(tries):
        r = client.get(url)
        if r.status_code not in (403, 429, 500, 502, 503, 504) or attempt == tries - 1:
            r.raise_for_status()
            return r.content
        time.sleep(float(r.headers.get("retry-after", 2 ** (attempt + 1))))
    raise AssertionError("unreachable")


def text_hash(path: Path) -> str:
    return hashlib.sha256(load_file(path).text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    docs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    RAW.mkdir(parents=True, exist_ok=True)
    problems = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        for d in docs:
            path = RAW / f"{d['id']}{EXT[d['type']]}"
            if not path.exists():
                content = get(client, d["url"])
                path.write_bytes(content)
                print(f"fetched  {path.name:<34} {len(content)/1024:8.1f} KB")
                time.sleep(1)  # be polite to arXiv / Wikipedia
            digest = text_hash(path)
            if d["id"] not in lock:
                if args.verify:
                    problems.append(f"{d['id']}: no lock entry")
                lock[d["id"]] = digest
            elif lock[d["id"]] != digest:
                problems.append(f"{d['id']}: extracted text differs from the lock")

    if not args.verify:
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    for p in problems:
        print(f"MISMATCH {p}")
    print(f"{len(docs)} documents in {RAW.relative_to(ROOT)}; {len(problems)} problem(s)")
    return 1 if (args.verify and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
