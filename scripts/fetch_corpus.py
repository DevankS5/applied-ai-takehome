"""Download the evaluation corpus listed in data/corpus/manifest.yaml.

Idempotent: files already on disk are skipped. A lock file records the sha256 of every
download so a later run can tell when an upstream document changed (which would move
the evidence quotes the eval set depends on).

    python scripts/fetch_corpus.py            # fetch missing files
    python scripts/fetch_corpus.py --verify   # also check hashes against the lock
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
MANIFEST = ROOT / "data/corpus/manifest.yaml"
RAW = ROOT / "data/corpus/raw"
LOCK = ROOT / "data/corpus/corpus.lock.json"
EXT = {"pdf": ".pdf", "html": ".html", "md": ".md"}
HEADERS = {"User-Agent": "applied-ai-takehome-corpus-fetcher/0.1 (research use)"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    docs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    RAW.mkdir(parents=True, exist_ok=True)
    changed = 0

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        for d in docs:
            path = RAW / f"{d['id']}{EXT[d['type']]}"
            if not path.exists():
                r = client.get(d["url"])
                r.raise_for_status()
                path.write_bytes(r.content)
                print(f"fetched  {path.name:<34} {len(r.content)/1024:8.1f} KB")
                time.sleep(1)  # be polite to arXiv / Wikipedia
            digest = sha256(path)
            if d["id"] in lock and lock[d["id"]] != digest:
                changed += 1
                print(f"CHANGED  {path.name} (hash differs from lock; eval evidence may have moved)")
            lock.setdefault(d["id"], digest)

    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"{len(docs)} documents in {RAW.relative_to(ROOT)}; {changed} changed vs lock")
    return 1 if (args.verify and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
