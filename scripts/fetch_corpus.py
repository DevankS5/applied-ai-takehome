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
HEADERS = {"User-Agent": "applied-ai-takehome-corpus-fetcher/0.2 (research use)"}


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
                r = client.get(d["url"])
                r.raise_for_status()
                path.write_bytes(r.content)
                print(f"fetched  {path.name:<34} {len(r.content)/1024:8.1f} KB")
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
