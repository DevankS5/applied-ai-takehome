"""Idempotent ingestion: folder of PDF/HTML/MD -> chunks -> embeddings -> LanceDB.

    python -m rag.ingest data/corpus/raw                       # defaults: 400 tokens, 60 overlap
    python -m rag.ingest data/corpus/raw --chunk-size 200 --overlap 40 --table chunks_c200

Re-running on the same corpus changes nothing: chunk ids are content hashes, chunks
already in the table are not re-embedded, and writes go through merge_insert on chunk_id.
An edited document only re-embeds its changed chunks; its old chunks are deleted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from rag.chunking import chunk_document
from rag.config import get_settings
from rag.embed import embed_passages
from rag.loaders import LOADERS, load_file
from rag.store import Store

MANIFEST = Path("data/corpus/manifest.yaml")


def embed_text(title: str, heading: str, text: str) -> str:
    """Contextual header: the title/section a chunk came from is embedded with it."""
    head = title if not heading or heading == title else f"{title}. {heading}"
    return f"{head}\n{text}"


def ingest(folder: Path, chunk_size: int, overlap: int, table: str | None = None, prune: bool = False) -> dict:
    s = get_settings()
    store = Store(table=table)
    titles = {}
    if MANIFEST.exists():
        titles = {d["id"]: d["title"] for d in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]}
    config_tag = f"{s.embed_model}|{chunk_size}|{overlap}"
    t0 = time.perf_counter()
    before = store.count()
    stats = {"docs": 0, "chunks": 0, "embedded": 0, "unchanged": 0, "deleted": 0}
    seen_docs = set()

    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in LOADERS):
        doc = load_file(path, doc_id=path.stem, title=titles.get(path.stem))
        chunks = chunk_document(doc, chunk_size, overlap, config_tag)
        existing = store.chunk_ids_for(doc.doc_id)
        new = [c for c in chunks if c.chunk_id not in existing]
        vecs = embed_passages([embed_text(doc.title, c.heading, c.text) for c in new]) if new else []
        now = time.time()
        store.upsert(
            [
                {
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": doc.source, "doc_type": doc.doc_type,
                    "title": doc.title, "heading": c.heading, "page": c.page, "chunk_index": c.chunk_index,
                    "n_tokens": c.n_tokens, "text": c.text, "ingested_at": now, "vector": v.tolist(),
                }
                for c, v in zip(new, vecs)
            ]
        )
        stats["deleted"] += store.delete_except(doc.doc_id, {c.chunk_id for c in chunks})
        stats["docs"] += 1
        stats["chunks"] += len(chunks)
        stats["embedded"] += len(new)
        stats["unchanged"] += len(chunks) - len(new)
        seen_docs.add(doc.doc_id)

    if prune:
        stats["deleted"] += store.delete_docs_not_in(seen_docs)
    stats.update(
        table=store.name, chunk_size=chunk_size, overlap=overlap, embed_model=s.embed_model, embed_dim=s.embed_dim,
        vectors_before=before, vectors_after=store.count(), seconds=round(time.perf_counter() - t0, 2),
    )
    return stats


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--table", default=None)
    ap.add_argument("--prune", action="store_true", help="delete documents no longer in the folder")
    a = ap.parse_args()
    stats = ingest(a.folder, a.chunk_size, a.overlap, a.table, a.prune)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
