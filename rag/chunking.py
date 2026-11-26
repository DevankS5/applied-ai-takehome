"""Token-aware recursive chunker.

Splits on paragraph, then sentence boundaries, packs sentences greedily up to
`chunk_size` tokens and carries the last sentences (up to `overlap` tokens) into
the next chunk. Sizes are measured with the embedding model's own tokenizer.
A sentence longer than `chunk_size` is cut at token boundaries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag.embed import count_tokens, tokenizer
from rag.loaders import Document

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    n_tokens: int
    start: int  # char offsets into Document.text
    end: int
    page: int | None
    heading: str


def _units(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of sentences, never crossing a paragraph break."""
    spans = []
    for para in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        p0, body, cursor = para.start(), para.group(), 0
        for m in SENT_END.finditer(body):
            spans.append((p0 + cursor, p0 + m.start()))
            cursor = m.end()
        spans.append((p0 + cursor, p0 + len(body)))
    return [s for s in spans if s[1] > s[0]]


def _split_long(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    enc = tokenizer().encode(text[start:end], add_special_tokens=False)
    offs = enc.offsets
    return [(start + offs[i][0], start + offs[min(i + size, len(offs)) - 1][1]) for i in range(0, len(offs), size)]


def chunk_document(doc: Document, chunk_size: int, overlap: int, config_tag: str) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    units = _units(doc.text)
    sizes = count_tokens([doc.text[a:b] for a, b in units])
    # Break any oversized sentence into token windows.
    flat: list[tuple[int, int, int]] = []
    for (a, b), n in zip(units, sizes):
        if n <= chunk_size:
            flat.append((a, b, n))
        else:
            pieces = _split_long(doc.text, a, b, chunk_size)
            flat += [(x, y, n2) for (x, y), n2 in zip(pieces, count_tokens([doc.text[x:y] for x, y in pieces]))]

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    i = 0
    while i < len(flat):
        j, total = i, 0
        while j < len(flat) and total + flat[j][2] <= chunk_size:
            total += flat[j][2]
            j += 1
        j = max(j, i + 1)  # always make progress
        a, b = flat[i][0], flat[j - 1][1]
        text = doc.text[a:b].strip()
        digest = hashlib.sha256(f"{doc.doc_id}|{config_tag}|{text}".encode()).hexdigest()[:24]
        seen[digest] = seen.get(digest, 0) + 1  # identical text twice in one doc stays distinct
        cid = digest if seen[digest] == 1 else f"{digest}-{seen[digest]}"
        span = doc.locate(a)
        chunks.append(Chunk(cid, doc.doc_id, len(chunks), text, sum(f[2] for f in flat[i:j]), a, b, span.page, span.heading))
        if j >= len(flat):
            break
        # Overlap: step back over trailing sentences worth <= `overlap` tokens.
        k, back = j, 0
        while k - 1 > i and back + flat[k - 1][2] <= overlap:
            k -= 1
            back += flat[k][2]
        i = k
    return chunks
