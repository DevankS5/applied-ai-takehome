"""Turn PDF / HTML / Markdown files into plain text with page and heading markers.

Every loader returns a `Document` whose `text` is one string plus a list of `Span`s
mapping character offsets to a page number (PDF) and the nearest heading (HTML/MD),
so chunks can carry citation metadata no matter where the chunker cuts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}


@dataclass
class Span:
    start: int  # char offset where this span begins in Document.text
    page: int | None
    heading: str


@dataclass
class Document:
    doc_id: str
    source: str
    doc_type: str  # pdf | html | md
    title: str
    text: str
    spans: list[Span] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def locate(self, offset: int) -> Span:
        """Span (page/heading) that contains a character offset."""
        hit = self.spans[0] if self.spans else Span(0, None, "")
        for s in self.spans:
            if s.start > offset:
                break
            hit = s
        return hit


def clean(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate PDF line breaks
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, doc_id: str, title: str) -> Document:
    reader = PdfReader(str(path))
    parts, spans, pos = [], [], 0
    for i, page in enumerate(reader.pages, start=1):
        # Single newlines inside a PDF page are layout, not structure: join them.
        t = clean(page.extract_text() or "")
        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
        if not t:
            continue
        spans.append(Span(pos, i, ""))
        parts.append(t)
        pos += len(t) + 2
    return Document(doc_id, path.name, "pdf", title, "\n\n".join(parts), spans)


def load_html(path: Path, doc_id: str, title: str) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for m in soup.find_all("math"):  # keep formulas readable (MathML -> LaTeX alttext)
        m.replace_with(f" {m.get('alttext', '')} ")
    junk = ["script", "style", "nav", "footer", "sup", "link", "meta", "figure"]
    for tag in soup.find_all(junk):
        tag.decompose()
    for sel in [".reflist", ".references", ".navbox", ".infobox", ".mw-editsection", ".hatnote", ".metadata", ".shortdescription", ".sidebar", ".vertical-navbox"]:
        for tag in soup.select(sel):
            tag.decompose()
    root = soup.body or soup
    parts, spans, pos, heading = [], [], 0, title
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "dd", "pre", "td", "th", "blockquote"]):
        if el.find_parent(["li", "td", "th"]) and el.name in ("p", "li"):
            continue  # nested; parent already captured it
        t = clean(el.get_text(" ", strip=True))
        t = re.sub(r"\s+([.,;:!?)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", t))  # undo tag-join spacing
        if not t:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if t.lower() in {"references", "external links", "see also", "further reading", "notes"}:
                break  # back matter: citations and links, not content
            heading = t
            spans.append(Span(pos, None, heading))
        parts.append(t)
        pos += len(t) + 2
    if not spans or spans[0].start != 0:
        spans.insert(0, Span(0, None, title))
    return Document(doc_id, path.name, "html", title, "\n\n".join(parts), spans)


def load_md(path: Path, doc_id: str, title: str) -> Document:
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)  # images / badges
    raw = re.sub(r"\[\s*\]\([^)]*\)", "", raw)  # links left empty by the line above
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)  # links -> anchor text
    raw = re.sub(r"<[^>]+>", "", raw)  # inline HTML
    parts, spans, pos, heading = [], [Span(0, None, title)], 0, title
    for block in re.split(r"\n\s*\n", raw):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", block)
        if m:
            heading = m.group(2).strip()
            spans.append(Span(pos, None, heading))
        keeps_lines = block.startswith(("```", "-", "*", "|", "1."))  # code, lists, tables
        t = clean(block if keeps_lines else block.replace("\n", " "))
        parts.append(t)
        pos += len(t) + 2
    return Document(doc_id, path.name, "md", title, "\n\n".join(parts), spans)


LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html, ".md": load_md, ".markdown": load_md}


def load_file(path: Path, doc_id: str | None = None, title: str | None = None) -> Document:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return loader(path, doc_id or path.stem, title or path.stem.replace("-", " "))
