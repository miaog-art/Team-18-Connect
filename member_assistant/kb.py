"""Local knowledge base: chunking at build time, BM25 retrieval at query time.

The corpus is small (a few dozen pages and reports), so a plain BM25 index in
memory is fast enough and needs no embedding service or vector database.
"""

import json
import math
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass

from . import config

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be been but by can could do does for from had has have how i if in "
    "into is it its me my of on or our so than that the their them there these they this to "
    "us was we were what when where which who why will with would you your about any".split()
)

CHUNK_CHARS = 1500  # ~350 tokens: big enough to hold a paragraph, small enough to cite precisely
MAX_HITS_PER_DOC = 2  # keep results from several sources instead of five passages of one report


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and a light plural strip."""
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def canonical(url: str) -> str:
    """Normalize a URL for comparison: https, lowercase host, no query/fragment, trailing slash on pages."""
    parts = urllib.parse.urlsplit(url.strip())
    path = parts.path or "/"
    if not re.search(r"\.[A-Za-z0-9]{2,5}$", path) and not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit(("https", parts.netloc.lower(), path, "", ""))


def _pieces(text: str, max_chars: int) -> list[str]:
    """Split text into pieces of at most ``max_chars``, breaking at lines, then sentences, then words."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            out.append(line)
            continue
        buf = ""
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(sentence[:cut])
                sentence = sentence[cut:].strip()
            if buf and len(buf) + len(sentence) + 1 > max_chars:
                out.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            out.append(buf)
    return out


def _pack(units: list[tuple[int | None, str]], max_chars: int) -> list[tuple[int | None, int | None, str]]:
    """Greedily pack (page, text) units into chunks; returns (first_page, last_page, text)."""
    chunks, buf, size, first, last = [], [], 0, None, None
    for page, text in units:
        if buf and size + len(text) + 1 > max_chars:
            chunks.append((first, last, "\n".join(buf)))
            buf, size, first = [], 0, None
        buf.append(text)
        size += len(text) + 1
        first = page if first is None else first
        last = page
    if buf:
        chunks.append((first, last, "\n".join(buf)))
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """Split one crawled document into retrieval chunks.

    Reports keep their PDF page numbers so a chunk can be cited as "p. N"
    (the PDF page index, which may differ from the number printed on the page).
    """
    if doc["kind"] == "report":
        units = [(n, piece) for n, page in enumerate(doc.get("pages", []), start=1)
                 for piece in _pieces(page, CHUNK_CHARS)]
    else:
        units = [(None, piece) for piece in _pieces(doc["text"], CHUNK_CHARS)]
    packed = _pack(units, CHUNK_CHARS)

    chunks = []
    for i, (first, last, text) in enumerate(packed):
        if doc["kind"] == "report":
            locator = f"p. {first}" if first == last else f"pp. {first}-{last}"
        else:
            locator = "" if len(packed) == 1 else f"part {i + 1} of {len(packed)}"
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{i}",
            "doc_id": doc["doc_id"],
            "kind": doc["kind"],
            "title": doc["title"],
            "url": doc["url"],
            "locator": locator,
            "text": text,
        })
    return chunks


@dataclass
class Hit:
    score: float
    chunk: dict


class KnowledgeBase:
    """BM25 index over the crawled chunks, plus lookup of whole documents.

    ``tier`` on each chunk comes from private/access_policy.json at load time,
    so the access model can change without re-crawling. The prototype assumes
    every document is public until the client decides on member-only content.
    """

    def __init__(self, documents: list[dict], chunks: list[dict], policy: dict | None = None):
        prefixes = tuple((policy or {}).get("member_only_url_prefixes", []))
        self.documents = {canonical(d["url"]): d for d in documents}
        for d in documents:
            for mirror in d.get("mirror_urls", []):
                self.documents.setdefault(canonical(mirror), d)
        self.chunks = chunks
        for c in chunks:
            c["tier"] = "member" if prefixes and c["url"].startswith(prefixes) else "public"

        # Title words are counted twice so a chunk from "Future of Fire" ranks for "fire".
        self._tf = [Counter(tokenize(c["title"]) * 2 + tokenize(c["text"])) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if chunks else 1.0
        df = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def load(cls) -> "KnowledgeBase":
        if not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"{config.CHUNKS_PATH} not found. Run `python -m member_assistant.crawl --site <url>` first."
            )

        def read(path):
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

        return cls(read(config.DOCUMENTS_PATH), read(config.CHUNKS_PATH),
                   config.load_json(config.ACCESS_POLICY_PATH, {}))

    def _score(self, i: int, q_terms: set[str]) -> float:
        k1, b = 1.5, 0.75
        tf = self._tf[i]
        score = 0.0
        for term in q_terms & tf.keys():
            f = tf[term]
            score += self._idf[term] * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / self._avg_len))
        return score

    def search(self, query: str, k: int = 5, kind: str = "any",
               allowed_tiers: tuple[str, ...] = ("public",), doc_id: str | None = None) -> list[Hit]:
        """Top ``k`` chunks for ``query`` (at most two per document); empty if no query term matches."""
        q_terms = set(tokenize(query))
        hits = []
        for i, chunk in enumerate(self.chunks):
            if chunk["tier"] not in allowed_tiers:
                continue
            if kind != "any" and chunk["kind"] != kind:
                continue
            if doc_id and chunk["doc_id"] != doc_id:
                continue
            score = self._score(i, q_terms)
            if score > 0:
                hits.append(Hit(score, chunk))
        hits.sort(key=lambda h: h.score, reverse=True)

        per_doc_limit = k if doc_id else MAX_HITS_PER_DOC
        picked, per_doc = [], Counter()
        for hit in hits:
            if per_doc[hit.chunk["doc_id"]] < per_doc_limit:
                picked.append(hit)
                per_doc[hit.chunk["doc_id"]] += 1
            if len(picked) == k:
                break
        return picked

    def get_document(self, url: str) -> dict | None:
        return self.documents.get(canonical(url))

    def chunks_for(self, doc_id: str, allowed_tiers: tuple[str, ...] = ("public",)) -> list[dict]:
        return [c for c in self.chunks if c["doc_id"] == doc_id and c["tier"] in allowed_tiers]
