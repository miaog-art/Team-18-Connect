"""Crawl the client's public website into a local knowledge base.

    python -m member_assistant.crawl --site https://<client-site> --county-wiki "<Wikipedia article title>"
    python -m member_assistant.crawl --rebuild      # re-chunk saved documents without network access

Later runs can omit --site/--county-wiki; they are saved in private/profile.json.
Everything is written under private/ (gitignored):

    kb/documents.jsonl      one record per page, post, or report PDF (duplicate files merged)
    kb/chunks.jsonl         retrieval chunks built from the documents
    contacts.json           contacts and the general inbox published on the site
    programs.json           priority programs and their categories
    municipalities.json     municipality names, types and size bands (used to anonymize logs)
    site_issues.json        problems found while crawling (broken links, mislinked files, ...)
    content_inventory.csv   catalogue of every source, for the team to review
"""

import argparse
import csv
import hashlib
import html
import io
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from . import config
from .kb import canonical, chunk_document, tokenize

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings on some PDFs are harmless

USER_AGENT = "Mozilla/5.0 (compatible; student-course-project-crawler/0.1; one request per second)"
DELAY_SECONDS = 1.0
MAX_DOWNLOAD_BYTES = 60_000_000
MAX_LINK_CHECKS = 80

DRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NAME_RE = re.compile(r"^(Dr\.\s)?[A-Z][\w’'.-]+(\s[A-Z][\w’'.-]+){1,3}$")
GENERIC_LINK_RE = re.compile(r"^\s*$|\b(download|click|here|view|read more|learn more)\b", re.I)
GENERIC_INBOXES = ("info", "contact", "hello", "office", "admin", "general")
BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "figure", "figcaption",
              "td", "th", "tr", "section", "article", "blockquote"]

# Size bands for anonymized municipality tags (population, U.S. Census).
SIZE_BANDS = [(5_000, "small (<5k)"), (20_000, "medium (5k-20k)"), (float("inf"), "large (20k+)")]


# ---------------------------------------------------------------- fetching

def fetch(url: str, timeout: int = 60) -> tuple[int, bytes]:
    """GET ``url`` after a polite delay. Returns (status, body); status 0 means a network error."""
    time.sleep(DELAY_SECONDS)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, b""


def slug_of(url: str) -> str:
    path = urllib.parse.urlsplit(url).path.strip("/")
    return path.split("/")[-1].replace("-", "_") if path else "home"


def is_generic_link_text(text: str) -> bool:
    return bool(GENERIC_LINK_RE.search(text)) or len(text) > 150


# ---------------------------------------------------------------- parsing

def read_sitemaps(site: str) -> list[dict]:
    """List page and post URLs from the WordPress/Yoast sitemap index (falls back to /sitemap.xml)."""
    status, body = fetch(urllib.parse.urljoin(site, "/sitemap_index.xml"))
    index = body.decode("utf-8", "ignore") if status == 200 else ""
    subs = re.findall(r"<loc>\s*(.*?)\s*</loc>", index) if "<sitemapindex" in index else []
    subs = subs or [urllib.parse.urljoin(site, "/sitemap.xml")]

    entries, seen = [], set()
    for sub in subs:
        name = sub.rsplit("/", 1)[-1]
        if any(skip in name for skip in ("category", "author", "tag")):
            continue
        status, body = fetch(sub)
        if status != 200:
            continue
        for block in re.findall(r"<url>(.*?)</url>", body.decode("utf-8", "ignore"), re.S):
            loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block)
            if not loc:
                continue
            url = canonical(html.unescape(loc.group(1)))
            if url in seen:
                continue
            seen.add(url)
            lastmod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block)
            is_post = "post" in name and urllib.parse.urlsplit(url).path != "/"
            entries.append({
                "url": url,
                "kind": "post" if is_post else "page",
                "lastmod": lastmod.group(1)[:10] if lastmod else None,
            })
    return entries


def _meta(soup: BeautifulSoup, key: str) -> str:
    tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
    return (tag.get("content") or "").strip() if tag else ""


def _link_labels(a) -> list[str]:
    """Candidate human labels for a link that only says "Download" or wraps an image."""
    labels = []
    file_block = a.find_parent(class_=re.compile(r"\bwp-block-file\b"))
    if file_block:
        prev = file_block.find_previous_sibling()
        if prev and prev.get_text(strip=True):
            labels.append(prev.get_text(" ", strip=True))
    column = a.find_parent(class_=re.compile(r"\bwp-block-column\b")) or a.find_parent(["figure", "li"])
    if column:
        text = re.sub(r"\s*\bDownload\b\s*", " ", column.get_text(" ", strip=True)).strip()
        if 0 < len(text) < 200:
            labels.append(text)
    return labels


def _block_text(node) -> str:
    """Visible text of ``node`` with one line per block element."""
    for br in node.find_all("br"):
        br.replace_with("\n")
    for tag in node.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = (re.sub(r"\s+", " ", line).strip() for line in node.get_text(" ").splitlines())
    text = "\n".join(line for line in lines if line)
    return re.sub(r" ([,.;:])", r"\1", text)


def parse_page(body: bytes, url: str) -> dict:
    """Extract title, main text (with link targets inlined), links, and email addresses from one page."""
    soup = BeautifulSoup(body, "html.parser")
    site_name = _meta(soup, "og:site_name")
    emails = [a["href"][7:].split("?")[0].strip().lower()
              for a in soup.find_all("a", href=True) if a["href"].lower().startswith("mailto:")]

    for tag in soup.find_all("footer"):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    for tag in main(["script", "style", "noscript", "svg", "form", "iframe", "header", "nav"]):
        tag.decompose()

    title = _meta(soup, "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    if site_name and title.endswith(f" - {site_name}"):
        title = title[: -len(site_name) - 3].strip()
    if not title or title.startswith("-") or title == site_name:
        heading = main.find(["h1", "h2"])
        title = heading.get_text(" ", strip=True) if heading else slug_of(url).replace("_", " ").title()

    anchors = list(main.find_all("a", href=True))
    links = []
    for a in anchors:  # collect every label before any link text is rewritten below
        links.append({
            "text": a.get_text(" ", strip=True),
            "href": urllib.parse.urljoin(url, a["href"].strip()),
            "labels": _link_labels(a),
        })
    for a, link in zip(anchors, links):
        # Inline the target so the assistant can hand visitors a working link from the retrieved text.
        text, href = link["text"], link["href"]
        if href.lower().startswith("mailto:"):
            address = href[7:].split("?")[0].strip()
            # When the visible text is itself an address, keep what visitors see; a mismatch is reported as a site issue.
            if text and not EMAIL_RE.search(text):
                a.string = f"{text} (email: {address})"
        elif text and not is_generic_link_text(text) and href.startswith("http"):
            a.string = f"{text} ({href})"

    return {
        "title": title,
        "site_name": site_name,
        "description": _meta(soup, "og:description") or _meta(soup, "description"),
        "text": _block_text(main),
        "links": links,
        "emails": emails,
    }


def extract_pdf(body: bytes) -> tuple[list[str], str]:
    """Return (text per page, embedded title) for a PDF; empty pages if the file is a scan."""
    try:
        reader = PdfReader(io.BytesIO(body))
        pages = [re.sub(r"[ \t]+", " ", page.extract_text() or "").strip() for page in reader.pages]
        meta_title = str((reader.metadata or {}).get("/Title") or "").strip()
    except (PdfReadError, ValueError, KeyError, TypeError) as e:
        print(f"    could not parse PDF: {e}")
        return [], ""
    return pages, meta_title


def best_title(candidates: list[str], first_pages: str) -> str:
    """Pick the candidate title whose words best match the opening pages of the file."""
    page_tokens = set(tokenize(first_pages))

    def overlap(title: str) -> float:
        tokens = set(tokenize(title))
        return len(tokens & page_tokens) / len(tokens) if tokens else 0.0

    return max(candidates, key=overlap)  # max() keeps the earliest candidate on ties


def remove_boilerplate(documents: list[dict], min_docs: int = 5) -> None:
    """Drop lines that repeat across many pages (share buttons, sidebars, newsletter prompts)."""
    web = [d for d in documents if d["kind"] != "report"]
    counts = Counter(line for d in web for line in set(d["text"].splitlines()))
    threshold = max(min_docs, len(web) // 4)
    repeated = {line for line, n in counts.items() if n >= threshold}
    for d in web:
        d["text"] = "\n".join(line for line in d["text"].splitlines() if line not in repeated)


def parse_contacts(pages: dict[str, dict], staff_url: str) -> tuple[list[dict], list[str], str | None]:
    """Contacts published on the site, names of people on the staff/board page, and the general inbox.

    Only addresses on the organization's own domain are kept, so the assistant never
    refers visitors to a partner organization's staff by mistake.
    """
    org_domain = urllib.parse.urlsplit(staff_url).netloc.lower().removeprefix("www.")
    contacts = {}

    def add(email, name, title, url, context):
        if not email.endswith("@" + org_domain):
            return
        entry = contacts.setdefault(email, {"email": email, "name": None, "title": "", "source_urls": [],
                                            "contexts": []})
        entry["name"] = entry["name"] or (name if name and NAME_RE.match(name) else None)
        entry["title"] = entry["title"] or title
        context = re.sub(r"\s*\(email: [^)]*\)", "", context).strip()[:200]
        if context and context not in entry["contexts"]:
            entry["contexts"].append(context)
        if url not in entry["source_urls"]:
            entry["source_urls"].append(url)

    for url, page in pages.items():
        lines = page["text"].splitlines()
        for i, line in enumerate(lines):
            for email in EMAIL_RE.findall(line):
                email = email.lower()
                # "Name, Title – email" on one line
                inline = re.match(r"^(?P<name>[^,–-]+),\s*(?P<title>.+?)\s*[–-]\s*\S+@", line)
                if inline:
                    add(email, inline["name"].strip(), inline["title"].strip(), url, line)
                elif line.strip() == email and i >= 2:
                    # Staff page layout: name line, title line, email line
                    add(email, lines[i - 2].rstrip(",").strip(), lines[i - 1].strip(), url, "")
                elif m := re.search(r"(?P<name>[A-Z][a-z]+(?:\s[A-Z][a-z]+)+),\s*(?P<title>[^()]{3,80}?)\s*"
                                    r"\(email:\s*" + re.escape(email), line):
                    # "... please email Jane Doe, Director of X (email: jane@...)"
                    add(email, m["name"], m["title"].strip(), url, line)
                else:
                    names = re.findall(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+),\s*" + re.escape(email), line)
                    add(email, names[0] if names else None, "", url, line)
        # Labelled mailto links ("Get in contact with Infrastructure & Resilience ...") route by topic.
        for link in page["links"]:
            if link["href"].lower().startswith("mailto:"):
                email = link["href"][7:].split("?")[0].strip().lower()
                if link["text"] and not EMAIL_RE.fullmatch(link["text"]):
                    add(email, None, "", url, link["text"])

    staff_text = pages.get(staff_url, {}).get("text", "").lower()
    for c in contacts.values():
        c["listed_on_staff_page"] = c["email"] in staff_text

    staff_lines = pages.get(staff_url, {}).get("text", "").splitlines()
    people = sorted({ln.split(",")[0].strip() for ln in staff_lines
                     if "," in ln and NAME_RE.match(ln.split(",")[0].strip())})

    all_emails = [e for p in pages.values() for e in p["emails"]] + list(contacts)
    generic = [e for e in all_emails if e.split("@")[0] in GENERIC_INBOXES and e.endswith("@" + org_domain)]
    general = Counter(generic).most_common(1)[0][0] if generic else None
    return sorted(contacts.values(), key=lambda c: c["email"]), people, general


def parse_programs(page: dict, url: str) -> list[dict]:
    """Programs listed on the priorities page, grouped under the upper-case category headings."""
    host = urllib.parse.urlsplit(url).netloc
    targets = {}
    for link in page["links"]:
        if not is_generic_link_text(link["text"]) and urllib.parse.urlsplit(link["href"]).netloc == host:
            targets.setdefault(link["text"], canonical(link["href"]))

    programs, category = [], "General"
    for line in page["text"].splitlines():
        label = re.sub(r"\s*\(https?://\S+\)$", "", line).strip()
        if label.isupper() and len(label) > 3:
            category = label.title().replace("&", "and")
        elif label in targets and targets[label] != canonical(url):
            if all(p["name"] != label for p in programs):
                programs.append({"id": slug_of(targets[label]), "name": label,
                                 "url": targets[label], "category": category})
    return programs


def _wiki_name(link_markup: str) -> str:
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", link_markup)
    return (m.group(2) or m.group(1).split(",")[0]).strip() if m else ""


def fetch_municipalities(wiki_title: str) -> list[dict]:
    """Municipality names, types and populations from a Wikipedia county article.

    Names and types come from the Cities/Boroughs/Townships/Town lists; population
    comes from the article's census ranking table when it lists the place.
    """
    query = urllib.parse.urlencode({"action": "parse", "page": wiki_title, "prop": "wikitext",
                                    "format": "json", "formatversion": "2", "redirects": "1"})
    status, body = fetch(f"https://en.wikipedia.org/w/api.php?{query}")
    if status != 200:
        print(f"  municipality list unavailable (HTTP {status})")
        return []
    wikitext = json.loads(body).get("parse", {}).get("wikitext", "")

    municipalities = []
    for heading, kind in [("Cities", "city"), ("Boroughs", "borough"), ("Townships", "township"),
                          ("Towns?", "town")]:
        section = re.search(rf"^===\s*{heading}\s*===\s*$(.*?)(?=^==)", wikitext, re.M | re.S)
        if not section:
            continue
        for line in re.findall(r"^\*\s*(\[\[.*)$", section.group(1), re.M):
            name = _wiki_name(line)
            if name:
                municipalities.append({"name": name, "type": kind, "population": None})

    ranking = re.search(r"^===\s*Population ranking\s*===\s*$(.*?)(?=^==[^=])", wikitext, re.M | re.S)
    population = {}
    if ranking:
        for row in ranking.group(1).split("\n|-")[1:]:
            cells = [c.strip() for c in re.split(r"\n\|", "\n" + row.split("\n", 1)[-1]) if c.strip()]
            if len(cells) >= 3:
                name, digits = _wiki_name(cells[1]), re.sub(r"\D", "", cells[2])
                if name and digits:
                    population[normalize_place(name)] = int(digits)
    common_words = load_common_words()
    for m in municipalities:
        m["population"] = population.get(normalize_place(m["name"]))
        m["size_band"] = size_band(m["population"])
        # Names that are also ordinary words ("Moon", "Forward") or the region's anchor city are only
        # redacted when the text makes clear a municipality is meant (see privacy.py).
        lowered = m["name"].lower()
        m["ambiguous"] = (lowered in common_words if common_words else " " not in lowered) \
            or (m["population"] or 0) >= 100_000
    return municipalities


def load_common_words() -> set[str]:
    """Lowercased English words from the system dictionary (macOS/Linux); empty set if unavailable."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip()}
        except OSError:
            continue
    return set()


def normalize_place(name: str) -> str:
    """Lowercase name without municipal suffixes, with "Mt." spelled out."""
    name = name.lower().replace("’", "'")
    name = re.sub(r"\bmt\.?(?=\s)", "mount", name)
    name = re.sub(r"\b(city of|borough of|township of|township|borough|twp\.?|city)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def size_band(population: int | None) -> str:
    if population is None:
        return "unknown size"
    return next(label for limit, label in SIZE_BANDS if population < limit)


def mark_members(municipalities: list[dict], member_center_text: str) -> int:
    """Flag municipalities that appear in the member list on the member page. Returns the match count."""
    lines = member_center_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "MEMBERS" in ln and ln.isupper()), None)
    if start is None:
        return 0
    listed = []
    for line in lines[start + 1:]:
        if len(line) > 40 or line.isupper():
            break
        listed.append(line)
    by_name = {}
    for m in municipalities:
        by_name.setdefault(normalize_place(m["name"]), []).append(m)
    matched = 0
    for entry in listed:
        candidates = by_name.get(normalize_place(entry), [])
        typed = [m for m in candidates if m["type"] in entry.lower()]
        for m in typed or candidates[:1]:
            m["is_member"] = True
            matched += 1
    for m in municipalities:
        m.setdefault("is_member", False)
    return matched


# ---------------------------------------------------------------- crawl

def crawl(site: str, county_wiki: str | None, staff_path: str, programs_path: str,
          members_path: str) -> None:
    site = canonical(site)
    host = urllib.parse.urlsplit(site).netloc
    for d in (config.KB_DIR, config.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Reading sitemap of {site}")
    entries = read_sitemaps(site)
    print(f"  {len(entries)} pages and posts")

    documents, parsed, issues = [], {}, []
    for entry in entries:
        status, body = fetch(entry["url"])
        if status != 200:
            issues.append({"type": "page_unavailable", "url": entry["url"], "detail": f"HTTP {status}"})
            continue
        page = parse_page(body, entry["url"])
        parsed[entry["url"]] = page
        documents.append({
            "doc_id": f"{entry['kind']}:{slug_of(entry['url'])}",
            "kind": entry["kind"],
            "title": page["title"],
            "url": entry["url"],
            "lastmod": entry["lastmod"],
            "text": page["text"],
        })
        print(f"  {entry['kind']:<5} {page['title'][:60]:<60} {len(page['text']):>6} chars")
    remove_boilerplate(documents)

    # Reports: every Drive file or PDF linked from any page, keyed by file.
    reports = {}
    for page_url, page in parsed.items():
        for link in page["links"]:
            href = link["href"]
            drive = DRIVE_FILE_RE.search(href)
            if drive:
                key = drive.group(1)
                view_url = f"https://drive.google.com/file/d/{key}/view"
                download_url = f"https://drive.usercontent.google.com/download?id={key}&export=download&confirm=t"
            elif urllib.parse.urlsplit(href).path.lower().endswith(".pdf"):
                key = view_url = download_url = href
            else:
                continue
            report = reports.setdefault(key, {"url": view_url, "download_url": download_url, "labels": []})
            labels = ([] if is_generic_link_text(link["text"]) else [link["text"]]) + link["labels"]
            for label in labels:
                if (label, page_url) not in report["labels"]:
                    report["labels"].append((label, page_url))
            if not labels:
                report["labels"].append(("", page_url))

    print(f"Downloading {len(reports)} linked report files")
    by_hash = {}
    for key, report in reports.items():
        found_on = sorted({page for _, page in report["labels"]})
        titles = list(dict.fromkeys(label for label, _ in report["labels"] if label))
        status, body = fetch(report["download_url"], timeout=180)
        if status != 200 or not body.startswith(b"%PDF"):
            reason = ("server not reachable (the domain may no longer exist)" if status == 0
                      else f"HTTP {status}" if status != 200 else "the link does not open a PDF")
            issues.append({"type": "report_unavailable", "url": report["url"], "titles": titles,
                           "found_on": found_on, "detail": reason})
            print(f"  unavailable: {report['url']} ({reason})")
            continue
        pages, meta_title = extract_pdf(body)
        digest = hashlib.sha256("\n".join(pages).encode()).hexdigest() if any(pages) else key

        if digest in by_hash:  # the same file hosted at another URL: merge
            doc = by_hash[digest]
            doc["mirror_urls"].append(report["url"])
            doc["found_on"] = sorted(set(doc["found_on"]) | set(found_on))
            doc["_labels"] += report["labels"]
            print(f"  duplicate of '{doc['title'][:50]}': {report['url']}")
            continue

        opening = " ".join(pages[:2])
        candidates = titles + ([meta_title] if meta_title else [])
        title = best_title(candidates, opening) if candidates else f"Report {key[:8]}"
        doc = {
            "doc_id": f"report:{hashlib.sha1(key.encode()).hexdigest()[:10]}",
            "kind": "report",
            "title": title,
            "url": report["url"],
            "mirror_urls": [],
            "lastmod": None,
            "pages": pages,
            "text": "",
            "found_on": found_on,
            "_labels": list(report["labels"]),
        }
        by_hash[digest] = doc
        documents.append(doc)
        if not any(pages):
            issues.append({"type": "report_without_text", "url": report["url"], "titles": titles,
                           "found_on": found_on,
                           "detail": "No extractable text (likely a scanned image): not searchable and hard to "
                                     "read with a screen reader."})
        print(f"  report {title[:60]:<60} {len(pages):>4} pages")

    # A link whose label has nothing in common with the file it opens probably points to the wrong file.
    for doc in by_hash.values():
        opening = set(tokenize(doc["title"] + " " + " ".join(doc["pages"][:2])))
        for label, page_url in doc.pop("_labels"):
            label_tokens = set(tokenize(label))
            # Long labels are surrounding paragraphs, not link titles; comparing them gives false alarms.
            if label and len(label) <= 100 and label_tokens and len(label_tokens & opening) / len(label_tokens) < 0.3:
                issues.append({"type": "link_opens_wrong_file", "url": doc["url"], "found_on": [page_url],
                               "detail": f"The link labelled '{label}' opens '{doc['title']}'."})

    # Mismatched email links and broken internal links.
    known_urls = {e["url"] for e in entries}
    to_check = {}
    for page_url, page in parsed.items():
        for link in page["links"]:
            href = link["href"]
            if href.lower().startswith("mailto:"):
                target = href[7:].split("?")[0].lower()
                shown = EMAIL_RE.findall(link["text"])
                if shown and shown[0].lower() != target:
                    issues.append({"type": "mailto_mismatch", "url": page_url, "found_on": [page_url],
                                   "detail": f"Link text shows {shown[0]} but the link opens an email to {target}."})
                continue
            parts = urllib.parse.urlsplit(href)
            if parts.netloc == host and "/wp-content/" not in parts.path and canonical(href) not in known_urls:
                info = to_check.setdefault(canonical(href), {"found_on": set(), "text": link["text"]})
                info["found_on"].add(page_url)
    for url, info in list(to_check.items())[:MAX_LINK_CHECKS]:
        status, _ = fetch(url, timeout=30)
        if status >= 400 or status == 0:
            issues.append({"type": "broken_link", "url": url, "found_on": sorted(info["found_on"]),
                           "detail": f"The link '{info['text'] or url}' returns HTTP {status}."})

    # Structured lookups used by the tools.
    staff_url = canonical(urllib.parse.urljoin(site, staff_path))
    programs_url = canonical(urllib.parse.urljoin(site, programs_path))
    members_url = canonical(urllib.parse.urljoin(site, members_path))
    contacts, people, general_email = parse_contacts(parsed, staff_url)
    for c in contacts:
        if not c["listed_on_staff_page"] and c["email"] != general_email:
            issues.append({"type": "contact_not_on_staff_page", "url": c["source_urls"][0],
                           "found_on": c["source_urls"],
                           "detail": f"{c['email']} is published on the site but not listed on the staff page; "
                                     "confirm it is still monitored."})
    programs = parse_programs(parsed[programs_url], programs_url) if programs_url in parsed else []
    # A program's contacts are the organization addresses linked from its own page ("Get in Touch").
    org_emails = {c["email"] for c in contacts}
    for program in programs:
        page = parsed.get(program["url"], {})
        emails = [link["href"][7:].split("?")[0].strip().lower() for link in page.get("links", [])
                  if link["href"].lower().startswith("mailto:")]
        emails += [e.lower() for e in EMAIL_RE.findall(page.get("text", ""))]
        program["contacts"] = [e for e in dict.fromkeys(emails) if e in org_emails and e != general_email]
    home = parsed.get(site) or next(iter(parsed.values()))

    municipalities = fetch_municipalities(county_wiki) if county_wiki else []
    members = mark_members(municipalities, parsed.get(members_url, {}).get("text", "")) if municipalities else 0

    write_jsonl(config.DOCUMENTS_PATH, documents)
    write_json(config.CONTACTS_PATH, {"contacts": contacts, "general_email": general_email,
                                      "contact_page": staff_url})
    write_json(config.PROGRAMS_PATH, programs)
    write_json(config.MUNICIPALITIES_PATH, municipalities)
    write_json(config.SITE_ISSUES_PATH, issues)
    write_json(config.PROFILE_PATH, {
        "org_name": home["site_name"] or host,
        "org_description": home["description"],
        "site_url": site,
        "general_email": general_email,
        "contact_page": staff_url,
        "known_people": people,
        "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crawl_args": {"site": site, "county_wiki": county_wiki, "staff_path": staff_path,
                       "programs_path": programs_path, "members_path": members_path},
    })
    if not config.ACCESS_POLICY_PATH.exists():
        # Prototype assumption pending the client's answer on registration: everything is public.
        write_json(config.ACCESS_POLICY_PATH, {"mode": "all_public", "member_only_url_prefixes": []})
    n_chunks = rebuild_chunks()

    kinds = Counter(d["kind"] for d in documents)
    print(f"\nDone: {dict(kinds)} -> {n_chunks} chunks; {len(contacts)} contacts (general inbox: {general_email}), "
          f"{len(programs)} programs, {len(municipalities)} municipalities ({members} on the member list), "
          f"{len(issues)} site issues. Output in {config.PRIVATE_DIR}")


def rebuild_chunks() -> int:
    """Re-chunk documents.jsonl and rewrite chunks.jsonl and the content inventory."""
    documents = [json.loads(line) for line in config.DOCUMENTS_PATH.read_text(encoding="utf-8").splitlines() if line]
    chunks = [c for doc in documents for c in chunk_document(doc)]
    write_jsonl(config.CHUNKS_PATH, chunks)
    write_inventory(documents, config.load_json(config.PROGRAMS_PATH, []))
    return len(chunks)


def write_inventory(documents: list[dict], programs: list[dict]) -> None:
    """Content inventory for the scope deliverable; 'topic' is filled where a program page matches."""
    program_by_url = {p["url"]: p for p in programs}
    with open(config.PRIVATE_DIR / "content_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "format", "url", "other_urls", "last_modified", "size", "topic", "linked_from"])
        for d in documents:
            size = f"{len(d['pages'])} pages" if d["kind"] == "report" else f"{len(d['text'].split())} words"
            program = program_by_url.get(d["url"])
            topic = f"{program['category']} / {program['name']}" if program else ""
            w.writerow([d["title"], d["kind"], d["url"], "; ".join(d.get("mirror_urls", [])),
                        d.get("lastmod") or "", size, topic, "; ".join(d.get("found_on", []))])


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", help="Root URL of the client website")
    parser.add_argument("--county-wiki", help="Wikipedia article listing the region's municipalities")
    parser.add_argument("--staff-path", default="/our-board-and-staff/")
    parser.add_argument("--programs-path", default="/priorities/")
    parser.add_argument("--members-path", default="/member-center/")
    parser.add_argument("--rebuild", action="store_true", help="Only re-chunk the saved documents (no network)")
    args = parser.parse_args()

    if args.rebuild:
        print(f"Rebuilt {rebuild_chunks()} chunks")
        return
    saved = (config.load_json(config.PROFILE_PATH) or {}).get("crawl_args", {})
    site = args.site or saved.get("site")
    if not site:
        parser.error("--site is required on the first run")

    def arg(name):
        value = getattr(args, name)
        return value if args.site or not saved.get(name) else saved[name]

    crawl(site, args.county_wiki or saved.get("county_wiki"),
          arg("staff_path"), arg("programs_path"), arg("members_path"))


if __name__ == "__main__":
    main()
