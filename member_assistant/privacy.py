"""Remove personal and identifying details from visitor text before it is stored.

Guardrail used by the interaction log and the escalation queue. It replaces:

- email addresses (except the organization's own published addresses) and phone numbers
- street addresses
- a visitor's own name when they introduce or sign themselves ("I'm Jane Doe", "Thanks, Jane")
- municipality names, which become a tag with the municipality's type and size band
  ("[municipality: small borough]"), so staff can still see which kinds of members
  need what without being able to tell which town asked.

Pattern-based redaction misses things (a name with no introduction, a town described
only by a landmark), so the log also stores no account, IP address, or identity, and
staff reports show aggregates.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?![\d-])")
STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\.?\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Pike|Highway|Hwy|Alley)\b\.?"
)
NAME_WORD = r"[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)?"
SELF_NAME_RE = re.compile(
    rf"(?P<lead>\b(?:[Mm]y name is|[Nn]ame:|[Tt]his is|I am|I'm|I’m)\s+)(?P<name>{NAME_WORD}(?:\s+{NAME_WORD}){{0,2}})"
)
SIGNOFF_RE = re.compile(
    rf"(?P<lead>\b(?i:best regards|kind regards|thanks|thank you|regards|best|sincerely|cheers)[,!.]?\s*[-–—]?\s*)"
    rf"(?P<name>{NAME_WORD}(?:\s+{NAME_WORD})?)\s*[.!]?\s*$"
)
# Words that follow "I'm" / "This is" but are roles or ordinary words, not names.
NOT_NAMES = {
    "Borough", "Township", "City", "Town", "County", "Manager", "Council", "Councilman", "Councilwoman",
    "Councilmember", "Member", "Mayor", "Chief", "Director", "Commissioner", "President", "Secretary",
    "Treasurer", "Officer", "Supervisor", "Solicitor", "Engineer", "Clerk", "Administrator", "Assistant",
    "Deputy", "Fire", "Police", "Emergency", "Resident", "Student", "Here", "Interested", "Looking",
    "Writing", "Calling", "Not", "Just", "Also", "The", "From", "With", "On", "In", "Hi", "Hello", "Sorry",
    "Curious", "Wondering", "New", "Urgent", "Regarding", "About", "A", "An", "My", "Our", "We", "Trying",
}

MUNICIPAL_TYPES = {"borough": "borough", "boro": "borough", "township": "township", "twp": "township",
                   "twp.": "township", "city": "city", "town": "town"}
CONTEXT_BEFORE_RE = re.compile(
    r"\b(?:of|from|in|for|at|near|represent|representing|serve|serving)\s+(?:the\s+)?$", re.IGNORECASE
)
PROTECTED_PHRASES = ("Carnegie Mellon",)
ABBREVIATIONS = [("Mount", ("Mt.", "Mt")), ("Saint", ("St.", "St"))]


@dataclass
class Redaction:
    text: str
    counts: Counter = field(default_factory=Counter)
    municipalities: list[dict] = field(default_factory=list)
    values: list[str] = field(default_factory=list)  # the original personal details that were removed


def scrub(text: str, values) -> str:
    """Remove personal details found earlier in the session (e.g. a name the visitor gave) from ``text``."""
    for value in sorted({v for v in values if v and len(v) >= 3}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w@.]){re.escape(value)}(?![\w@])", "[redacted]", text, flags=re.IGNORECASE)
    return text


def _variants(name: str) -> set[str]:
    """Spellings of a place name: Mt./Mt/Mount, St./St/Saint, straight and curly apostrophes."""
    forms = {name}
    for long, shorts in ABBREVIATIONS:
        for form in list(forms):
            for token in (long, *shorts):
                pattern = rf"\b{re.escape(token)}(?=\s)"
                if re.search(pattern, form):
                    forms |= {re.sub(pattern, repl, form) for repl in (long, *shorts)}
    forms |= {f.replace("'", "’") for f in forms} | {f.replace("’", "'") for f in forms}
    return forms


class Redactor:
    """Redacts personal details and replaces municipality names with type/size tags."""

    def __init__(self, municipalities: list[dict], keep_emails: set[str] = frozenset()):
        self.keep_emails = {e.lower() for e in keep_emails}
        groups: dict[str, list[dict]] = {}
        for m in municipalities:
            groups.setdefault(m["name"], []).append(m)
        types = r"borough|boro|township|twp\.?|city|town"
        self._groups = []
        # Longest names first so "East McKeesport" is replaced before "McKeesport".
        for name in sorted(groups, key=len, reverse=True):
            alts = "|".join(re.escape(v) for v in sorted(_variants(name), key=len, reverse=True))
            self._groups.append({
                "entries": groups[name],
                "ambiguous": any(e.get("ambiguous") for e in groups[name]),
                "pre": re.compile(rf"(?<![\w-])(?P<kind>city|borough|township|town)\s+of\s+(?:the\s+)?(?:{alts})(?![\w-])",
                                  re.IGNORECASE),
                "post": re.compile(rf"(?<![\w-])(?:{alts})\s+(?P<kind>{types})(?!\w)", re.IGNORECASE),
                "bare": re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])"),
            })

    @staticmethod
    def _tag(entries: list[dict], kind: str | None, counts: Counter, found: list[dict]) -> str:
        kind = MUNICIPAL_TYPES.get((kind or "").lower())
        matching = [e for e in entries if kind is None or e["type"] == kind] or entries
        bands = {e.get("size_band", "unknown size") for e in matching}
        types = {e["type"] for e in matching}
        profile = {
            "type": types.pop() if len(types) == 1 else (kind or "unknown"),
            "size_band": bands.pop() if len(bands) == 1 else "unknown size",
            "member": matching[0].get("is_member") if len(matching) == 1 else None,
        }
        counts["municipality"] += 1
        found.append(profile)
        size = profile["size_band"].split(" (")[0]
        words = [w for w in (size, profile["type"]) if w and w not in ("unknown", "unknown size")]
        return f"[municipality: {' '.join(words)}]" if words else "[municipality]"

    def redact(self, text: str) -> Redaction:
        counts, found, values = Counter(), [], []
        if not text:
            return Redaction(text or "", counts, found, values)

        protected = {}
        for i, phrase in enumerate(PROTECTED_PHRASES):
            if phrase in text:
                protected[f"\x00{i}\x00"] = phrase
                text = text.replace(phrase, f"\x00{i}\x00")

        def email_sub(m):
            if m.group(0).lower() in self.keep_emails:
                return m.group(0)
            counts["email"] += 1
            values.append(m.group(0))
            return "[email]"

        def pattern_sub(placeholder, key):
            def sub(m):
                counts[key] += 1
                values.append(m.group(0))
                return placeholder
            return sub

        text = EMAIL_RE.sub(email_sub, text)
        text = PHONE_RE.sub(pattern_sub("[phone]", "phone"), text)
        text = STREET_RE.sub(pattern_sub("[street address]", "street_address"), text)

        for group in self._groups:
            entries = group["entries"]

            def tag(m, kind, entries=entries):
                # Remember the name too, so a later bare mention ("Etna asked...") is scrubbed from stored text.
                values.extend([m.group(0), entries[0]["name"]])
                return self._tag(entries, kind, counts, found)

            text = group["pre"].sub(lambda m: tag(m, m["kind"]), text)
            text = group["post"].sub(lambda m: tag(m, m["kind"]), text)

            def bare_sub(m, ambiguous=group["ambiguous"]):
                # Names that are also ordinary words ("Moon", "Plum") count only after "of", "in", "from", ...
                if ambiguous and not CONTEXT_BEFORE_RE.search(m.string[max(0, m.start() - 30):m.start()]):
                    return m.group(0)
                return tag(m, None)

            text = group["bare"].sub(bare_sub, text)

        def name_sub(m):
            kept = []
            for word in m["name"].split():
                if word in NOT_NAMES:
                    break
                kept.append(word)
            if not kept:
                return m.group(0)
            counts["name"] += 1
            values.append(" ".join(kept))
            values.extend(word for word in kept if len(word) >= 3)
            return f"{m['lead']}[name]{m['name'][len(' '.join(kept)):]}"

        text = SELF_NAME_RE.sub(name_sub, text)
        text = SIGNOFF_RE.sub(name_sub, text)

        for placeholder, phrase in protected.items():
            text = text.replace(placeholder, phrase)
        return Redaction(text, counts, found, values)
