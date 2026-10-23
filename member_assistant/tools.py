"""The custom tools the assistant can call.

The tool descriptions are written for the model: what the tool does, when to call
it, and what comes back. All schemas use strict mode, so arguments always match
the schema, and fixed vocabularies (topics, roles, outcomes) are enums, which keeps
the needs log consistent enough to aggregate.
"""

import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .kb import KnowledgeBase, tokenize
from .privacy import Redactor, scrub

VISITOR_ROLES = ["elected_official", "municipal_staff", "public_safety", "council_of_governments",
                 "partner_organization", "resident", "student_or_researcher", "vendor_or_consultant",
                 "media", "unknown"]

OUTCOMES = {
    "answered": "the reply answers the question from published material",
    "referred_to_resource": "the reply points to a page, report, or form that covers it",
    "referred_to_staff": "the reply gives a staff contact because the material doesn't cover it",
    "escalated": "escalate_to_staff was called",
    "no_answer_found": "nothing published answers it and no specific contact fits",
    "out_of_scope": "unrelated to the organization's work",
    "safety_redirect": "emergency or crisis redirected to 911/988",
}

GENERAL_TOPICS = {
    "membership": "joining, dues, member benefits, who the members are",
    "events_and_meetings": "events, member meetings, minutes, presentations to councils",
    "reports_and_research": "research reports not tied to one program",
    "governance": "bylaws, resolutions, board, organizational history",
    "jobs_and_procurement": "job openings, internships, RFPs",
    "partnership_and_sponsorship": "partnering, sponsoring, collaborating as an organization",
    "contact_request": "reaching a specific person or the organization",
    "website_feedback": "comments about the website itself",
    "other": "anything else",
}

# Words that appear in almost every routing request and say nothing about who should handle it.
GENERIC_REQUEST_WORDS = set(tokenize(
    "staff contact contacted talk email call follow up program municipality borough township city town manager "
    "official want asked ask question help join joining member information learn more need request visitor "
    "someone get touch work director"))

STAGE_RE = re.compile(r"STAGE OF ENGAGEMENT:\s*([A-Z ]+)")
INLINE_LINK_RE = re.compile(r"\s*\((?:https?://|email: )[^)]*\)")


class ToolError(Exception):
    """A problem the model can fix (bad URL, missing argument); returned as an is_error tool result."""


@dataclass
class TurnState:
    """What the tools recorded while handling one visitor message."""
    need: dict | None = None
    escalations: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)


def _text_blocks(text: str, max_chars: int = 400) -> list[dict]:
    """Paragraph-sized text blocks: the unit Claude cites, so smaller blocks give tighter citations."""
    blocks, buf = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if buf and len(buf) + len(line) + 1 > max_chars:
            blocks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        blocks.append(buf)
    return [{"type": "text", "text": b} for b in blocks] or [{"type": "text", "text": "(no text)"}]


def search_result_block(chunk: dict) -> dict:
    title = chunk["title"] + (f" ({chunk['locator']})" if chunk["locator"] else "")
    return {
        "type": "search_result",
        "source": chunk["url"],
        "title": title,
        "content": _text_blocks(chunk["text"]),
        "citations": {"enabled": True},
    }


class Toolbox:
    def __init__(self, kb: KnowledgeBase, profile: dict, contacts: dict, programs: list[dict],
                 redactor: Redactor, escalations_path: Path, store_raw: bool = False,
                 allowed_tiers: tuple[str, ...] = ("public",)):
        self.kb = kb
        self.profile = profile
        self.contacts = contacts.get("contacts", [])
        self.general_email = contacts.get("general_email")
        self.contact_page = contacts.get("contact_page")
        self.programs = programs
        self.redactor = redactor
        self.escalations_path = escalations_path
        self.store_raw = store_raw
        self.allowed_tiers = allowed_tiers
        self.org_words = set(tokenize(profile["org_name"]))  # the org's own name matches nearly every contact
        self.topics = [p["id"] for p in programs] + list(GENERAL_TOPICS)
        self.sensitive: set[str] = set()  # personal details the visitor gave this session (set by Assistant)
        self.program_info = [self._describe_program(p) for p in programs]

    # ------------------------------------------------------------ definitions

    def definitions(self) -> list[dict]:
        org = self.profile["org_name"]
        program_topics = "; ".join(f"{p['id']} = {p['name']}" for p in self.programs)
        general_topics = "; ".join(f"{k} = {v}" for k, v in GENERAL_TOPICS.items())
        categories = sorted({p["category"] for p in self.programs})
        topic_schema = {
            "type": "string",
            "enum": self.topics,
            "description": f"Closest topic. Programs: {program_topics}. General: {general_topics}.",
        }
        outcomes = "; ".join(f"{k} = {v}" for k, v in OUTCOMES.items())

        return [
            {
                "name": "search_knowledge_base",
                "description": (
                    f"Search {org}'s published website pages, blog posts, and report PDFs. Call this before "
                    f"answering any question about {org}'s programs, research, membership, events, governance, "
                    "or staff, and search again with different wording if the first results don't answer the "
                    "question. Returns up to 5 passages as citable search results, each with its source URL "
                    "(and PDF page numbers for reports). If nothing matches, the topic isn't covered in the "
                    "published material: don't answer it from memory."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Keywords or a short question, e.g. 'volunteer fire department "
                                                 "staffing' or 'how does a municipality become a member'."},
                        "source_type": {"type": "string", "enum": ["any", "page", "post", "report"],
                                        "description": "'report' for research reports and other PDFs, 'page' for "
                                                       "program and information pages, 'post' for news and blog "
                                                       "posts, 'any' when unsure."},
                    },
                    "required": ["query", "source_type"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_document",
                "description": (
                    "Read more of one page or report that a search or list_programs already returned. Pass the "
                    "URL exactly as it appeared there and what you are looking for; returns the most relevant "
                    "passages (up to 4) from that document as citable search results. Use it to find details "
                    "inside a long report, such as findings, recommendations, or figures."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Source URL from an earlier tool result."},
                        "looking_for": {"type": "string",
                                        "description": "What to find in the document, e.g. 'recommendations' or "
                                                       "'number of departments surveyed'."},
                    },
                    "required": ["url", "looking_for"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_programs",
                "description": (
                    f"List {org}'s programs (its current priorities) with each program's category, stage, "
                    "one-line summary, page link, and the published staff contact for it. Call this when a "
                    f"visitor asks what {org} works on, which program fits a problem they describe, or how to "
                    "get involved in a program."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["all", *categories],
                                     "description": "Program category to list, or 'all'."},
                    },
                    "required": ["category"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "find_contact",
                "description": (
                    f"Find the right published {org} contact for a topic. Returns staff email addresses "
                    "published on the website that match the topic, plus the general inbox and the staff page. "
                    "Call this whenever you refer a visitor to a person, and when the published material doesn't "
                    "answer their question. Give visitors only the contacts this tool returns."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string",
                                  "description": "What the visitor wants to discuss, e.g. 'solar for municipal "
                                                 "buildings' or 'sponsoring an event'."},
                    },
                    "required": ["topic"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "escalate_to_staff",
                "description": (
                    f"Send the visitor's request to {org} staff for follow-up. Call this when the visitor needs "
                    "something only staff can do or decide (requesting a presentation, joining a cohort, a "
                    "time-sensitive question) or asks to be contacted. Staff see only the summary you write, so "
                    "leave out names, contact details, and the town's name. Returns a reference number and the "
                    "staff address the visitor should email, quoting the reference, if they want a reply; this "
                    "assistant does not collect contact details."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": topic_schema,
                        "summary": {"type": "string",
                                    "description": "One or two sentences on what the visitor needs from staff."},
                        "urgency": {"type": "string", "enum": ["routine", "time_sensitive"],
                                    "description": "'time_sensitive' only if the visitor mentions a deadline "
                                                   "within about two weeks."},
                    },
                    "required": ["topic", "summary", "urgency"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "log_visitor_need",
                "description": (
                    f"Record what this visitor needed, for {org}'s anonymous needs report. Call exactly once per "
                    "visitor message, just before your final reply. Staff use these records to see which topics "
                    "members ask about and where the website falls short, so pick the closest topic and the "
                    "honest outcome."
                ),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": topic_schema,
                        "need": {"type": "string",
                                 "description": "One sentence on what the visitor wanted, without names, contact "
                                                "details, or the town's name."},
                        "visitor_role": {"type": "string", "enum": VISITOR_ROLES,
                                         "description": "The visitor's role if they said or clearly implied it, "
                                                        "otherwise 'unknown'."},
                        "outcome": {"type": "string", "enum": list(OUTCOMES), "description": outcomes},
                        "sources": {"type": "array", "items": {"type": "string"},
                                    "description": "URLs your reply relies on; empty if none."},
                    },
                    "required": ["topic", "need", "visitor_role", "outcome", "sources"],
                    "additionalProperties": False,
                },
            },
        ]

    # ------------------------------------------------------------ dispatch

    def run(self, name: str, args: dict, state: TurnState):
        """Execute one tool call. Returns (content, is_error); content is a string or a list of blocks."""
        state.tools_used.append(name)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Unknown tool '{name}'.", True
        try:
            return handler(args or {}, state), False
        except ToolError as e:
            return str(e), True

    @staticmethod
    def _arg(args: dict, key: str, allowed=None) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"Missing or empty argument '{key}'.")
        if allowed is not None and value not in allowed:
            raise ToolError(f"'{key}' must be one of: {', '.join(allowed)}.")
        return value.strip()

    # ------------------------------------------------------------ tools

    def _tool_search_knowledge_base(self, args, state):
        query = self._arg(args, "query")
        kind = self._arg(args, "source_type", ["any", "page", "post", "report"])
        hits = self.kb.search(query, k=5, kind=kind, allowed_tiers=self.allowed_tiers)
        if not hits:
            return ("No published material matches this query. Try other keywords; if nothing fits, tell the "
                    "visitor it isn't covered and offer a contact.")
        return [search_result_block(h.chunk) for h in hits]

    def _tool_read_document(self, args, state):
        url = self._arg(args, "url")
        looking_for = self._arg(args, "looking_for")
        doc = self.kb.get_document(url)
        if doc is None:
            raise ToolError("No published document has this URL. Use a URL exactly as returned by "
                            "search_knowledge_base or list_programs.")
        chunks = self.kb.chunks_for(doc["doc_id"], self.allowed_tiers)
        if not chunks:
            raise ToolError("This document isn't available to this visitor.")
        hits = [h.chunk for h in self.kb.search(looking_for, k=4, doc_id=doc["doc_id"],
                                                allowed_tiers=self.allowed_tiers)]
        return [search_result_block(c) for c in (hits or chunks[:3])]

    def _tool_list_programs(self, args, state):
        category = self._arg(args, "category")
        programs = [p for p in self.program_info if category == "all" or p["category"] == category]
        if not programs:
            raise ToolError("No programs in that category.")
        return json.dumps(programs, indent=1, ensure_ascii=False)

    def _tool_find_contact(self, args, state):
        topic = self._arg(args, "topic")
        return json.dumps(self.match_contacts(topic), indent=1, ensure_ascii=False)

    def _tool_escalate_to_staff(self, args, state):
        topic = self._arg(args, "topic", self.topics)
        summary = self._arg(args, "summary")
        urgency = self._arg(args, "urgency", ["routine", "time_sensitive"])
        match = self.match_contacts(f"{topic.replace('_', ' ')} {summary}")
        routed_to = match["matches"][0]["email"] if match["matches"] else self.general_email
        reference = f"REQ-{secrets.token_hex(3).upper()}"
        record = {
            "reference": reference,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topic": topic,
            "urgency": urgency,
            "summary": summary if self.store_raw else scrub(self.redactor.redact(summary).text, self.sensitive),
            "routed_to": routed_to,
        }
        append_jsonl(self.escalations_path, record)
        state.escalations.append(record)
        return json.dumps({
            "reference": reference,
            "routed_to": routed_to,
            "tell_visitor": f"Staff will see this request (reference {reference}). To get a reply, email "
                            f"{routed_to} and mention {reference}; this assistant doesn't collect contact details.",
        }, indent=1)

    def _tool_log_visitor_need(self, args, state):
        if state.need is not None:
            return "Already recorded for this message."
        state.need = {
            "topic": self._arg(args, "topic", self.topics),
            "need": self._arg(args, "need"),
            "visitor_role": self._arg(args, "visitor_role", VISITOR_ROLES),
            "outcome": self._arg(args, "outcome", list(OUTCOMES)),
            "sources": [s for s in args.get("sources", []) if isinstance(s, str)][:10],
        }
        return "Recorded."

    # ------------------------------------------------------------ helpers

    def _describe_program(self, program: dict) -> dict:
        doc = self.kb.get_document(program["url"])
        text = doc["text"] if doc else ""
        stage = STAGE_RE.search(text)
        summary = next((INLINE_LINK_RE.sub("", ln).strip() for ln in text.splitlines()
                        if len(ln) > 80 and not STAGE_RE.search(ln)), "")
        return {
            "name": program["name"],
            "category": program["category"],
            "stage": stage.group(1).strip().title() if stage else None,
            "summary": summary[:300],
            "url": program["url"],
            "contacts": program.get("contacts") or ([self.general_email] if self.general_email else []),
        }

    def match_contacts(self, topic: str) -> dict:
        """Published contacts ranked by word overlap between the topic and what each contact handles."""
        query = set(tokenize(topic)) - GENERIC_REQUEST_WORDS - self.org_words
        scored = []
        for contact in self.contacts:
            # The general inbox competes too: the site sends some requests (council presentations, sponsorship) there.
            handles = [p for p in self.programs if contact["email"] in p.get("contacts", [])]
            # A program the contact handles counts most, then their job title, then the text around their address.
            score = 3 * sum(len(query & set(tokenize(f"{p['name']} {p['id'].replace('_', ' ')}"))) for p in handles)
            score += 2 * len(query & set(tokenize(contact.get("title") or "")))
            score += len(query & set(tokenize(" ".join(contact.get("contexts", [])))))
            if score:
                scored.append((score, contact, handles))
        scored.sort(key=lambda s: s[0], reverse=True)
        matches = [{
            "name": c.get("name"),
            "role": c.get("title") or None,
            "email": c["email"],
            "handles": [p["name"] for p in handles],
            "note": "General inbox." if c["email"] == self.general_email else
                    None if c.get("listed_on_staff_page") else
                    "Published on a program page but not on the staff page.",
        } for _, c, handles in scored[:3]]
        return {
            "matches": matches,
            "general_inbox": self.general_email,
            "staff_page": self.contact_page,
            "note": None if matches else "No specific staff contact matches this topic; use the general inbox.",
        }


_append_lock = threading.Lock()  # the evaluation runs several sessions in parallel threads


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _append_lock, open(path, "a", encoding="utf-8") as f:
        f.write(line)
