"""The agent loop: one visitor message -> tool calls -> grounded, cited reply.

A manual tool-use loop over the Messages API. It uses the beta namespace only for
the server-side refusal fallback. The conversation history is append-only (each
response is appended exactly as returned), which keeps the prompt cache warm and
the model's thinking blocks valid across turns.
"""

import secrets
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from . import config
from .guardrails import (URL_RE, LinkVerifier, baseline_system_prompt, emergency_reply,
                         guarded_system_prompt, is_emergency, link_key)
from .kb import KnowledgeBase
from .privacy import Redactor, scrub
from .tools import Toolbox, TurnState, append_jsonl

# USD per million tokens (Claude Opus 5.5 list prices; 5-minute cache writes).
PRICES = {"claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20}}

REFUSAL_REPLY = ("I can't help with that request here. For questions about our programs, reports, or membership, "
                 "please ask again or contact {contact}.")
FALLBACK_REPLY = "Sorry, I couldn't finish that answer. Please try rephrasing, or contact {contact}."


@dataclass
class Reply:
    text: str                      # what the visitor sees
    raw_text: str                  # the model's reply before the link check
    citations: list[dict]          # sources the reply relies on
    citation_mode: str             # "api" (citation blocks) or "links" (retrieved sources linked in the reply)
    retrieved: list[dict]          # passages the tools returned this turn: source, title, text
    trace: list[dict]              # tool calls made for this message
    need: dict | None              # what log_visitor_need recorded
    escalations: list[dict]
    guardrail_events: list[str]
    link_problems: list[dict]      # unverified links/emails in raw_text (measured with or without guardrails)
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""
    stop_reason: str = ""
    log_record: dict | None = None


def echo_safe(content: list) -> list:
    """Blocks to append to history. After a mid-output fallback, blocks of the declined attempt are dropped."""
    last = max((i for i, b in enumerate(content) if b.type == "fallback"), default=None)
    if last is None:
        return list(content)
    return [b for i, b in enumerate(content)
            if not (i < last and b.type in ("thinking", "redacted_thinking", "tool_use"))]


def summarize_result(content) -> str:
    if isinstance(content, list):
        return "; ".join(block["title"] for block in content if block.get("type") == "search_result")
    return content[:200]


class Assistant:
    """One visitor session. Call ``ask()`` once per visitor message."""

    def __init__(self, guardrails: bool = True, client: anthropic.Anthropic | None = None,
                 log_dir: Path | None = None, model: str | None = None, effort: str | None = None,
                 kb: KnowledgeBase | None = None):
        self.guardrails = guardrails
        self.model = model or config.MODEL
        self.effort = effort or config.EFFORT
        self.log_dir = Path(log_dir or config.LOG_DIR)
        self.profile = config.load_profile()
        self.kb = kb or KnowledgeBase.load()

        contacts = config.load_json(config.CONTACTS_PATH, {})
        programs = config.load_json(config.PROGRAMS_PATH, [])
        municipalities = config.load_json(config.MUNICIPALITIES_PATH, [])
        site_issues = config.load_json(config.SITE_ISSUES_PATH, [])
        org_emails = {c["email"] for c in contacts.get("contacts", [])}
        if contacts.get("general_email"):
            org_emails.add(contacts["general_email"])

        self.redactor = Redactor(municipalities, keep_emails=org_emails)
        self.toolbox = Toolbox(self.kb, self.profile, contacts, programs, self.redactor,
                               escalations_path=self.log_dir / "escalations.jsonl",
                               store_raw=not guardrails)
        self.verifier = LinkVerifier(self.kb, site_issues,
                                     extra_urls=[self.profile.get("site_url"), contacts.get("contact_page")],
                                     extra_emails=org_emails)
        self.contact = contacts.get("general_email") or contacts.get("contact_page") or "the organization"
        now = datetime.now()
        today = f"{now:%A}, {now:%B} {now.day}, {now.year}"
        prompt = guarded_system_prompt if guardrails else baseline_system_prompt
        self.system = prompt(self.profile, today)
        self.tools = self.toolbox.definitions()
        self.client = client or anthropic.Anthropic()
        self.messages: list[dict] = []
        self.session_id = secrets.token_hex(4)
        self.sensitive = self.toolbox.sensitive  # shared with the escalation tool

    # ------------------------------------------------------------ API call

    def _call(self, **overrides):
        kwargs = {
            "model": self.model,
            "max_tokens": config.MAX_TOKENS,
            "system": self.system,
            "tools": self.tools,
            "messages": self.messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},  # caches system + tools + history up to the last block
        }
        if config.USE_FALLBACKS:
            kwargs.update(betas=[config.FALLBACK_BETA], fallbacks="default")
        kwargs.update(overrides)
        return self.client.beta.messages.create(**kwargs)

    # ------------------------------------------------------------ one turn

    def ask(self, text: str) -> Reply:
        started = time.monotonic()
        start_len = len(self.messages)
        state, trace, retrieved, events, usage = TurnState(), [], [], [], Counter()
        if self.guardrails:
            # Remember personal details the visitor gives so they're removed wherever they reappear in stored text.
            self.sensitive.update(self.redactor.redact(text).values)

        try:
            if self.guardrails and is_emergency(text):
                events.append("emergency_screen")
                final_text = emergency_reply(self.profile)
                self.messages.append({"role": "user", "content": text})
                self.messages.append({"role": "assistant", "content": final_text})
                state.need = {"topic": "other", "need": "Emergency or crisis message", "visitor_role": "unknown",
                              "outcome": "safety_redirect", "sources": []}
                citations, stop_reason, served_by = [], "emergency_screen", "none"
            else:
                self.messages.append({"role": "user", "content": text})
                response = self._loop(state, trace, retrieved, usage, events)
                final_text, citations = self._read_final(response, events)
                stop_reason, served_by = response.stop_reason, response.model
        except Exception:
            # Leave the session as it was before this message so the visitor can simply retry.
            del self.messages[start_len:]
            raise

        citation_mode = "api"
        if not citations:
            citations, citation_mode = self._linked_sources(final_text, retrieved), "links"

        if state.need is None:
            events.append("need_not_logged")
            state.need = {"topic": "other", "need": "(not recorded by the assistant)", "visitor_role": "unknown",
                          "outcome": "answered" if citations else "no_answer_found", "sources": []}

        raw_text = final_text
        link_problems = self.verifier.problems(raw_text)
        if self.guardrails and link_problems:
            final_text, removed = self.verifier.clean(raw_text, fallback_contact=self.contact)
            events.append(f"removed_unverified_links:{len(removed)}")

        reply = Reply(
            text=final_text, raw_text=raw_text, citations=citations, citation_mode=citation_mode,
            retrieved=retrieved, trace=trace, need=state.need, escalations=state.escalations,
            guardrail_events=events, link_problems=link_problems, usage=dict(usage),
            cost_usd=round(self._cost(usage), 5), latency_s=round(time.monotonic() - started, 2),
            model=served_by, stop_reason=stop_reason,
        )
        reply.log_record = self._log(text, reply)
        return reply

    def _loop(self, state: TurnState, trace: list, retrieved: list, usage: Counter, events: list):
        """Call the model, run the tools it asks for, and repeat until it replies without a tool call."""
        for _ in range(config.MAX_AGENT_STEPS):
            response = self._call()
            self._add_usage(usage, response)
            content = echo_safe(response.content)
            self.messages.append({"role": "assistant", "content": content})
            if any(b.type == "fallback" for b in response.content):
                events.append(f"refusal_fallback:{response.model}")
            if response.stop_reason != "tool_use":
                # A reply cut off mid tool call still needs tool results, or the next request is rejected.
                dangling = [b for b in content if b.type == "tool_use"]
                if dangling:
                    self.messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                         "content": "Not run: the response was cut off."} for b in dangling]})
                return response

            results = []
            for block in (b for b in content if b.type == "tool_use"):
                args = block.input if isinstance(block.input, dict) else {}
                output, is_error = self.toolbox.run(block.name, args, state)
                trace.append({"tool": block.name, "input": args, "result": summarize_result(output),
                              "output": output if isinstance(output, str) else None, "is_error": is_error})
                if isinstance(output, list):
                    retrieved.extend({"source": r["source"], "title": r["title"],
                                      "text": "\n".join(t["text"] for t in r["content"])}
                                     for r in output if r.get("type") == "search_result")
                result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            self.messages.append({"role": "user", "content": results})

        # Step cap reached: ask for a final answer without further tool calls.
        events.append("step_cap_reached")
        response = self._call(tool_choice={"type": "none"})
        self._add_usage(usage, response)
        self.messages.append({"role": "assistant", "content": echo_safe(response.content)})
        return response

    def _read_final(self, response, events: list) -> tuple[str, list[dict]]:
        content = echo_safe(response.content)
        text = "".join(b.text for b in content if b.type == "text").strip()
        citations, seen = [], set()
        for block in content:
            if block.type != "text":
                continue
            for c in getattr(block, "citations", None) or []:
                if c.type == "search_result_location" and (c.source, c.title) not in seen:
                    seen.add((c.source, c.title))
                    citations.append({"source": c.source, "title": c.title})

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            events.append(f"refusal:{category}")
            return REFUSAL_REPLY.format(contact=self.contact), []
        if response.stop_reason == "max_tokens":
            events.append("max_tokens")
        if not text:
            return FALLBACK_REPLY.format(contact=self.contact), citations
        return text, citations

    @staticmethod
    def _linked_sources(text: str, retrieved: list[dict]) -> list[dict]:
        """Retrieved sources the reply links to (used when the API returns no citation blocks)."""
        by_key = {}
        for r in retrieved:
            by_key.setdefault(link_key(r["source"]), {"source": r["source"], "title": r["title"].split(" (p")[0]})
        found = {}
        for url in URL_RE.findall(text or ""):
            match = by_key.get(link_key(url))
            if match:
                found.setdefault(match["source"], match)
        return list(found.values())

    # ------------------------------------------------------------ accounting

    @staticmethod
    def _add_usage(total: Counter, response) -> None:
        u = response.usage
        total["calls"] += 1
        total["input_tokens"] += u.input_tokens or 0
        total["output_tokens"] += u.output_tokens or 0
        total["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
        total["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    def _cost(self, usage: Counter) -> float:
        p = PRICES.get(self.model)
        if not p:
            return 0.0
        return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]
                + usage["cache_creation_input_tokens"] * p["cache_write"]
                + usage["cache_read_input_tokens"] * p["cache_read"]) / 1_000_000

    def _log(self, question: str, reply: Reply) -> dict:
        """Append one record to the interaction log. With guardrails on, text is redacted first."""
        redacted = self.redactor.redact(question)
        need = dict(reply.need)
        if self.guardrails:
            need["need"] = scrub(self.redactor.redact(need["need"]).text, self.sensitive)
        sources = list(dict.fromkeys(need.get("sources", []) + [c["source"] for c in reply.citations]))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": self.session_id,
            "guardrails": self.guardrails,
            "question": redacted.text if self.guardrails else question,
            "topic": need["topic"],
            "need": need["need"],
            "visitor_role": need["visitor_role"],
            "outcome": need["outcome"],
            "sources": sources,
            "municipality_profiles": redacted.municipalities,
            "redactions": dict(redacted.counts) if self.guardrails else {},
            "escalation_refs": [e["reference"] for e in reply.escalations],
            "tools_used": [t["tool"] for t in reply.trace],
            "guardrail_events": reply.guardrail_events,
            "model": reply.model,
            "usage": reply.usage,
            "cost_usd": reply.cost_usd,
            "latency_s": reply.latency_s,
        }
        append_jsonl(self.log_dir / "interactions.jsonl", record)
        return record
