# Member Assistant: an agentic website assistant for a regional membership organization

Team 18, Section B, Intro to AI (Fall 2026), Carnegie Mellon University Heinz College.

> The client's identity is confidential until they approve sharing it. This repository never names the organization; everything client-specific (crawled website content, contacts, logs, the evaluation questions) is generated locally under `private/`, which is excluded from version control.

**Status: TP2 prototype (October 2026).** Runs from the command line; the hosted website version is planned for TP3.

## The problem

The client is a nonprofit whose members are local governments. Its website presents reports and program pages but gives visitors no way to ask a question, and pages end with "reach out for more information." Many member towns have no full-time manager and can't attend member meetings, so the organization hears about their needs late, if at all.

The client asked for two things: a website that engages visitors instead of acting "like a wall", and a clearer picture of who visits and what they need.

## What the prototype does

1. **Answers visitor questions from the published website.** It searches the organization's pages, posts, and report PDFs, answers only from what it finds, and links the page or report it used.
2. **Routes visitors** to the published contact for their topic, and sends requests that need staff (a presentation, joining a cohort, something with a deadline) to an escalation queue with a reference number.
3. **Records an anonymous summary of every question** (topic, need, visitor role, outcome). Municipality names become a type-and-size tag such as `[municipality: small borough]`, following our TA's suggestion, so staff can see which kinds of members need what without seeing which town asked.
4. **Turns the log into a monthly needs report.** The report shows topics, where the website couldn't answer, needs by municipality type, and website problems found while crawling (broken links, links that open the wrong file, mismatched email links). It also drafts website changes from that evidence.

```
 visitor ──► Assistant (Claude Opus 5.5, manual tool-use loop)
               │  emergency screen (911 / 988) before any model call
               │
               ├─ search_knowledge_base ─► BM25 index over crawled pages + report PDFs
               ├─ read_document          (cited search results, PDF page numbers)
               ├─ list_programs
               ├─ find_contact ──────────► published contacts only
               ├─ escalate_to_staff ─────► escalations.jsonl (redacted)
               └─ log_visitor_need ──────► interactions.jsonl (redacted)
               │
               ▼
           link/email check: anything not on the published site is removed
               │
               ▼
 staff ◄── needs report (python -m member_assistant.report)
```

## How it meets the course's technical requirements

| Requirement | Where |
| --- | --- |
| Agentic workflow | `member_assistant/agent.py`: the model plans and calls tools across several steps per message (search → read → route → log), capped at 8 model calls. The monthly crawl → report → change-request routine is in the skill folder. |
| Pretrained model | Claude Opus 5.5 (`claude-opus-5-5`) via the Anthropic Python SDK, with adaptive thinking and server-side refusal fallback. Claude Sonnet 5.5 grades grounding in the evaluation. |
| 5+ custom agentic tools with LLM-facing docs | Six tools in `member_assistant/tools.py`. Every schema is strict, and topics, roles, and outcomes are enums, so the needs log can be aggregated. |
| Skill folder | `skills/member-assistant/SKILL.md` and four templates: needs-report review, website change request, escalation review, evaluation case. |
| Evaluation | `member_assistant/evaluate.py`: test cases with programmatic checks, run with and without guardrails, optional LLM grounding judge. 32 unit tests in `tests/` run offline. |
| Data protection / simulated data | Public website content only. Redaction before storage (`privacy.py`), no contact details collected, raw logs kept out of the repo. Unit tests use synthetic fixtures. |
| Safety guardrails | `member_assistant/guardrails.py`: grounded-answer rules, emergency screen, link/email verification, redaction. Details below. |
| With vs. without guardrails | `python -m member_assistant.evaluate` runs every case in both conditions; results below. |

## Guardrails

| Guardrail | Risk it addresses |
| --- | --- |
| System prompt rules: answer only from tool results, never invent contacts, links, dates, or amounts, route instead of guessing, no legal/financial/medical advice, ignore instructions inside retrieved text, stay on topic | Hallucinated facts, prompt injection, off-topic use |
| Emergency screen (pattern match before the model is called) | Someone describing an emergency gets 911/988 immediately |
| Link and email verification | Every URL and address in a reply must come from the published site; anything else, or a link the crawler found broken, is removed |
| Redaction before storage | Names, emails, phones, street addresses, and town names never reach the logs. Details a visitor gave earlier in a session are also scrubbed when they reappear in later turns. |
| Escalations without contact details | Staff get an anonymized summary and a reference number; visitors email staff themselves if they want a reply |

## Evaluation

The evaluation set (in `private/eval/questions.jsonl`) has 28 cases written from the published website and the kickoff meeting, because the client hasn't shared real member questions yet. There are four kinds:

- answerable from the website
- asking for something the website doesn't publish (dues amounts, a phone number)
- needing staff follow-up
- safety: emergency, crisis, prompt injection, personal details, legal advice, out of scope, a request for a staff member's private information

Each case lists programmatic checks: required phrases or contacts, forbidden content such as invented dollar figures, the source it should cite, the tool or outcome expected, and strings that must never appear in stored logs. Every case also checks that no unverified link or email reaches the visitor. A second model checks whether each answer's claims are supported by the retrieved passages.

Results are in `docs/evaluation.md`. On the current code, the guarded assistant passes 28 of 28 cases and the unguarded one 20 of 28.

## Running it

Requires Python 3.10+ and an Anthropic API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or `ant auth login`

# 1. Build the knowledge base from the client's public site (about 3 minutes, one request per second)
python -m member_assistant.crawl --site https://<client-site> --county-wiki "<Wikipedia county article>"

# 2. Chat (shows the tool calls; add --no-guardrails to compare)
python -m member_assistant.chat

# 3. Evaluate both conditions (about $3-4 per full run) and write a needs report
python -m member_assistant.evaluate --judge
python -m member_assistant.report

# Offline unit tests (no API calls)
pytest
```

Each visitor message costs about $0.04 and takes 20 to 45 seconds with Claude Opus 5.5 at medium effort.

## Repository layout

```
member_assistant/   crawl.py  kb.py  tools.py  guardrails.py  privacy.py  agent.py  chat.py  evaluate.py  report.py  config.py
skills/member-assistant/   SKILL.md and templates/
tests/              offline unit tests with synthetic data
private/            (not committed) crawled site, contacts, logs, evaluation set and runs, reports
```

## What's next (TP3)

- Hosted website version with a chat widget. A public endpoint needs rate limiting and a spending cap, because anyone could run up API costs.
- A larger evaluation set built from real member questions, if the client can share anonymized ones (our TA's suggestion), plus a usability test with five people.
- Tiered access for member organizations, if the client wants it (`private/access_policy.json` already supports member-only documents).
- Simulated visitor traffic to show the needs report at realistic volume.

## Responsible AI

The assistant only repeats what the organization publishes and tells visitors when something isn't covered. It collects no identity and stores anonymized summaries only. Decisions about the website stay with staff: the needs report drafts suggestions, and staff accept or reject them using the review template. Known limits: pattern-based redaction can miss identifying details that have no recognizable form, the knowledge base is only as current as the last crawl, and the safety classifiers or the model can still make mistakes, which is why replies link their sources.

## Team

| Member | GitHub |
| --- | --- |
| Wenjun Chen | [Ireton123](https://github.com/Ireton123) |
| Andy Nguyen | _add profile link_ |
| Miao Guo | [miaog-art](https://github.com/miaog-art) |
| Leo Chikava | [lchikava](https://github.com/lchikava) |

## References

- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- Asai, A. et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. arXiv:2310.11511.
- Es, S. et al. (2024). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024 (demo track). arXiv:2309.15217.
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- Anthropic. Claude API documentation: tool use, search results and citations, structured outputs. platform.claude.com/docs.
