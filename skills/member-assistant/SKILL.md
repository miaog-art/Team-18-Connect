---
name: member-assistant
description: Operate and improve the website member assistant. Use when refreshing its knowledge base from the public website, reviewing the monthly needs report, turning recurring member questions into website changes, handling staff escalations, or changing the assistant's prompt, tools, or guardrails.
---

# Member assistant: operating and improving it

The assistant answers website visitors from the organization's published pages and reports, routes them to the right published contact, sends requests that need staff to an escalation queue, and records an anonymous summary of every question. This skill covers the work around it: keeping its knowledge current, reading what visitors need, and acting on that.

All commands run from the repository root with the project virtual environment active. Client data lives in `private/` and must never be committed.

## Monthly routine (about 30 minutes)

1. **Refresh the knowledge base** so answers match the current website:
   `python -m member_assistant.crawl`
   Read the summary line. A sharp drop in pages or reports usually means the site changed structure; check `private/site_issues.json` before going further.
2. **Generate the needs report**:
   `python -m member_assistant.report --since <first day of last month>`
   It groups logged questions by topic, shows where the assistant could not answer, breaks needs down by municipality type and size, and drafts suggested website changes.
3. **Review the report with `templates/needs_report_review.md`.** Treat the drafted suggestions as a starting point: check each against the counts and example needs in the report.
4. **Turn each accepted change into a request** using `templates/website_change_request.md`, and give it to whoever edits the website.
5. **Clear the escalation queue** (`private/logs/escalations.jsonl`) using `templates/escalation_review.md`.

## Deciding what becomes a website change

Propose a change when at least one of these holds:

- A topic has **3 or more questions** in the period, and more than a third of them ended as `no_answer_found` or `referred_to_staff`. The website is missing something people look for.
- The crawler reports a **broken link, a link that opens the wrong file, or a contact address that doesn't match** its label. Fix these regardless of volume.
- Several questions come from the **same kind of municipality** (for example small boroughs) about the same topic. Consider a page or FAQ written for that audience.

Don't propose changes from a single question, and never act on a guess about which municipality asked: the logs deliberately record only type and size.

## Changing the assistant safely

Prompts, tool descriptions, and guardrails interact, so measure every change:

1. **Write the test first.** Add a case to `private/eval/questions.jsonl` that shows the problem (format in `templates/eval_case.json`). Use questions real visitors asked, rewritten without personal details.
2. Run the case alone and confirm it fails: `python -m member_assistant.evaluate --cases <ID> --conditions guarded`
3. Make the change, then run the full evaluation: `python -m member_assistant.evaluate --judge`
4. Keep the change only if the new case passes and no other case got worse. With about 30 cases, one case moves a rate by several points, so look at which cases changed, not only the totals.

Each full run calls the model about 70 times (roughly $3-4). Run single cases while iterating.

## Privacy rules (non-negotiable)

- The assistant must not ask visitors for personal details, and stored text is redacted before it is written. Don't turn redaction off outside an evaluation run.
- Municipality names become a type and size tag in the logs (for example `[municipality: small borough]`). Don't add fields that would identify a town.
- Escalations carry a summary and a reference number, not contact details. Visitors who want a reply email staff themselves and quote the reference.
- `private/` stays out of version control. Share the needs report, not the raw logs.

## Files

| Path | What it is |
| --- | --- |
| `templates/needs_report_review.md` | Checklist for reading the monthly needs report |
| `templates/website_change_request.md` | One website change, with the evidence behind it |
| `templates/escalation_review.md` | Checklist for working through staff escalations |
| `templates/eval_case.json` | Format of an evaluation case, with every field explained |
