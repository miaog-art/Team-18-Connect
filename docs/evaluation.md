# Evaluation results

Run on October 22, 2026: 28 cases, each run once with the guardrails (`guarded`) and once with a one-line system prompt and the same tools (`unguarded`). The cases are described in the README. Grounding was judged by Claude Sonnet 5.5.

## Headline numbers (third run, current code)

| Metric | guarded | unguarded |
|---|---|---|
| Cases passing every check | 28/28 (100%) | 20/28 (71%) |
| Coverage: answerable + not-published cases passing (scope target 80%) | 20/20 (100%) | 13/20 (65%) |
| Answerable | 16/16 | 9/16 |
| Not published | 4/4 | 4/4 |
| Escalation | 1/1 | 1/1 |
| Safety | 7/7 | 6/7 |
| Answerable replies that link a source | 14/16 (88%) | 9/16 (56%) |
| Replies showing an unverified link or email | 0 | 0 |
| Personal details kept out of stored logs (D03) | pass | fail |
| LLM judge: every claim supported by the tool results | 15/20 (75%) | 9/20 (45%) |
| Average cost per case | $0.039 | $0.044 |
| Average latency per case | 37 s | 34 s |

With 28 cases, one case moves a rate by about 4 points, so differences of one or two cases are noise.

## What the failures show

- **Without guardrails the assistant gives up too early.** Most unguarded failures say "I couldn't find anything on that" when the answer was in the knowledge base (meeting minutes, internships, research on councils of governments, EMS funding). The guarded prompt tells the model to search first and to use `read_document` for details, and those cases pass.
- **Without guardrails personal details reach the logs.** In D03 a visitor gives their name and town; the unguarded run stored both. The guarded run kept the redactor's output only.
- **The guarded failures the judge found are small additions.** In the five guarded replies the judge marked, the model added a detail the source doesn't state: who a program is for, that a report is "older", example topics for a presentation, or which researchers produced a report. None invented a contact, link, date, or amount.

## Changes between runs

| Run | Guarded | What we changed afterwards |
|---|---|---|
| 1 | 28/28 checks, judge 9/20 | The judge only saw retrieved passages, so it marked real contacts and escalation receipts as invented; it now also sees the contact, program, and escalation data. Council-presentation requests were routed to a program director instead of the general inbox because the organization's own name matched every contact; contact matching now ignores that name and lets the general inbox compete (new test). |
| 2 | 26/28, judge 16/20 | We had also added a prompt rule against adding details the source doesn't state. It made the model answer "I couldn't find it" for questions the knowledge base does answer (4 of 4 retries failed, 4 of 4 passed without it), so we removed it. The judge now also gets today's date. |
| 3 | 28/28, judge 15/20 | Current code. |

## Limits

- The cases were written by the team from the public website and the kickoff meeting. They are not real member questions.
- Each condition ran once per case. Model replies vary between runs (run 2 vs. run 3 above), so small differences need repeated runs.
- The LLM judge has not yet been checked against hand grading.
- These runs went through an API gateway that adds its own hidden instructions to every request and returns no citation blocks. That affects both conditions, inflates input-token costs, and blurs the guarded/unguarded comparison. We will re-run with an official API key before TP3.
