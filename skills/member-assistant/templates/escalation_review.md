# Escalation review

Escalations are in `private/logs/escalations.jsonl`, one request per line with a reference number (`REQ-...`), topic, urgency, an anonymized summary, and the staff address the visitor was told to email.

For each open request:

- [ ] **Time-sensitive first.** Requests marked `time_sensitive` mention a deadline within about two weeks.
- [ ] **Match it to an email.** The visitor was asked to email the listed address and quote the reference. Search that inbox for the reference number. The log never holds contact details, so if no email arrives there is nobody to reply to; close the request as "no follow-up".
- [ ] **Check the routing.** If the request went to the wrong person, forward it and note the topic: repeated misrouting means a program page is missing its contact, which belongs in a website change request.
- [ ] **Record the outcome** in your own tracker (handled, forwarded, no follow-up). Don't add names or contact details back into the log file.
- [ ] **Feed it back.** If the assistant escalated something the website should have answered, add it to the next needs report review.
