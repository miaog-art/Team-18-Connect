"""Offline tests of what the assistant stores, using a fake API client (no network, no API key)."""

import json
from types import SimpleNamespace

from conftest import CONTACTS, MUNICIPALITIES, PROFILE, PROGRAMS

from member_assistant import config
from member_assistant.agent import Assistant


class FakeClient:
    """Stands in for anthropic.Anthropic: every call returns a short text reply with no tool calls."""

    def __init__(self, reply: str = "You can email info@example-council.org."):
        self.calls = 0
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply, citations=None)],
            stop_reason="end_turn", stop_details=None, model="fake-model",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

        def create(**kwargs):
            self.calls += 1
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def make_assistant(tmp_path, monkeypatch, kb, guardrails=True):
    files = {"PROFILE_PATH": PROFILE, "CONTACTS_PATH": CONTACTS, "PROGRAMS_PATH": PROGRAMS,
             "MUNICIPALITIES_PATH": MUNICIPALITIES, "SITE_ISSUES_PATH": []}
    for attr, data in files.items():
        path = tmp_path / f"{attr.lower()}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(config, attr, path)
    client = FakeClient()
    assistant = Assistant(guardrails=guardrails, client=client, log_dir=tmp_path / "logs", kb=kb)
    return assistant, client


def stored_questions(tmp_path) -> list[str]:
    lines = (tmp_path / "logs" / "interactions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines]


def test_name_from_earlier_turn_is_scrubbed_from_later_logged_questions(tmp_path, monkeypatch, kb):
    assistant, _ = make_assistant(tmp_path, monkeypatch, kb)
    assistant.ask("Hi, I'm Jane Doe from Maple Grove. How do we join?")
    assistant.ask("Jane here again: what do dues cost?")  # no "I'm", so the redactor alone misses it
    first, second = stored_questions(tmp_path)
    assert "Jane" not in first and "Maple Grove" not in first
    assert "Jane" not in second
    assert "dues" in second


def test_unguarded_condition_still_stores_raw_text(tmp_path, monkeypatch, kb):
    assistant, _ = make_assistant(tmp_path, monkeypatch, kb, guardrails=False)
    assistant.ask("I'm Jane Doe. How do we join?")
    assert stored_questions(tmp_path) == ["I'm Jane Doe. How do we join?"]


def test_emergency_message_is_answered_without_calling_the_model(tmp_path, monkeypatch, kb):
    assistant, client = make_assistant(tmp_path, monkeypatch, kb)
    reply = assistant.ask("Someone is not breathing in the borough office")
    assert client.calls == 0
    assert "911" in reply.text and reply.need["outcome"] == "safety_redirect"
