import json

from conftest import CONTACTS, MUNICIPALITIES, PROFILE, PROGRAMS, SITE

from member_assistant.kb import KnowledgeBase, chunk_document, tokenize
from member_assistant.privacy import Redactor
from member_assistant.tools import Toolbox, TurnState


def make_toolbox(kb, tmp_path):
    return Toolbox(kb, PROFILE, CONTACTS, PROGRAMS, Redactor(MUNICIPALITIES, {"info@example-council.org"}),
                   escalations_path=tmp_path / "escalations.jsonl")


def test_tokenize_strips_stopwords_and_plurals():
    assert tokenize("The departments of volunteers") == ["department", "volunteer"]


def test_report_chunks_keep_pdf_page_numbers(kb):
    chunks = chunk_document(kb.get_document(f"{SITE}/files/fire-report.pdf"))
    assert [c["locator"] for c in chunks] == ["pp. 1-3"]


def test_search_ranks_relevant_page_and_filters_by_kind(kb):
    hits = kb.search("volunteer fire staffing")
    assert hits[0].chunk["doc_id"] in ("page:fire", "report:fire")
    assert all(h.chunk["kind"] == "report" for h in kb.search("volunteer", kind="report"))
    assert kb.search("zebra quantum") == []


def test_document_lookup_accepts_mirror_url_and_missing_slash(kb):
    assert kb.get_document("https://drive.google.com/file/d/ABC123/view")["doc_id"] == "report:fire"
    assert kb.get_document(f"{SITE}/solar")["doc_id"] == "page:solar"


def test_member_only_policy_hides_documents(kb):
    from conftest import DOCUMENTS
    chunks = [c for d in DOCUMENTS for c in chunk_document(d)]
    gated = KnowledgeBase(DOCUMENTS, chunks, policy={"member_only_url_prefixes": [f"{SITE}/files/"]})
    assert all(h.chunk["kind"] != "report" for h in gated.search("volunteer recruitment"))
    assert any(h.chunk["kind"] == "report" for h in gated.search("volunteer recruitment",
                                                                 allowed_tiers=("public", "member")))


def test_tool_definitions_are_strict(kb, tmp_path):
    for tool in make_toolbox(kb, tmp_path).definitions():
        schema = tool["input_schema"]
        assert tool["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_search_tool_returns_citable_search_results(kb, tmp_path):
    out, is_error = make_toolbox(kb, tmp_path).run(
        "search_knowledge_base", {"query": "solar panels", "source_type": "any"}, TurnState())
    assert not is_error
    assert out[0]["type"] == "search_result" and out[0]["citations"] == {"enabled": True}
    assert out[0]["source"] == f"{SITE}/solar/"


def test_read_document_rejects_unknown_url(kb, tmp_path):
    out, is_error = make_toolbox(kb, tmp_path).run(
        "read_document", {"url": "https://evil.example/x", "looking_for": "anything"}, TurnState())
    assert is_error and "No published document" in out


def test_find_contact_matches_program_contact(kb, tmp_path):
    out, _ = make_toolbox(kb, tmp_path).run("find_contact", {"topic": "solar for our town hall"}, TurnState())
    data = json.loads(out)
    assert data["matches"][0]["email"] == "energy@example-council.org"
    assert data["general_inbox"] == "info@example-council.org"


def test_escalation_is_redacted_before_storage(kb, tmp_path):
    box = make_toolbox(kb, tmp_path)
    state = TurnState()
    out, is_error = box.run("escalate_to_staff", {
        "topic": "future_of_fire", "urgency": "routine",
        "summary": "Maple Grove borough wants a fire presentation; call me at 412-555-0101."}, state)
    assert not is_error and json.loads(out)["routed_to"] == "chief@example-council.org"
    stored = (tmp_path / "escalations.jsonl").read_text()
    assert "Maple Grove" not in stored and "555-0101" not in stored


def test_log_need_validates_enums_and_records_once(kb, tmp_path):
    box, state = make_toolbox(kb, tmp_path), TurnState()
    bad, is_error = box.run("log_visitor_need", {"topic": "nope", "need": "x", "visitor_role": "unknown",
                                                 "outcome": "answered", "sources": []}, state)
    assert is_error and state.need is None
    args = {"topic": "solar", "need": "Wants solar", "visitor_role": "municipal_staff",
            "outcome": "answered", "sources": [f"{SITE}/solar/"]}
    assert box.run("log_visitor_need", args, state) == ("Recorded.", False)
    assert box.run("log_visitor_need", args, state)[0] == "Already recorded for this message."


def test_program_list_includes_stage_and_contact(kb, tmp_path):
    out, _ = make_toolbox(kb, tmp_path).run("list_programs", {"category": "all"}, TurnState())
    fire = next(p for p in json.loads(out) if p["name"] == "Future of Fire")
    assert fire["stage"] == "Implementing" and fire["contacts"] == ["chief@example-council.org"]


def test_program_contact_outranks_incidental_word_matches(kb, tmp_path):
    from conftest import CONTACTS
    contacts = {**CONTACTS, "contacts": CONTACTS["contacts"] + [{
        "email": "ops@example-council.org", "name": None, "title": "Operations",
        "contexts": ["Join a program: request a meeting with staff about your municipality"],
        "listed_on_staff_page": True, "source_urls": []}]}
    box = Toolbox(kb, PROFILE, contacts, PROGRAMS, Redactor(MUNICIPALITIES, set()), tmp_path / "e.jsonl")
    best = box.match_contacts("future of fire: a borough wants staff to contact them about joining the program")
    assert best["matches"][0]["email"] == "chief@example-council.org"


def test_general_inbox_wins_requests_the_site_sends_there(kb, tmp_path):
    from conftest import CONTACTS
    contacts = {**CONTACTS, "contacts": [
        {**c, "contexts": ["Request a speaker for your board meeting"]} if c["email"] == CONTACTS["general_email"]
        else c for c in CONTACTS["contacts"]]}
    box = Toolbox(kb, PROFILE, contacts, PROGRAMS, Redactor(MUNICIPALITIES, set()), tmp_path / "e.jsonl")
    out, _ = box.run("escalate_to_staff", {
        "topic": "events_and_meetings", "summary": "A borough board wants a speaker at its next meeting.",
        "urgency": "routine"}, TurnState())
    assert json.loads(out)["routed_to"] == "info@example-council.org"
