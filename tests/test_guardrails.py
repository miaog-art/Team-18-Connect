from conftest import SITE

from member_assistant.guardrails import LinkVerifier, is_emergency, link_key


def test_emergency_screen_is_narrow():
    assert is_emergency("There's a fire at our borough building right now")
    assert is_emergency("Someone is not breathing in the lobby")
    assert is_emergency("I've been thinking about ending my life")
    assert not is_emergency("Our fire department needs volunteers")
    assert not is_emergency("Is there a fire department consolidation report?")
    assert not is_emergency("Do you offer suicide prevention training for officers?")


def test_emergency_screen_catches_self_harm_phrasing():
    assert is_emergency("I keep thinking I might hurt myself")
    assert is_emergency("I've been self-harming again")
    assert is_emergency("Honestly there's no reason to live anymore")
    assert is_emergency("Everyone would be better off dead without me")
    assert not is_emergency("Do you have self-harm prevention resources for our schools?")
    assert not is_emergency("Our council wants to talk about self harm awareness month")


def test_link_key_normalizes_drive_and_slashes():
    assert link_key("https://drive.google.com/file/d/ABC123/view?usp=sharing") == "drive:ABC123"
    assert link_key("https://www.Example.org/solar/") == link_key("https://example.org/solar")


def test_verifier_removes_invented_links_and_emails(kb):
    verifier = LinkVerifier(kb, site_issues=[], extra_emails={"info@example-council.org"})
    reply = (f"See [the Solar page]({SITE}/solar/) or https://example-council.org/grants-portal "
             "and email director@example-council.org.")
    problems = verifier.problems(reply)
    assert {p["value"] for p in problems} == {"https://example-council.org/grants-portal",
                                              "director@example-council.org"}
    clean, removed = verifier.clean(reply, fallback_contact="info@example-council.org")
    assert "grants-portal" not in clean and "director@" not in clean
    assert f"{SITE}/solar/" in clean and len(removed) == 2


def test_verifier_flags_known_broken_links(kb):
    verifier = LinkVerifier(kb, site_issues=[{"type": "broken_link", "url": f"{SITE}/solar/"}])
    assert verifier.problems(f"Go to {SITE}/solar/")[0]["reason"] == "broken"
