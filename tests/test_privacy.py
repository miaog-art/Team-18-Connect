from conftest import MUNICIPALITIES

from member_assistant.privacy import Redactor

redactor = Redactor(MUNICIPALITIES, keep_emails={"info@example-council.org"})


def test_redacts_email_phone_and_self_introduced_name():
    out = redactor.redact("Hi, I'm Jane Doe. Reach me at jane.doe@gmail.com or (412) 555-0199.")
    assert "Jane" not in out.text and "jane.doe" not in out.text and "555" not in out.text
    assert out.counts["name"] == 1 and out.counts["email"] == 1 and out.counts["phone"] == 1


def test_keeps_the_organizations_own_inbox():
    out = redactor.redact("Should I write to info@example-council.org?")
    assert "info@example-council.org" in out.text
    assert out.counts.get("email", 0) == 0


def test_municipality_becomes_type_and_size_tag():
    out = redactor.redact("Our borough manager in Maple Grove asked about solar.")
    assert "Maple Grove" not in out.text
    assert "[municipality: small borough]" in out.text
    assert out.municipalities == [{"type": "borough", "size_band": "small (<5k)", "member": True}]


def test_abbreviation_variants_match():
    for spelling in ("Mt. Hope", "Mt Hope", "Mount Hope"):
        assert "[municipality: large township]" in redactor.redact(f"We serve {spelling}.").text


def test_ambiguous_names_need_place_context():
    assert redactor.redact("Plum is my favorite fruit.").text == "Plum is my favorite fruit."
    assert "[municipality" in redactor.redact("I'm on council in Plum.").text
    assert "[municipality" in redactor.redact("Plum Borough wants a stormwater study.").text


def test_type_disambiguates_duplicate_names():
    assert "[municipality: small township]" in redactor.redact("Baldwin Township needs help.").text
    assert "[municipality: medium borough]" in redactor.redact("Baldwin Borough needs help.").text


def test_role_words_are_not_treated_as_names():
    text = "I'm the Mayor and I'm interested in solar."
    assert redactor.redact(text).text == text


def test_sign_off_name_and_street_address():
    out = redactor.redact("Please send it to 123 Main Street. Thanks, Bob")
    assert "Main Street" not in out.text and "Bob" not in out.text


def test_values_and_session_scrub():
    from member_assistant.privacy import scrub
    out = redactor.redact("I'm Jane Doe, email jane@x.org")
    assert "Jane Doe" in out.values and "jane@x.org" in out.values
    later = "Jane from the borough wants a call back; Doe said it is urgent."
    assert scrub(later, out.values) == "[redacted] from the borough wants a call back; [redacted] said it is urgent."


def test_municipality_from_first_message_is_scrubbed_later():
    from member_assistant.privacy import scrub
    first = redactor.redact("I'm on council in Plum.")
    assert scrub("Plum council asked about stormwater.", first.values).startswith("[redacted]")
