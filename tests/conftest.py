"""Synthetic fixtures (no client data) shared by the offline tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from member_assistant.kb import KnowledgeBase, chunk_document  # noqa: E402

SITE = "https://example-council.org"

DOCUMENTS = [
    {
        "doc_id": "page:fire", "kind": "page", "title": "Future of Fire", "url": f"{SITE}/future-of-fire/",
        "text": "Future of Fire\nSTAGE OF ENGAGEMENT: IMPLEMENTING\n"
                "Volunteer fire departments face reduced staffing capacity and declining volunteerism across the "
                "county, and the program helps municipalities build collaborative capacity.\n"
                f"Get in Touch (email: chief@example-council.org)\nFire Report ({SITE}/files/fire-report.pdf)",
    },
    {
        "doc_id": "page:solar", "kind": "page", "title": "Solar", "url": f"{SITE}/solar/",
        "text": "Solar\nSTAGE OF ENGAGEMENT: EXPLORING\n"
                "Free help for member governments to put solar panels on public buildings, from site walkthroughs "
                "to financing.\nGet in Touch (email: energy@example-council.org)",
    },
    {
        "doc_id": "page:about", "kind": "page", "title": "About", "url": f"{SITE}/about/",
        "text": "About\nMembers are municipalities. Municipalities become members by paying modest annual dues "
                "relative to their population.\nEmail us (email: info@example-council.org)",
    },
    {
        "doc_id": "report:fire", "kind": "report", "title": "Fire Report", "url": f"{SITE}/files/fire-report.pdf",
        "mirror_urls": ["https://drive.google.com/file/d/ABC123/view"], "text": "",
        "pages": ["Executive summary of volunteer recruitment.", "", "Recommendation: share training across towns."],
        "found_on": [f"{SITE}/future-of-fire/"],
    },
]

CONTACTS = {
    "contacts": [
        {"email": "chief@example-council.org", "name": "Pat Lee", "title": "Public Safety Director",
         "contexts": ["Get in contact with Public Safety (Future of Fire)"], "listed_on_staff_page": True,
         "source_urls": [f"{SITE}/future-of-fire/"]},
        {"email": "energy@example-council.org", "name": None, "title": "",
         "contexts": ["Get in contact with Infrastructure (Solar)"], "listed_on_staff_page": False,
         "source_urls": [f"{SITE}/solar/"]},
        {"email": "info@example-council.org", "name": None, "title": "", "contexts": [],
         "listed_on_staff_page": False, "source_urls": [f"{SITE}/about/"]},
    ],
    "general_email": "info@example-council.org",
    "contact_page": f"{SITE}/staff/",
}

PROGRAMS = [
    {"id": "future_of_fire", "name": "Future of Fire", "url": f"{SITE}/future-of-fire/",
     "category": "Public Safety", "contacts": ["chief@example-council.org"]},
    {"id": "solar", "name": "Solar", "url": f"{SITE}/solar/", "category": "Infrastructure",
     "contacts": ["energy@example-council.org"]},
]

MUNICIPALITIES = [
    {"name": "Maple Grove", "type": "borough", "population": 3000, "size_band": "small (<5k)", "is_member": True,
     "ambiguous": False},
    {"name": "Mt. Hope", "type": "township", "population": 25000, "size_band": "large (20k+)", "is_member": False,
     "ambiguous": False},
    {"name": "Plum", "type": "borough", "population": 27000, "size_band": "large (20k+)", "is_member": False,
     "ambiguous": True},
    {"name": "Baldwin", "type": "borough", "population": 19000, "size_band": "medium (5k-20k)", "is_member": True,
     "ambiguous": True},
    {"name": "Baldwin", "type": "township", "population": 2000, "size_band": "small (<5k)", "is_member": False,
     "ambiguous": True},
]

PROFILE = {"org_name": "Example Council", "org_description": "a council of local governments",
           "site_url": SITE + "/", "general_email": "info@example-council.org"}


@pytest.fixture
def kb():
    chunks = [c for d in DOCUMENTS for c in chunk_document(d)]
    return KnowledgeBase(DOCUMENTS, chunks, policy={})
