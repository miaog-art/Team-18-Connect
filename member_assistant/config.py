"""Paths and settings shared by every module.

Client-specific material (crawled pages, reports, contacts, logs, the
evaluation questions) lives under ``private/``, which is gitignored. The
client's identity stays confidential until they approve sharing it, so the
committed code never names the organization.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything client-specific goes here; override with MEMBER_ASSISTANT_PRIVATE_DIR.
PRIVATE_DIR = Path(os.environ.get("MEMBER_ASSISTANT_PRIVATE_DIR", REPO_ROOT / "private"))
KB_DIR = PRIVATE_DIR / "kb"
LOG_DIR = PRIVATE_DIR / "logs"
EVAL_DIR = PRIVATE_DIR / "eval"
REPORT_DIR = PRIVATE_DIR / "reports"

DOCUMENTS_PATH = KB_DIR / "documents.jsonl"
CHUNKS_PATH = KB_DIR / "chunks.jsonl"
PROFILE_PATH = PRIVATE_DIR / "profile.json"
CONTACTS_PATH = PRIVATE_DIR / "contacts.json"
PROGRAMS_PATH = PRIVATE_DIR / "programs.json"
MUNICIPALITIES_PATH = PRIVATE_DIR / "municipalities.json"
SITE_ISSUES_PATH = PRIVATE_DIR / "site_issues.json"
ACCESS_POLICY_PATH = PRIVATE_DIR / "access_policy.json"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"

# Model settings. Claude Opus 5.5 always thinks; effort is the only depth control
# and defaults to "medium" on this model, so we set it explicitly.
MODEL = os.environ.get("MEMBER_ASSISTANT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("MEMBER_ASSISTANT_EFFORT", "medium")
MAX_TOKENS = 16000
MAX_AGENT_STEPS = 8  # hard cap on model calls per visitor message

# Server-side refusal fallback (beta): a declined request is re-run on the model
# Anthropic recommends for that refusal category. Set MEMBER_ASSISTANT_FALLBACKS=0
# if your API endpoint rejects the parameter.
USE_FALLBACKS = os.environ.get("MEMBER_ASSISTANT_FALLBACKS", "1") == "1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def load_json(path: Path, default=None):
    """Read a JSON file, returning ``default`` when it does not exist yet."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile() -> dict:
    """Organization profile written by the crawler (name, site URL, general contact)."""
    profile = load_json(PROFILE_PATH)
    if profile is None:
        raise FileNotFoundError(
            f"{PROFILE_PATH} not found. Run `python -m member_assistant.crawl --site <url>` first (see README)."
        )
    return profile
