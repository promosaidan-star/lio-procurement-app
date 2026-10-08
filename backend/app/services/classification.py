"""Per-organization commodity rules (task 3).

Acme books cable ties under Electrical Engineering, printer toner under
Hardware and small software subscriptions under Office Supplies, whatever the
AI thinks. Those are accounting conventions, not classification errors, so
they live in the organization's settings as ordered rules:

    {"keyword": "cable ties", "commodity_group_id": 11}

A rule fires when every word of its keyword appears (stemmed, case-insensitive)
in the request title or in any order-line description. The first rule that
fires wins. The same function runs after AI extraction (so the form is
pre-filled correctly) and again when a request is saved (so the rule is
non-negotiable even if someone changes the dropdown).
"""
from __future__ import annotations  # forward references in type hints

import uuid  # organization id type
from dataclasses import dataclass  # small result record

from sqlalchemy.orm import Session  # DB session type

from app.models import CommodityGroup, Organization  # ORM rows read here
from app.services.catalog import tokenize  # same tokeniser as the catalog matcher: lower-case, stopwords, stems


@dataclass
class RuleHit:
    keyword: str  # the rule text that fired, shown to the user
    commodity_group_id: int  # the group the organization insists on
    commodity_group_name: str  # its display name


def load_rules(db: Session, organization_id: uuid.UUID | None) -> list[dict]:
    """The org's ordered rule list from its settings JSON (empty when none)."""
    if organization_id is None:  # callers without an org context (tests, scripts)
        return []
    org = db.get(Organization, organization_id)  # primary-key lookup
    if org is None:
        return []
    return list((org.settings or {}).get("commodity_rules", []))  # settings may be {} on old rows


def match_rules(rules: list[dict], texts: list[str]) -> dict | None:
    """Return the first rule whose keyword words all appear in some text."""
    text_tokens = [tokenize(t) for t in texts if t]  # tokenise each title / line once
    for rule in rules:  # rules are ordered: first match wins
        wanted = tokenize(rule.get("keyword", ""))  # "printer toner" -> {"printer", "toner"}
        if not wanted:  # blank keyword can never fire
            continue
        if any(wanted <= toks for toks in text_tokens):  # subset test: every keyword word present
            return rule
    return None


def apply_commodity_rules(
    db: Session, organization_id: uuid.UUID | None, texts: list[str]
) -> RuleHit | None:
    """Load the org's rules, test them against the texts, resolve the group name."""
    rule = match_rules(load_rules(db, organization_id), texts)  # None when nothing fires
    if rule is None:
        return None
    group = db.get(CommodityGroup, int(rule["commodity_group_id"]))  # the group the rule points at
    if group is None:  # the group was removed after the rule was written: ignore the rule
        return None
    return RuleHit(
        keyword=rule["keyword"],
        commodity_group_id=group.id,
        commodity_group_name=group.name,
    )
