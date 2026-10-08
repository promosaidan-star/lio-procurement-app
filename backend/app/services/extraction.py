"""Vendor-offer extraction: PDF text -> structured request data + commodity group.

What changed versus the first version and why (see the PR description):

* **Structured outputs.** The model is called through
  ``chat.completions.parse`` with a strict JSON schema, so it cannot answer
  with prose or fenced markdown. Previously any such answer broke
  ``json.loads`` and the upload "errored out completely".
* **Compact taxonomy.** The catalog has 2,000 commodity rows but only ~380
  distinct names; the rest are pack-size / billing-period variants
  ("Toner Cartridges — Bulk"). We send the distinct names grouped by category
  (~6K tokens instead of ~36K) and resolve the chosen name back to the
  canonical row. Faster, cheaper, and the model is no longer choosing between
  six near-identical rows.
* **Quote-shaped rules in the prompt.** Alternatives, carried-forward
  subtotals, discounts, net vs. gross totals, and vendor vs. customer were the
  recurring errors on the customer's five quotes.
* **Arithmetic checks after the call.** Line totals are compared with the net
  subtotal and the grand total. Mismatches are returned as ``warnings`` so the
  requester checks the numbers instead of submitting them unseen.
* **Bounded retries and timeouts** on the OpenAI client instead of a single
  unbounded attempt.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import CommodityGroup
from app.schemas.extraction import (
    ExtractedVendorData,
    ExtractionResponse,
    OrderLineData,
)

logger = logging.getLogger(__name__)

# Quotes are a few thousand characters; anything far beyond that is a catalog
# or a contract, and the model would lose the totals in the middle of it.
MAX_INPUT_CHARS = 60_000

# Tolerances for the arithmetic checks: per-unit prices are often rounded on
# the quote, so lines get a loose relative tolerance; printed totals should
# agree to the cent.
LINE_TOLERANCE = 0.015
TOTAL_TOLERANCE = 0.05

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2

# Matches names like "Toner Cartridges — Bulk" / "Training Subscription - Annual".
_VARIANT_SUFFIX = re.compile(r"\s+[—–-]\s+.*$")


# ---------------------------------------------------------------------------
# Model output schema (strict: every field required, nullable where unknown)
# ---------------------------------------------------------------------------
class LlmOrderLine(BaseModel):
    position_description: str
    unit_price: float
    amount: float
    unit: str
    total_price: float


class LlmExtraction(BaseModel):
    title: str
    vendor_name: str
    tax_id: str | None
    customer: str | None
    order_lines: list[LlmOrderLine]
    net_subtotal: float | None
    shipping: float | None
    tax: float | None
    other_fees: float | None
    grand_total: float | None
    commodity_group_name: str | None
    classification_reason: str


SYSTEM_PROMPT = (
    "You are a procurement analyst. You read vendor quotes and offers "
    "(US business documents) and fill in a structured purchase request exactly "
    "as a careful buyer would. Be literal: copy numbers from the document, never "
    "estimate them."
)


def _build_prompt(pdf_text: str, taxonomy: str) -> str:
    return f"""Extract the purchase request from the quote below and classify it.

RULES FOR THE FIELDS
- vendor_name: the company ISSUING the quote (letterhead, "Vendor", signature, email domain). Never the
  recipient. Use the trading name as printed; drop legal suffixes like Inc./LLC only if the document does.
- customer: the company RECEIVING the quote ("Prepared for", "Bill to", "Quote recipient", the addressee).
  Null if the document does not name one.
- tax_id: the VENDOR's EIN / Federal Tax ID / TIN / VAT ID. Nine digits, formatted XX-XXXXXXX. Null if the
  vendor's tax id is not printed. Do not use the customer's number, a seller's permit, LLC number, or a
  bank routing/account number.
- order_lines: only items the customer would be CHARGED for if they accept the quote as offered:
    * EXCLUDE alternatives ("Alt.", "Alternative to the preceding item", "optional", "instead of").
    * EXCLUDE page subtotals, "carried forward", "page subtotal", and summary rows (subtotal, tax, grand total).
    * EXCLUDE upsell blurbs ("add X to your order for $79").
    * INCLUDE shipping/delivery/installation ONLY when it is a numbered line item with its own price.
    * total_price is the line total AFTER any discount shown on that line; unit_price is before discount.
    * amount is the quantity as printed, fractional if fractional (e.g. 13.78 sq ft). unit written out
      ("ea." -> "pieces", "lic." -> "licenses", "lump sum" -> "lump sum").
    * position_description: the item name plus the key spec on the same line, not the full paragraph.
- net_subtotal: the sum of the items before tax and shipping, as printed ("Items subtotal (net)",
  "Subtotal", "Net total"). Null if not printed.
- shipping, tax, other_fees: as printed in the totals block (null if absent). other_fees covers recycling
  fees, surcharges, etc. that appear in the totals block but are not numbered items.
- grand_total: the total amount payable as printed ("Grand total", "Total (USD)", "Total (gross)").
- title: 2-5 words naming what is being bought (e.g. "Moss art panel with logo", "Industrial grinding machine").
- Numbers: plain decimals, no currency symbols, US formatting ("1,438.00" = 1438.00).

CLASSIFICATION
Choose the SINGLE best commodity group for the PRIMARY item being bought from the list below and copy its
name exactly. Base it on what the items ARE, not on who sells them. If nothing fits, use null and explain
in classification_reason. Guidance: branded decor and furniture -> Office Equipment; brochures, giveaways,
banners -> Promotional Materials; computers, laptops, monitors -> Hardware; licenses -> Software;
machines for manufacturing -> Production Machinery.

COMMODITY GROUPS (category: names)
{taxonomy}

QUOTE TEXT
<<<
{pdf_text}
>>>"""


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------
@dataclass
class Taxonomy:
    prompt_block: str
    by_name: dict[str, CommodityGroup]  # lower-cased canonical name -> row
    all_ids: set[int]


def base_name(name: str) -> str:
    """'Toner Cartridges — Bulk' -> 'Toner Cartridges'."""
    return _VARIANT_SUFFIX.sub("", name).strip()


def build_taxonomy(groups: list[CommodityGroup]) -> Taxonomy:
    """Collapse variant rows onto their canonical (un-suffixed) row.

    The canonical row is the one whose name has no variant suffix; if a base
    name only ever appears with suffixes, the lowest id wins.
    """
    canonical: dict[str, CommodityGroup] = {}
    for g in sorted(groups, key=lambda g: g.id):
        key = base_name(g.name).lower()
        current = canonical.get(key)
        is_plain = base_name(g.name) == g.name
        if current is None or (is_plain and base_name(current.name) != current.name):
            canonical[key] = g

    by_category: dict[str, list[str]] = {}
    for g in canonical.values():
        by_category.setdefault(g.category, []).append(base_name(g.name))
    lines = [
        f"{category}: " + "; ".join(sorted(names))
        for category, names in sorted(by_category.items())
    ]
    return Taxonomy(
        prompt_block="\n".join(lines),
        by_name=canonical,
        all_ids={g.id for g in groups},
    )


def resolve_commodity_group(
    name: str | None, taxonomy: Taxonomy
) -> tuple[int | None, str | None]:
    """Map the model's chosen name back to a catalog row (exact, then fuzzy)."""
    if not name or not name.strip():
        return None, None
    key = base_name(name).lower()
    row = taxonomy.by_name.get(key)
    if row is None:
        close = difflib.get_close_matches(key, list(taxonomy.by_name), n=1, cutoff=0.85)
        if close:
            row = taxonomy.by_name[close[0]]
            logger.info("Fuzzy-matched commodity group %r -> %r", name, row.name)
    if row is None:
        logger.warning("Model returned unknown commodity group: %r", name)
        return None, None
    return row.id, row.name


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def normalize_tax_id(value: str | None) -> str:
    """'94 1985704' / '941985704' -> '94-1985704'; other formats pass through."""
    if not value:
        return ""
    value = value.strip()
    if re.search(r"[A-Za-z]", value):
        return value  # EU-style VAT ids keep their country prefix untouched
    digits = re.sub(r"\D", "", value)
    if len(digits) == 9:
        return f"{digits[:2]}-{digits[2:]}"
    return value


def _close_line(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, LINE_TOLERANCE * max(abs(a), abs(b)))


def _close_total(a: float, b: float) -> bool:
    return abs(a - b) <= TOTAL_TOLERANCE


def check_arithmetic(ex: LlmExtraction) -> list[str]:
    """Return human-readable warnings for numbers that do not add up."""
    warnings: list[str] = []
    for line in ex.order_lines:
        expected = line.unit_price * line.amount
        if line.total_price and expected and not _close_line(expected, line.total_price):
            if line.total_price < expected:
                # Quotes often show a discount column; don't nag about those.
                continue
            warnings.append(
                f'Line "{line.position_description[:40]}": {line.amount} x {line.unit_price:.2f} '
                f"= {expected:.2f} but the line total is {line.total_price:.2f}."
            )

    lines_sum = sum(line.total_price for line in ex.order_lines)
    if ex.net_subtotal is not None and ex.order_lines and not _close_total(lines_sum, ex.net_subtotal):
        warnings.append(
            f"The order lines add up to {lines_sum:.2f} but the quote's subtotal is "
            f"{ex.net_subtotal:.2f}. An item may be missing or an alternative may have been included."
        )

    if ex.grand_total is not None:
        base = ex.net_subtotal if ex.net_subtotal is not None else lines_sum
        rebuilt = base + (ex.shipping or 0) + (ex.tax or 0) + (ex.other_fees or 0)
        if not _close_total(rebuilt, ex.grand_total):
            warnings.append(
                f"Subtotal + shipping + tax + fees = {rebuilt:.2f} but the quote's grand total is "
                f"{ex.grand_total:.2f}. Please check the totals."
            )
    return warnings


def to_vendor_data(ex: LlmExtraction, taxonomy: Taxonomy) -> ExtractedVendorData:
    group_id, group_name = resolve_commodity_group(ex.commodity_group_name, taxonomy)
    lines_sum = sum(line.total_price for line in ex.order_lines)
    total = ex.grand_total if ex.grand_total is not None else lines_sum
    return ExtractedVendorData(
        title=ex.title.strip(),
        vendor_name=ex.vendor_name.strip(),
        vat_id=normalize_tax_id(ex.tax_id),
        department=(ex.customer or "").strip(),
        order_lines=[
            OrderLineData(
                position_description=line.position_description.strip(),
                unit_price=line.unit_price,
                amount=line.amount,
                unit=line.unit.strip(),
                total_price=line.total_price,
            )
            for line in ex.order_lines
        ],
        total_cost=round(total, 2),
        commodity_group_id=group_id,
        commodity_group_name=group_name,
    )


def missing_fields_for(data: ExtractedVendorData) -> list[str]:
    missing: list[str] = []
    if not data.title:
        missing.append("Title/Short Description")
    if not data.vendor_name:
        missing.append("Vendor Name")
    if not data.vat_id:
        missing.append("Tax ID")
    if not data.department:
        missing.append("Department")
    if not data.order_lines:
        missing.append("Order Lines")
    if not data.total_cost:
        missing.append("Total Cost")
    if not data.commodity_group_id or not data.commodity_group_name:
        missing.append("Commodity Group")
    return missing


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def _call_model(pdf_text: str, taxonomy: Taxonomy) -> LlmExtraction:
    """One structured-output call. Separated so tests can replace it."""
    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,  # SDK retries 408/409/429/5xx and connection errors
    )
    completion = client.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(pdf_text, taxonomy.prompt_block)},
        ],
        response_format=LlmExtraction,
        max_completion_tokens=6000,
        temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("The model returned no structured data (possibly a refusal).")
    return parsed


def extract_vendor_data(pdf_text: str, db: Session) -> ExtractionResponse:
    """Extract vendor data and classify the commodity group in one AI call."""
    if not pdf_text or not pdf_text.strip():
        return ExtractionResponse(success=False, error="No text provided for extraction")

    if not settings.openai_api_key:
        return ExtractionResponse(
            success=False,
            error="OPENAI_API_KEY is not configured on the server",
        )

    groups = list(db.scalars(select(CommodityGroup).order_by(CommodityGroup.id)))
    if not groups:
        return ExtractionResponse(
            success=False,
            error="Failed to load commodity groups for classification",
        )
    taxonomy = build_taxonomy(groups)

    text = pdf_text.strip()
    if len(text) > MAX_INPUT_CHARS:
        logger.info("Truncating extraction input from %d chars", len(text))
        text = text[:MAX_INPUT_CHARS]

    try:
        extraction = _call_model(text, taxonomy)
    except LengthFinishReasonError:
        logger.exception("Model output truncated")
        return ExtractionResponse(
            success=False,
            error="The quote is too long to extract in one pass. Please fill in the form manually.",
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope to the UI
        logger.exception("Error extracting vendor data")
        return ExtractionResponse(success=False, error=f"Extraction failed: {exc}")

    data = to_vendor_data(extraction, taxonomy)
    return ExtractionResponse(
        success=True,
        data=data,
        missing_fields=missing_fields_for(data) or None,
        warnings=check_arithmetic(extraction) or None,
    )
