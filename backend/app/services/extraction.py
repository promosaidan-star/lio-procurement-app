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
from __future__ import annotations  # forward references in type hints

import difflib  # fuzzy matching of the model's chosen group name to the catalog
import logging  # diagnostics for fuzzy matches, unknown names and failures
import re  # variant-suffix stripping and tax-id normalisation
import uuid  # organization id type
from dataclasses import dataclass  # lightweight container for the compacted taxonomy

from openai import LengthFinishReasonError, OpenAI  # client + the "output was cut off" error
from pydantic import BaseModel  # strict schema the model must fill
from sqlalchemy import select  # query the commodity groups
from sqlalchemy.orm import Session  # DB session type

from app.core.config import settings  # API key and model name from the environment
from app.models import CommodityGroup  # ORM row for the taxonomy
from app.schemas.extraction import (  # wire-format models returned to the frontend
    ExtractedVendorData,
    ExtractionResponse,
    OrderLineData,
)
from app.services.classification import apply_commodity_rules  # per-org "always book X under Y" rules

logger = logging.getLogger(__name__)  # module logger

# Quotes are a few thousand characters; anything far beyond that is a catalog
# or a contract, and the model would lose the totals in the middle of it.
MAX_INPUT_CHARS = 60_000

# Tolerances for the arithmetic checks: per-unit prices are often rounded on
# the quote, so lines get a loose relative tolerance; printed totals should
# agree to the cent.
LINE_TOLERANCE = 0.015  # 1.5 % relative on qty x unit price vs. line total
TOTAL_TOLERANCE = 0.05  # 5 cents absolute on subtotal / grand total checks

REQUEST_TIMEOUT_SECONDS = 60  # a quote should extract in seconds; do not hang the upload
MAX_RETRIES = 2  # the SDK retries rate limits and transient errors this many times

# Matches names like "Toner Cartridges — Bulk" / "Training Subscription - Annual".
_VARIANT_SUFFIX = re.compile(r"\s+[—–-]\s+.*$")


# ---------------------------------------------------------------------------
# Model output schema (strict: every field required, nullable where unknown)
# ---------------------------------------------------------------------------
class LlmOrderLine(BaseModel):
    position_description: str  # item name plus key spec
    unit_price: float  # before discount
    amount: float  # quantity as printed, fractional allowed
    unit: str  # written out: "pieces", "licenses", "sq ft"
    total_price: float  # after any line discount


class LlmExtraction(BaseModel):
    title: str  # 2-5 word purchase summary
    vendor_name: str  # issuer of the quote
    tax_id: str | None  # vendor EIN/VAT if printed
    customer: str | None  # bill-to / prepared-for party if printed
    order_lines: list[LlmOrderLine]  # chargeable items only
    net_subtotal: float | None  # printed items subtotal before tax/shipping
    shipping: float | None  # printed shipping amount in the totals block
    tax: float | None  # printed tax amount
    other_fees: float | None  # recycling fees, surcharges in the totals block
    grand_total: float | None  # amount payable as printed
    commodity_group_name: str | None  # one name copied from the taxonomy list, or null
    classification_reason: str  # one sentence; makes the choice auditable


SYSTEM_PROMPT = (  # who the model is and the one behaviour we care most about: copy, don't guess
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
>>>"""  # the quote goes last, fenced, so instructions are not confused with document text


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------
@dataclass
class Taxonomy:
    prompt_block: str  # "Category: name; name; ..." lines sent to the model
    by_name: dict[str, CommodityGroup]  # lower-cased canonical name -> row
    all_ids: set[int]  # every row id, for validation


def base_name(name: str) -> str:
    """'Toner Cartridges — Bulk' -> 'Toner Cartridges'."""
    return _VARIANT_SUFFIX.sub("", name).strip()  # cut everything from the dash separator on


def build_taxonomy(groups: list[CommodityGroup]) -> Taxonomy:
    """Collapse variant rows onto their canonical (un-suffixed) row.

    The canonical row is the one whose name has no variant suffix; if a base
    name only ever appears with suffixes, the lowest id wins.
    """
    canonical: dict[str, CommodityGroup] = {}  # base name -> the row we will resolve to
    for g in sorted(groups, key=lambda g: g.id):  # ascending id so "lowest id wins" holds
        key = base_name(g.name).lower()  # group variants under one key
        current = canonical.get(key)  # row chosen so far for this base name, if any
        is_plain = base_name(g.name) == g.name  # True when this row has no suffix
        if current is None or (is_plain and base_name(current.name) != current.name):  # first seen, or a plain row replacing a suffixed one
            canonical[key] = g

    by_category: dict[str, list[str]] = {}  # category -> list of canonical names
    for g in canonical.values():
        by_category.setdefault(g.category, []).append(base_name(g.name))  # group names under their category
    lines = [
        f"{category}: " + "; ".join(sorted(names))  # one compact line per category
        for category, names in sorted(by_category.items())  # stable order for prompt caching
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
    if not name or not name.strip():  # model said null / nothing fits
        return None, None
    key = base_name(name).lower()  # tolerate the model returning a variant name
    row = taxonomy.by_name.get(key)  # exact match first
    if row is None:  # try a close spelling ("Toner Cartridge" vs "Toner Cartridges")
        close = difflib.get_close_matches(key, list(taxonomy.by_name), n=1, cutoff=0.85)
        if close:
            row = taxonomy.by_name[close[0]]
            logger.info("Fuzzy-matched commodity group %r -> %r", name, row.name)
    if row is None:  # the model invented a name: leave the field for the user
        logger.warning("Model returned unknown commodity group: %r", name)
        return None, None
    return row.id, row.name  # id for the FK, name for display


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def normalize_tax_id(value: str | None) -> str:
    """'94 1985704' / '941985704' -> '94-1985704'; other formats pass through."""
    if not value:  # None or empty
        return ""
    value = value.strip()
    if re.search(r"[A-Za-z]", value):
        return value  # EU-style VAT ids keep their country prefix untouched
    digits = re.sub(r"\D", "", value)  # keep digits only
    if len(digits) == 9:  # a US EIN: two digits, dash, seven digits
        return f"{digits[:2]}-{digits[2:]}"
    return value  # unknown length: leave as printed


def _close_line(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, LINE_TOLERANCE * max(abs(a), abs(b)))  # within 1.5 % or $1, whichever is larger


def _close_total(a: float, b: float) -> bool:
    return abs(a - b) <= TOTAL_TOLERANCE  # within 5 cents


def check_arithmetic(ex: LlmExtraction) -> list[str]:
    """Return human-readable warnings for numbers that do not add up."""
    warnings: list[str] = []  # collected messages
    for line in ex.order_lines:  # per-line check: qty x unit price vs. line total
        expected = line.unit_price * line.amount
        if line.total_price and expected and not _close_line(expected, line.total_price):  # both known and they disagree
            if line.total_price < expected:
                # Quotes often show a discount column; don't nag about those.
                continue
            warnings.append(
                f'Line "{line.position_description[:40]}": {line.amount} x {line.unit_price:.2f} '
                f"= {expected:.2f} but the line total is {line.total_price:.2f}."
            )

    lines_sum = sum(line.total_price for line in ex.order_lines)  # what the extracted lines add up to
    if ex.net_subtotal is not None and ex.order_lines and not _close_total(lines_sum, ex.net_subtotal):  # a line is missing or an alternative crept in
        warnings.append(
            f"The order lines add up to {lines_sum:.2f} but the quote's subtotal is "
            f"{ex.net_subtotal:.2f}. An item may be missing or an alternative may have been included."
        )

    if ex.grand_total is not None:  # rebuild the grand total from its printed parts
        base = ex.net_subtotal if ex.net_subtotal is not None else lines_sum  # prefer the printed subtotal
        rebuilt = base + (ex.shipping or 0) + (ex.tax or 0) + (ex.other_fees or 0)  # None counts as zero
        if not _close_total(rebuilt, ex.grand_total):
            warnings.append(
                f"Subtotal + shipping + tax + fees = {rebuilt:.2f} but the quote's grand total is "
                f"{ex.grand_total:.2f}. Please check the totals."
            )
    return warnings


def to_vendor_data(ex: LlmExtraction, taxonomy: Taxonomy) -> ExtractedVendorData:
    group_id, group_name = resolve_commodity_group(ex.commodity_group_name, taxonomy)  # name -> catalog row
    lines_sum = sum(line.total_price for line in ex.order_lines)  # fallback total if none was printed
    total = ex.grand_total if ex.grand_total is not None else lines_sum  # total cost = amount payable
    return ExtractedVendorData(
        title=ex.title.strip(),
        vendor_name=ex.vendor_name.strip(),
        vat_id=normalize_tax_id(ex.tax_id),  # "94 1985704" -> "94-1985704"
        department=(ex.customer or "").strip(),  # the form calls the customer field "department"
        order_lines=[
            OrderLineData(
                position_description=line.position_description.strip(),
                unit_price=line.unit_price,
                amount=line.amount,
                unit=line.unit.strip(),
                total_price=line.total_price,
            )
            for line in ex.order_lines  # one wire-format line per model line
        ],
        total_cost=round(total, 2),  # money to the cent
        commodity_group_id=group_id,
        commodity_group_name=group_name,
    )


def missing_fields_for(data: ExtractedVendorData) -> list[str]:
    missing: list[str] = []  # labels match the form so the user knows what to fill in
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
        base_url=settings.openai_base_url,  # None -> OpenAI; set for a compatible provider such as Gemini
        timeout=REQUEST_TIMEOUT_SECONDS,  # per-request ceiling
        max_retries=MAX_RETRIES,  # SDK retries 408/409/429/5xx and connection errors
    )
    completion = client.chat.completions.parse(  # parse(): enforce the Pydantic schema on the output
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(pdf_text, taxonomy.prompt_block)},
        ],
        response_format=LlmExtraction,  # strict JSON schema derived from the model class
        max_completion_tokens=6000,  # enough for dozens of lines; LengthFinishReasonError if exceeded
        temperature=0,  # deterministic copying, not creativity
    )
    parsed = completion.choices[0].message.parsed  # already validated LlmExtraction, or None on refusal
    if parsed is None:
        raise ValueError("The model returned no structured data (possibly a refusal).")
    return parsed


def extract_vendor_data(
    pdf_text: str, db: Session, organization_id: uuid.UUID | None = None
) -> ExtractionResponse:
    """Extract vendor data and classify the commodity group in one AI call.

    ``organization_id`` enables that organization's commodity rules; None
    (scripts, tests) means the AI's classification stands.
    """
    if not pdf_text or not pdf_text.strip():  # nothing to work with
        return ExtractionResponse(success=False, error="No text provided for extraction")

    if not settings.openai_api_key:  # the app runs without a key; only this feature needs one
        return ExtractionResponse(
            success=False,
            error="OPENAI_API_KEY is not configured on the server",
        )

    groups = list(db.scalars(select(CommodityGroup).order_by(CommodityGroup.id)))  # full taxonomy from the DB
    if not groups:  # seed did not run
        return ExtractionResponse(
            success=False,
            error="Failed to load commodity groups for classification",
        )
    taxonomy = build_taxonomy(groups)  # 2,000 rows -> ~380 names

    text = pdf_text.strip()
    if len(text) > MAX_INPUT_CHARS:  # defensive cap, see the constant's comment
        logger.info("Truncating extraction input from %d chars", len(text))
        text = text[:MAX_INPUT_CHARS]

    try:
        extraction = _call_model(text, taxonomy)  # the one network call
    except LengthFinishReasonError:  # output hit max_completion_tokens
        logger.exception("Model output truncated")
        return ExtractionResponse(
            success=False,
            error="The quote is too long to extract in one pass. Please fill in the form manually.",
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope to the UI
        logger.exception("Error extracting vendor data")
        return ExtractionResponse(success=False, error=f"Extraction failed: {exc}")

    data = to_vendor_data(extraction, taxonomy)  # model shape -> wire shape

    note = None  # explanation shown to the user when a rule overrides the AI
    hit = apply_commodity_rules(  # the organization's own booking conventions beat the AI
        db, organization_id, [data.title] + [line.position_description for line in data.order_lines]
    )
    if hit is not None:
        data.commodity_group_id = hit.commodity_group_id
        data.commodity_group_name = hit.commodity_group_name
        note = (
            f"Commodity group set to \"{hit.commodity_group_name}\" by your organization's "
            f"rule for \"{hit.keyword}\"."
        )

    return ExtractionResponse(
        success=True,
        data=data,
        missing_fields=missing_fields_for(data) or None,  # None instead of [] keeps the old envelope
        warnings=check_arithmetic(extraction) or None,  # same
        classification_note=note,
    )
