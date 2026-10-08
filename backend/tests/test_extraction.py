"""Extraction pipeline tests that run without an OpenAI key.

* The PDF parser is exercised on the five real customer quotes.
* The post-model logic (taxonomy compaction, name resolution, tax-id
  normalisation, arithmetic warnings) is tested with the model call replaced
  by a canned answer, through the real HTTP endpoint.
"""
import json  # read expected.json
from pathlib import Path  # fixture paths

import pytest  # parametrize, raises, monkeypatch

from app.services import extraction as svc  # module handle so monkeypatch can swap _call_model
from app.services.extraction import (  # the pure functions under test
    _schema_for,
    openai_base_url,
    reconcile,
    tax_id_in_text,
    LlmExtraction,
    LlmOrderLine,
    base_name,
    build_taxonomy,
    check_arithmetic,
    normalize_tax_id,
    resolve_commodity_group,
)
from app.services.pdf import NoTextLayerError, extract_text  # parser + its scan error

# Reuse the seeded in-memory app from the smoke test module.
from tests.test_smoke import _auth, client

FIXTURES = Path(__file__).parent / "fixtures" / "quotes"  # the five PDFs
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))  # ground truth
QUOTES = [k for k in EXPECTED if not k.startswith("_")]  # filenames only (skip "_comment")


# ---------------------------------------------------------------------------
# PDF parsing on the customer's real quotes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", QUOTES)  # one test per quote
def test_quote_text_contains_vendor_and_total(filename):
    text = extract_text((FIXTURES / filename).read_bytes())  # parse the real file
    want = EXPECTED[filename]
    assert want["vendorName"].lower() in text.lower()  # vendor survived parsing
    assert f"{want['totalCost']:,.2f}" in text  # grand total survived, with its thousands separator


def test_hollis_quote_keeps_table_rows_together():
    """pdfplumber keeps a line item's description and price on one line and
    puts the EIN label next to its value; pypdf split both."""
    text = extract_text((FIXTURES / "QuoteA0492_23.Pdf").read_bytes())
    assert "EIN: 94 1985704" in text  # label and value on one line
    assert '1.1 1.00 00009 Moss panel "70:30" with "askLio" $1,560.00 $1,560.00' in text  # whole table row intact


def test_apple_quote_ligatures_and_order():
    text = extract_text(
        (FIXTURES / "Quote_1__Lio_Technologies_Inc__1x_MBA___2214703918.pdf").read_bytes()
    )
    assert "Configuration" in text  # was "Conﬁguration" (fi ligature)
    assert text.index("Prepared for") < text.index("MacBook Air")  # header before the line item, i.e. reading order


def test_blank_pdf_reports_no_text_layer():
    import io  # local imports: only this test builds a PDF

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # a page with no text at all
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(NoTextLayerError):  # the parser must say "scan", not return ""
        extract_text(buf.getvalue())


def test_parse_endpoint_accepts_odd_mime_and_rejects_scans():
    r = client.post("/auth/signup", json={"email": "mime@example.com", "password": "secret123"})
    token = r.json()["access_token"]  # any signed-in user may parse

    pdf_bytes = (FIXTURES / "QuoteA0492_23.Pdf").read_bytes()
    # Uppercase extension + generic MIME type, as some mail clients send it.
    r = client.post(
        "/pdf/parse",
        files={"file": ("QuoteA0492_23.Pdf", pdf_bytes, "application/octet-stream")},
        headers=_auth(token),
    )
    assert r.status_code == 200, r.text  # accepted by extension
    assert "Hollis Greenscapes" in r.json()["text"]

    # Scanned (blank) PDF: a clear 422, not an empty success.
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    r = client.post(
        "/pdf/parse",
        files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")},
        headers=_auth(token),
    )
    assert r.status_code == 422
    assert "no selectable text" in r.json()["detail"]  # the message the user will read


# ---------------------------------------------------------------------------
# Taxonomy compaction and resolution
# ---------------------------------------------------------------------------
class _Row:
    """Stand-in for a CommodityGroup ORM row: only id/category/name are read."""

    def __init__(self, id, category, name):
        self.id, self.category, self.name = id, category, name


def test_base_name_strips_variant_suffix():
    assert base_name("Toner Cartridges — Bulk") == "Toner Cartridges"  # em dash
    assert base_name("Training Subscription - Annual") == "Training Subscription"  # hyphen
    assert base_name("Courier, Express, and Postal Services") == "Courier, Express, and Postal Services"  # no separator: unchanged


def test_taxonomy_collapses_variants_to_canonical_row():
    rows = [
        _Row(204, "Office & Administration", "Toner Cartridges"),  # the canonical row
        _Row(534, "Office & Administration", "Toner Cartridges — Small Pack"),  # variant
        _Row(1194, "Office & Administration", "Toner Cartridges — Bulk"),  # variant
        _Row(29, "Information Technology", "Hardware"),
    ]
    tax = build_taxonomy(rows)
    assert set(tax.by_name) == {"toner cartridges", "hardware"}  # two names, not four
    assert tax.by_name["toner cartridges"].id == 204  # resolves to the un-suffixed row
    assert "Toner Cartridges" in tax.prompt_block and "Bulk" not in tax.prompt_block  # variants not sent to the model

    assert resolve_commodity_group("Toner Cartridges — Bulk", tax) == (204, "Toner Cartridges")  # variant name still resolves
    assert resolve_commodity_group("hardware", tax) == (29, "Hardware")  # case-insensitive
    assert resolve_commodity_group("Toner Cartridge", tax) == (204, "Toner Cartridges")  # fuzzy
    assert resolve_commodity_group("Spaceships", tax) == (None, None)  # invented name -> leave empty
    assert resolve_commodity_group(None, tax) == (None, None)  # model said null


def test_full_catalog_compacts_to_a_fraction_of_the_prompt():
    from app.data.commodity_groups import COMMODITY_GROUPS  # the real 2,000-row seed

    tax = build_taxonomy([_Row(*r) for r in COMMODITY_GROUPS])
    assert len(COMMODITY_GROUPS) == 2000
    assert len(tax.by_name) < 400  # ~380 distinct names
    assert len(tax.prompt_block) < 12_000  # was ~108K chars as "id | category | name" rows


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def test_normalize_tax_id():
    assert normalize_tax_id("94 1985704") == "94-1985704"  # space, as printed on the Hollis quote
    assert normalize_tax_id("941985704") == "94-1985704"  # bare digits
    assert normalize_tax_id("85-3252405") == "85-3252405"  # already formatted
    assert normalize_tax_id("DE123456789") == "DE123456789"  # EU VAT id: untouched
    assert normalize_tax_id(None) == ""  # missing -> empty string for the form


def _llm(**overrides) -> LlmExtraction:
    """A consistent model answer for the Greenmantle quote; overrides tweak one field."""
    base = dict(
        title="Moss art panel",
        vendor_name="Greenmantle Moss Studio, LLC",
        tax_id="85-3252405",
        customer="Lio Technologies, Inc.",
        order_lines=[
            LlmOrderLine(
                position_description="Moss Art Panel, Mixed Moss 63 x 31.5 in",
                unit_price=894.08,
                amount=1,
                unit="pieces",
                total_price=715.26,  # -20 % discount on the quote
            ),
            LlmOrderLine(
                position_description='Logo integration "asklio" horizontal',
                unit_price=622.0,
                amount=1,
                unit="pieces",
                total_price=622.0,
            ),
        ],
        net_subtotal=1337.26,  # 715.26 + 622.00
        shipping=215.0,
        tax=133.88,  # 115.34 + 18.54
        other_fees=None,
        grand_total=1686.14,  # 1337.26 + 215 + 133.88
        commodity_group_name="Office Equipment",
        classification_reason="branded decor",
    )
    base.update(overrides)  # apply the test's tweak
    return LlmExtraction(**base)


def test_normalize_tax_id_rejects_placeholders_and_words():
    assert normalize_tax_id("XX-XXXXXXX") == ""  # the mask from a prompt, not an id
    assert normalize_tax_id("null") == ""  # the word, not a value
    assert normalize_tax_id("N/A") == ""


def test_empty_base_url_means_openai(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_base_url", "")  # what docker-compose passes when unset
    assert openai_base_url() == "https://api.openai.com/v1"
    monkeypatch.setattr(svc.settings, "openai_base_url", "  ")
    assert openai_base_url() == "https://api.openai.com/v1"
    monkeypatch.setattr(svc.settings, "openai_base_url", "https://example.test/v1")
    assert openai_base_url() == "https://example.test/v1"


def test_schema_only_allows_taxonomy_names_for_the_group():
    from pydantic import ValidationError

    schema = _schema_for(("Hardware", "Production Machinery"))
    base = _llm().model_dump()
    assert schema.model_validate({**base, "commodity_group_name": "Hardware"}).commodity_group_name == "Hardware"
    assert schema.model_validate({**base, "commodity_group_name": None}).commodity_group_name is None
    with pytest.raises(ValidationError):  # a category label, which the model used to answer with
        schema.model_validate({**base, "commodity_group_name": "Production"})
    assert "enum" in str(schema.model_json_schema())  # the restriction is in the schema the API sees


def test_tax_id_must_be_printed_on_the_quote():
    text = "Hollis Greenscapes EIN: 94 1985704 Seller's Permit: 333-571207"
    assert tax_id_in_text("94-1985704", text)  # printed with a space, returned with a dash
    assert tax_id_in_text("941985704", text)
    assert not tax_id_in_text("12-3456789", text)  # the textbook example the model likes to invent
    assert tax_id_in_text("DE123", text)  # too short to judge: pass through


def test_reconcile_drops_shipping_counted_twice():
    ex = _llm(
        order_lines=[
            LlmOrderLine(position_description="Moss panel", unit_price=1560, amount=1, unit="pieces", total_price=1560),
            LlmOrderLine(position_description="Transport, packing and shipping", unit_price=320, amount=1, unit="pieces", total_price=320),
        ],
        net_subtotal=1880.0,
        shipping=320.0,  # the slip: already a line above
        tax=162.15,
        grand_total=2042.15,
    )
    notes = reconcile(ex)
    assert ex.shipping is None and "already an order line" in notes[0]
    assert check_arithmetic(ex) == []  # totals now balance


def test_reconcile_drops_the_one_line_the_subtotal_disproves():
    ex = _llm(
        order_lines=[
            LlmOrderLine(position_description="Panel", unit_price=894.08, amount=1, unit="pieces", total_price=715.26),
            LlmOrderLine(position_description="Logo horizontal", unit_price=622, amount=1, unit="pieces", total_price=622),
            LlmOrderLine(position_description="Logo vertical (Alt.)", unit_price=430, amount=1, unit="pieces", total_price=430),
        ],
        net_subtotal=1337.26,
        shipping=215.0,
        tax=133.88,
        grand_total=1686.14,
    )
    notes = reconcile(ex)
    assert [l.position_description for l in ex.order_lines] == ["Panel", "Logo horizontal"]
    assert "Logo vertical" in notes[0]


def test_reconcile_leaves_ambiguous_cases_alone():
    ex = _llm(
        order_lines=[  # two lines of 100: removing either would match, so do nothing
            LlmOrderLine(position_description="A", unit_price=100, amount=1, unit="pieces", total_price=100),
            LlmOrderLine(position_description="B", unit_price=100, amount=1, unit="pieces", total_price=100),
            LlmOrderLine(position_description="C", unit_price=50, amount=1, unit="pieces", total_price=50),
        ],
        net_subtotal=150.0,
        shipping=None,
        tax=None,
        grand_total=150.0,
    )
    assert reconcile(ex) == [] and len(ex.order_lines) == 3


def test_arithmetic_passes_on_a_consistent_quote_with_discount():
    assert check_arithmetic(_llm()) == []  # the discount must not raise a warning


def test_arithmetic_flags_included_alternative_and_bad_total():
    ex = _llm(
        order_lines=_llm().order_lines
        + [
            LlmOrderLine(  # the alternative item the model should have excluded
                position_description="Logo integration vertical (Alt.)",
                unit_price=430.0,
                amount=1,
                unit="pieces",
                total_price=430.0,
            )
        ]
    )
    warnings = check_arithmetic(ex)
    # The grand total is rebuilt from the printed subtotal, so it still adds
    # up; the lines-vs-subtotal check is what catches the stray alternative.
    assert len(warnings) == 1
    assert "1767.26" in warnings[0] and "1337.26" in warnings[0]  # both numbers shown to the user

    ex = _llm(grand_total=1700.00)  # a grand total that does not match its parts
    warnings = check_arithmetic(ex)
    assert len(warnings) == 1 and "grand total is 1700.00" in warnings[0]


def test_arithmetic_flags_line_that_does_not_multiply():
    ex = _llm(
        order_lines=[
            LlmOrderLine(
                position_description="Widget", unit_price=10.0, amount=3, unit="pieces", total_price=50.0  # 3 x 10 != 50
            )
        ],
        net_subtotal=50.0,
        shipping=None,
        tax=None,
        grand_total=50.0,
    )
    warnings = check_arithmetic(ex)
    assert len(warnings) == 1 and "3.0 x 10.00 = 30.00" in warnings[0]  # only the line check fires


# ---------------------------------------------------------------------------
# End-to-end through the endpoint with the model call stubbed
# ---------------------------------------------------------------------------
def test_extraction_endpoint_with_stubbed_model(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_api_key", "test-key")  # pass the "key configured" gate
    monkeypatch.setattr(  # replace the network call with a canned answer
        svc, "_call_model", lambda text, taxonomy: _llm(tax_id="94 1985704", customer=None)
    )

    r = client.post("/auth/signup", json={"email": "stub@example.com", "password": "secret123"})
    token = r.json()["access_token"]
    r = client.post("/extraction", json={"text": "Quote 4120 ... EIN: 94 1985704"}, headers=_auth(token))  # the id must be printed or the guard blanks it
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["vendorName"] == "Greenmantle Moss Studio, LLC"
    assert data["vatId"] == "94-1985704"  # normalised on the way out
    assert data["totalCost"] == 1686.14  # grand total, not the lines sum
    assert len(data["orderLines"]) == 2
    assert data["commodityGroupId"] == 15 and data["commodityGroupName"] == "Office Equipment"  # resolved against the seeded taxonomy
    assert body["missingFields"] == ["Department"]  # customer=None -> the user must fill it in
    assert body["warnings"] is None  # numbers add up


def test_extraction_endpoint_reports_model_failure_cleanly(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_api_key", "test-key")

    def boom(text, taxonomy):  # simulate the SDK giving up
        raise RuntimeError("rate limited")

    monkeypatch.setattr(svc, "_call_model", boom)
    r = client.post("/auth/signup", json={"email": "boom@example.com", "password": "secret123"})
    token = r.json()["access_token"]
    r = client.post("/extraction", json={"text": "Quote"}, headers=_auth(token))
    assert r.status_code == 200  # still a 200 envelope, never a 500
    assert r.json()["success"] is False
    assert "rate limited" in r.json()["error"]  # the reason reaches the UI
