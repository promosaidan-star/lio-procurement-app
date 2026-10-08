"""Extraction pipeline tests that run without an OpenAI key.

* The PDF parser is exercised on the five real customer quotes.
* The post-model logic (taxonomy compaction, name resolution, tax-id
  normalisation, arithmetic warnings) is tested with the model call replaced
  by a canned answer, through the real HTTP endpoint.
"""
import json
from pathlib import Path

import pytest

from app.services import extraction as svc
from app.services.extraction import (
    LlmExtraction,
    LlmOrderLine,
    base_name,
    build_taxonomy,
    check_arithmetic,
    normalize_tax_id,
    resolve_commodity_group,
)
from app.services.pdf import NoTextLayerError, extract_text

# Reuse the seeded in-memory app from the smoke test module.
from tests.test_smoke import _auth, client

FIXTURES = Path(__file__).parent / "fixtures" / "quotes"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
QUOTES = [k for k in EXPECTED if not k.startswith("_")]


# ---------------------------------------------------------------------------
# PDF parsing on the customer's real quotes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", QUOTES)
def test_quote_text_contains_vendor_and_total(filename):
    text = extract_text((FIXTURES / filename).read_bytes())
    want = EXPECTED[filename]
    assert want["vendorName"].lower() in text.lower()
    assert f"{want['totalCost']:,.2f}" in text


def test_hollis_quote_keeps_table_rows_together():
    """pdfplumber keeps a line item's description and price on one line and
    puts the EIN label next to its value; pypdf split both."""
    text = extract_text((FIXTURES / "QuoteA0492_23.Pdf").read_bytes())
    assert "EIN: 94 1985704" in text
    assert '1.1 1.00 00009 Moss panel "70:30" with "askLio" $1,560.00 $1,560.00' in text


def test_apple_quote_ligatures_and_order():
    text = extract_text(
        (FIXTURES / "Quote_1__Lio_Technologies_Inc__1x_MBA___2214703918.pdf").read_bytes()
    )
    assert "Configuration" in text  # was "Conﬁguration" (fi ligature)
    assert text.index("Prepared for") < text.index("MacBook Air")


def test_blank_pdf_reports_no_text_layer():
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(NoTextLayerError):
        extract_text(buf.getvalue())


def test_parse_endpoint_accepts_odd_mime_and_rejects_scans():
    r = client.post("/auth/signup", json={"email": "mime@example.com", "password": "secret123"})
    token = r.json()["access_token"]

    pdf_bytes = (FIXTURES / "QuoteA0492_23.Pdf").read_bytes()
    # Uppercase extension + generic MIME type, as some mail clients send it.
    r = client.post(
        "/pdf/parse",
        files={"file": ("QuoteA0492_23.Pdf", pdf_bytes, "application/octet-stream")},
        headers=_auth(token),
    )
    assert r.status_code == 200, r.text
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
    assert "no selectable text" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Taxonomy compaction and resolution
# ---------------------------------------------------------------------------
class _Row:
    def __init__(self, id, category, name):
        self.id, self.category, self.name = id, category, name


def test_base_name_strips_variant_suffix():
    assert base_name("Toner Cartridges — Bulk") == "Toner Cartridges"
    assert base_name("Training Subscription - Annual") == "Training Subscription"
    assert base_name("Courier, Express, and Postal Services") == "Courier, Express, and Postal Services"


def test_taxonomy_collapses_variants_to_canonical_row():
    rows = [
        _Row(204, "Office & Administration", "Toner Cartridges"),
        _Row(534, "Office & Administration", "Toner Cartridges — Small Pack"),
        _Row(1194, "Office & Administration", "Toner Cartridges — Bulk"),
        _Row(29, "Information Technology", "Hardware"),
    ]
    tax = build_taxonomy(rows)
    assert set(tax.by_name) == {"toner cartridges", "hardware"}
    assert tax.by_name["toner cartridges"].id == 204
    assert "Toner Cartridges" in tax.prompt_block and "Bulk" not in tax.prompt_block

    assert resolve_commodity_group("Toner Cartridges — Bulk", tax) == (204, "Toner Cartridges")
    assert resolve_commodity_group("hardware", tax) == (29, "Hardware")
    assert resolve_commodity_group("Toner Cartridge", tax) == (204, "Toner Cartridges")  # fuzzy
    assert resolve_commodity_group("Spaceships", tax) == (None, None)
    assert resolve_commodity_group(None, tax) == (None, None)


def test_full_catalog_compacts_to_a_fraction_of_the_prompt():
    from app.data.commodity_groups import COMMODITY_GROUPS

    tax = build_taxonomy([_Row(*r) for r in COMMODITY_GROUPS])
    assert len(COMMODITY_GROUPS) == 2000
    assert len(tax.by_name) < 400
    assert len(tax.prompt_block) < 12_000  # was ~108K chars as "id | category | name" rows


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def test_normalize_tax_id():
    assert normalize_tax_id("94 1985704") == "94-1985704"
    assert normalize_tax_id("941985704") == "94-1985704"
    assert normalize_tax_id("85-3252405") == "85-3252405"
    assert normalize_tax_id("DE123456789") == "DE123456789"
    assert normalize_tax_id(None) == ""


def _llm(**overrides) -> LlmExtraction:
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
                total_price=715.26,
            ),
            LlmOrderLine(
                position_description='Logo integration "asklio" horizontal',
                unit_price=622.0,
                amount=1,
                unit="pieces",
                total_price=622.0,
            ),
        ],
        net_subtotal=1337.26,
        shipping=215.0,
        tax=133.88,
        other_fees=None,
        grand_total=1686.14,
        commodity_group_name="Office Equipment",
        classification_reason="branded decor",
    )
    base.update(overrides)
    return LlmExtraction(**base)


def test_arithmetic_passes_on_a_consistent_quote_with_discount():
    assert check_arithmetic(_llm()) == []


def test_arithmetic_flags_included_alternative_and_bad_total():
    ex = _llm(
        order_lines=_llm().order_lines
        + [
            LlmOrderLine(
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
    assert "1767.26" in warnings[0] and "1337.26" in warnings[0]

    ex = _llm(grand_total=1700.00)
    warnings = check_arithmetic(ex)
    assert len(warnings) == 1 and "grand total is 1700.00" in warnings[0]


def test_arithmetic_flags_line_that_does_not_multiply():
    ex = _llm(
        order_lines=[
            LlmOrderLine(
                position_description="Widget", unit_price=10.0, amount=3, unit="pieces", total_price=50.0
            )
        ],
        net_subtotal=50.0,
        shipping=None,
        tax=None,
        grand_total=50.0,
    )
    warnings = check_arithmetic(ex)
    assert len(warnings) == 1 and "3.0 x 10.00 = 30.00" in warnings[0]


# ---------------------------------------------------------------------------
# End-to-end through the endpoint with the model call stubbed
# ---------------------------------------------------------------------------
def test_extraction_endpoint_with_stubbed_model(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(
        svc, "_call_model", lambda text, taxonomy: _llm(tax_id="94 1985704", customer=None)
    )

    r = client.post("/auth/signup", json={"email": "stub@example.com", "password": "secret123"})
    token = r.json()["access_token"]
    r = client.post("/extraction", json={"text": "Quote 4120 ..."}, headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["vendorName"] == "Greenmantle Moss Studio, LLC"
    assert data["vatId"] == "94-1985704"
    assert data["totalCost"] == 1686.14
    assert len(data["orderLines"]) == 2
    assert data["commodityGroupId"] == 15 and data["commodityGroupName"] == "Office Equipment"
    assert body["missingFields"] == ["Department"]
    assert body["warnings"] is None


def test_extraction_endpoint_reports_model_failure_cleanly(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_api_key", "test-key")

    def boom(text, taxonomy):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(svc, "_call_model", boom)
    r = client.post("/auth/signup", json={"email": "boom@example.com", "password": "secret123"})
    token = r.json()["access_token"]
    r = client.post("/extraction", json={"text": "Quote"}, headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["success"] is False
    assert "rate limited" in r.json()["error"]
