"""Catalog suggestions (task 2): match order lines to negotiated articles."""
import uuid  # fake ids for the stand-in articles and the "foreign article" case

from sqlalchemy import select  # look up a real seeded article

from app.models import Article, Organization  # ORM rows
from app.services.catalog import CatalogIndex, tokenize  # the matcher under test
from tests.test_smoke import TestingSession, _auth, _signin, client  # shared seeded test app


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
class _Art:
    """Stand-in for an Article row with just the attributes the matcher reads."""

    def __init__(self, number, description, price, supplier="S"):
        self.id = uuid.uuid4()
        self.article_number = number
        self.description = description
        self.unit_price = price
        self.supplier_id = uuid.uuid4()
        self.supplier = None
        self.currency = "USD"
        self.unit = "piece"


def test_tokenize_stems_and_drops_noise():
    assert tokenize("Toner cartridges, black (high-yield) for the printer") == {  # plurals stemmed, "for"/"the" dropped
        "toner", "cartridge", "black", "high", "yield", "printer",
    }
    assert tokenize('13" MacBook Air') == {"13", "macbook", "air"}  # numbers kept, quote mark ignored


def test_best_match_first_then_cheapest():
    idx = CatalogIndex(
        [
            _Art("A1", "Toner cartridge black, high-yield", 186.0),  # same text, dearer
            _Art("A2", "Toner cartridge black, high-yield", 139.0),  # same text, cheaper
            _Art("A3", "Toner cartridge cyan", 95.0),  # shares two words
            _Art("A4", "Black marker set", 4.0),  # shares one word only
            _Art("A5", "Stretch film machine grade", 8.0),  # shares nothing
        ]
    )
    got = idx.suggest("Toner cartridge black", limit=5)
    numbers = [m.article.article_number for m in got]
    assert numbers[:2] == ["A2", "A1"]  # same text, cheaper negotiated price first
    assert "A5" not in numbers  # no shared meaningful term
    assert got[0].matched_terms == ["black", "cartridge", "toner"]  # sorted, stemmed


def test_rare_terms_outweigh_common_ones():
    idx = CatalogIndex(
        [
            _Art("M1", "Framed preserved moss panel with company logo cut-out", 320.0),  # shares "moss" and "panel"
            _Art("P1", "Strapping PET 5/8 in", 32.0),
            _Art("P2", "Pallet wrap 18 in", 20.0),
            _Art("P3", "Tape 2 in", 3.0),
        ]
    )
    got = idx.suggest("Moss Art Panel, Mixed Moss 63 x 31.5 in", limit=3)
    assert got and got[0].article.article_number == "M1"  # the moss panel wins
    assert all(m.article.article_number != "P3" for m in got)  # "in" is a stopword, so no match


def test_no_match_for_unrelated_text():
    idx = CatalogIndex([_Art("A1", "Toner cartridge black", 10.0)])
    assert idx.suggest("Grinding machine HW-GS 450") == []  # no vocabulary overlap
    assert idx.suggest("") == []  # empty query


# ---------------------------------------------------------------------------
# Endpoint, on the seeded Acme catalog
# ---------------------------------------------------------------------------
def test_suggest_endpoint_matches_seeded_catalog_and_is_org_scoped():
    token = _signin("requester@acme.com")  # a requester, the role that creates requests
    lines = [
        {"positionDescription": "Toner cartridge black, high-yield"},  # on catalog from several suppliers
        {"positionDescription": "Moss Art Panel, Mixed Moss 63 x 31.5 in"},  # the Greenmantle quote line
        {"positionDescription": "Grinding machine HW-GS 450"},  # nothing like it on catalog
    ]
    r = client.post("/articles/suggest", json={"lines": lines}, headers=_auth(token))
    assert r.status_code == 200, r.text
    toner, moss, grinder = r.json()["suggestions"]  # one list per input line, same order

    assert len(toner) == 3  # default limit
    assert all("toner" in s["description"].lower() for s in toner)
    assert toner[0]["supplierName"] and toner[0]["articleNumber"]  # supplier is joined in
    # all top matches are the same article text -> ordered by negotiated price
    prices = [s["unitPrice"] for s in toner if s["description"] == toner[0]["description"]]
    assert prices == sorted(prices)

    assert moss and "moss" in moss[0]["description"].lower()
    assert grinder == []  # nothing like it on catalog: no false positive

    # Another tenant has an empty catalog and gets no suggestions.
    other = _signin("admin@walmart.com")
    r = client.post("/articles/suggest", json={"lines": lines}, headers=_auth(other))
    assert r.status_code == 200
    assert r.json()["suggestions"] == [[], [], []]

    # auth required
    assert client.post("/articles/suggest", json={"lines": lines}).status_code == 401


def test_request_records_catalog_article_and_rejects_foreign_ones():
    token = _signin("requester@acme.com")
    with TestingSession() as db:  # pick a real Acme article straight from the DB
        acme = db.scalar(select(Organization).where(Organization.slug == "acme"))
        article = db.scalar(
            select(Article)
            .where(Article.organization_id == acme.id)
            .order_by(Article.article_number)
        )
        article_id, article_number, price = (  # copy the values out before the session closes
            str(article.id),
            article.article_number,
            float(article.unit_price),
        )

    def payload(aid):  # a valid request whose only variable is the article link
        return {
            "requestorName": "Rene Requester",
            "titleShortDescription": "Docking stations",
            "vendorName": "Hudson Systems Inc.",
            "vatId": "12-3456789",
            "commodityGroupId": 29,  # Hardware
            "totalCost": price * 2,
            "department": "IT",
            "orderLines": [
                {
                    "positionDescription": "Docking station",
                    "unitPrice": price,
                    "amount": 2,
                    "unit": "piece",
                    "totalPrice": price * 2,
                    "articleId": aid,
                }
            ],
        }

    r = client.post("/requests", json=payload(article_id), headers=_auth(token))
    assert r.status_code == 201, r.text
    line = r.json()["order_lines"][0]
    assert line["article_id"] == article_id  # stored
    assert line["article_number"] == article_number  # and joined back for display

    # the link survives a reload
    r = client.get(f"/requests/{r.json()['id']}", headers=_auth(token))
    assert r.json()["order_lines"][0]["article_number"] == article_number

    # an article id from outside the org (or made up) is refused
    r = client.post("/requests", json=payload(str(uuid.uuid4())), headers=_auth(token))
    assert r.status_code == 422
    assert "not in your catalog" in r.json()["detail"]

    # lines without an article are still fine
    r = client.post("/requests", json=payload(None), headers=_auth(token))
    assert r.status_code == 201
    assert r.json()["order_lines"][0]["article_id"] is None
