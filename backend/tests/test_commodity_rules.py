"""Per-organization commodity rules (task 3)."""
from app.services import extraction as svc  # to stub the model call
from app.services.classification import match_rules  # pure rule matcher
from tests.test_extraction import _llm  # canned model answer (Greenmantle quote)
from tests.test_smoke import _auth, _signin, client  # shared seeded test app

ELECTRICAL = 11  # Facility Management / Electrical Engineering
HARDWARE = 29  # Information Technology / Hardware


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
def test_match_rules_needs_every_keyword_word_and_keeps_order():
    rules = [
        {"keyword": "cable ties", "commodity_group_id": ELECTRICAL},
        {"keyword": "toner", "commodity_group_id": HARDWARE},
    ]
    assert match_rules(rules, ["Nylon cable tie 200 mm, black"])["commodity_group_id"] == ELECTRICAL  # plural stemmed
    assert match_rules(rules, ["Toner cartridge cyan"])["commodity_group_id"] == HARDWARE  # one word is enough when the rule has one
    assert match_rules(rules, ["Charging cable USB-C"]) is None  # "cable" alone is not "cable ties"
    assert match_rules(rules, ["Cable ties", "Toner"])["commodity_group_id"] == ELECTRICAL  # first rule wins
    assert match_rules([], ["anything"]) is None  # no rules
    assert match_rules(rules, ["", None]) is None  # blank texts


# ---------------------------------------------------------------------------
# Settings endpoint
# ---------------------------------------------------------------------------
def test_acme_is_seeded_with_its_booking_rules():
    token = _signin("admin@acme.com")
    r = client.get("/organizations/me/settings", headers=_auth(token))
    assert r.status_code == 200
    rules = r.json()["commodity_rules"]
    by_keyword = {rule["keyword"]: rule["commodity_group_id"] for rule in rules}
    assert by_keyword["cable ties"] == ELECTRICAL
    assert by_keyword["toner"] == HARDWARE
    assert "software subscription" in by_keyword


def test_rules_are_validated_and_admin_only():
    admin = _signin("admin@acme.com")
    r = client.get("/organizations/me/settings", headers=_auth(admin))
    current = r.json()  # keep Acme's seeded rules intact for the other tests

    # a rule pointing at a non-existent group is refused
    bad = dict(current, commodity_rules=[{"keyword": "widgets", "commodity_group_id": 999999}])
    r = client.patch("/organizations/me/settings", json=bad, headers=_auth(admin))
    assert r.status_code == 422
    assert "999999" in r.json()["detail"]

    # a keyword shorter than two characters is refused by the schema
    bad = dict(current, commodity_rules=[{"keyword": "x", "commodity_group_id": HARDWARE}])
    assert client.patch("/organizations/me/settings", json=bad, headers=_auth(admin)).status_code == 422

    # only admins may change settings
    requester = _signin("requester@acme.com")
    assert client.patch("/organizations/me/settings", json=current, headers=_auth(requester)).status_code == 403

    # a valid round-trip keeps the rules
    r = client.patch("/organizations/me/settings", json=current, headers=_auth(admin))
    assert r.status_code == 200
    assert r.json()["commodity_rules"] == current["commodity_rules"]


# ---------------------------------------------------------------------------
# Extraction: the rule overrides the AI's classification
# ---------------------------------------------------------------------------
def test_extraction_applies_org_rule_and_explains_it(monkeypatch):
    monkeypatch.setattr(svc.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(  # the AI says "Office Equipment" for a toner line
        svc,
        "_call_model",
        lambda text, taxonomy: _llm(title="Printer toner restock", commodity_group_name="Office Equipment"),
    )
    token = _signin("requester@acme.com")
    r = client.post("/extraction", json={"text": "Toner cartridge quote"}, headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"]["commodityGroupId"] == HARDWARE  # Acme's rule wins
    assert body["data"]["commodityGroupName"] == "Hardware"
    assert 'rule for "toner"' in body["classificationNote"]

    # a user with no organization gets the AI's answer and no note
    r = client.post("/auth/signup", json={"email": "noorg@example.com", "password": "secret123"})
    r = client.post("/extraction", json={"text": "Toner"}, headers=_auth(r.json()["access_token"]))
    assert r.json()["data"]["commodityGroupId"] == 15  # Office Equipment, as the AI said
    assert r.json()["classificationNote"] is None


# ---------------------------------------------------------------------------
# Requests: the rule is enforced on save, with an audit entry
# ---------------------------------------------------------------------------
def test_request_save_enforces_rule_and_logs_it():
    token = _signin("requester@acme.com")
    payload = {
        "requestorName": "Rene Requester",
        "titleShortDescription": "Office supplies",
        "vendorName": "Lakeview Supply LP",
        "vatId": "12-3456789",
        "commodityGroupId": 15,  # the requester picked Office Equipment in the dropdown
        "totalCost": 186.0,
        "department": "Ops",
        "orderLines": [
            {"positionDescription": "Toner cartridge black, high-yield", "unitPrice": 186.0,
             "amount": 1, "unit": "piece", "totalPrice": 186.0}
        ],
    }
    r = client.post("/requests", json=payload, headers=_auth(token))
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["commodity_group_id"] == HARDWARE  # non-negotiable
    assert created["commodity_groups"]["name"] == "Hardware"

    r = client.get(f"/requests/{created['id']}/activity", headers=_auth(token))
    assert r.status_code == 200
    summaries = [a["summary"] for a in r.json()]
    assert any('organization rule "toner"' in s for s in summaries)  # the override is visible in the log

    # editing the request cannot move it off the rule's group either
    update = dict(payload, title=payload.pop("titleShortDescription"), commodityGroupId=31)
    r = client.put(f"/requests/{created['id']}", json=update, headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json()["commodity_group_id"] == HARDWARE

    # a request that matches no rule keeps the submitted group
    plain = dict(payload, titleShortDescription="Chairs", commodityGroupId=15,
                 orderLines=[{"positionDescription": "Office chair", "unitPrice": 186.0,
                              "amount": 1, "unit": "piece", "totalPrice": 186.0}])
    r = client.post("/requests", json=plain, headers=_auth(token))
    assert r.status_code == 201 and r.json()["commodity_group_id"] == 15
