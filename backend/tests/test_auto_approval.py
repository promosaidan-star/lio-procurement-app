"""Auto-approval below an organization threshold (bonus task)."""
from app.models import ProcurementRequest  # for the pure-function tests
from app.services.approval import apply_approval_policy  # the policy itself
from tests.test_smoke import _auth, _signin, client  # shared seeded test app


def _payload(total: float, title: str = "Chairs") -> dict:
    """A valid Acme request whose only variable is the total."""
    return {
        "requestorName": "Rene Requester",
        "titleShortDescription": title,
        "vendorName": "Montclair Supply Corp.",
        "vatId": "12-3456789",
        "commodityGroupId": 15,  # Office Equipment; matches no commodity rule
        "totalCost": total,
        "department": "Ops",
        "orderLines": [
            {"positionDescription": "Office chair", "unitPrice": total, "amount": 1,
             "unit": "piece", "totalPrice": total}
        ],
    }


# ---------------------------------------------------------------------------
# Policy as a pure function
# ---------------------------------------------------------------------------
def test_policy_is_strictly_below_and_never_overrides_a_human():
    r = ProcurementRequest(total_cost=1999.99, approval_status="pending", approved_by=None)
    assert "Auto-approved" in apply_approval_policy(r, 2000.0)  # just under
    assert r.approval_status == "approved" and r.approved_at is not None

    r = ProcurementRequest(total_cost=2000.0, approval_status="pending", approved_by=None)
    assert apply_approval_policy(r, 2000.0) is None  # exactly the threshold still needs a buyer
    assert r.approval_status == "pending"

    r = ProcurementRequest(total_cost=50.0, approval_status="pending", approved_by=None)
    assert apply_approval_policy(r, None) is None  # policy off

    r = ProcurementRequest(total_cost=5000.0, approval_status="approved", approved_by="someone")
    assert apply_approval_policy(r, 2000.0) is None  # a buyer's approval is never undone
    assert r.approval_status == "approved"

    r = ProcurementRequest(total_cost=5000.0, approval_status="approved", approved_by=None)
    assert "required again" in apply_approval_policy(r, 2000.0)  # auto-approved earlier, now too big
    assert r.approval_status == "pending" and r.approved_at is None


# ---------------------------------------------------------------------------
# Through the API on the seeded Acme org ($2,000 threshold)
# ---------------------------------------------------------------------------
def test_acme_is_seeded_with_the_threshold_and_it_is_validated():
    admin = _signin("admin@acme.com")
    r = client.get("/organizations/me/settings", headers=_auth(admin))
    assert r.json()["auto_approve_below"] == 2000.0

    current = r.json()
    bad = dict(current, auto_approve_below=-5)  # negative makes no sense
    assert client.patch("/organizations/me/settings", json=bad, headers=_auth(admin)).status_code == 422


def test_small_requests_skip_the_buyer_and_large_ones_do_not():
    requester = _signin("requester@acme.com")

    r = client.post("/requests", json=_payload(1900.0), headers=_auth(requester))
    assert r.status_code == 201, r.text
    small = r.json()
    assert small["approval_status"] == "approved"
    assert small["approved_by"] is None  # the policy, not a person
    assert small["approved_at"] is not None
    activity = client.get(f"/requests/{small['id']}/activity", headers=_auth(requester)).json()
    assert any("Auto-approved" in a["summary"] for a in activity)  # visible in the log

    r = client.post("/requests", json=_payload(2000.0), headers=_auth(requester))
    assert r.status_code == 201
    assert r.json()["approval_status"] == "pending"  # not strictly below

    # a buyer cannot "decide" an auto-approved request twice
    buyer = _signin("buyer@acme.com")
    r = client.patch(
        f"/requests/{small['id']}/approval", json={"decision": "rejected"}, headers=_auth(buyer)
    )
    assert r.status_code == 409


def test_editing_across_the_threshold_updates_approval():
    requester = _signin("requester@acme.com")
    r = client.post("/requests", json=_payload(500.0), headers=_auth(requester))
    created = r.json()
    assert created["approval_status"] == "approved"

    # edit it up to $2,500: back to pending
    update = dict(_payload(2500.0), title="Chairs")
    update.pop("titleShortDescription")
    r = client.put(f"/requests/{created['id']}", json=update, headers=_auth(requester))
    assert r.status_code == 200, r.text
    assert r.json()["approval_status"] == "pending"
    assert r.json()["approved_at"] is None

    # and back down: auto-approved again
    update = dict(_payload(800.0), title="Chairs")
    update.pop("titleShortDescription")
    r = client.put(f"/requests/{created['id']}", json=update, headers=_auth(requester))
    assert r.json()["approval_status"] == "approved"


def test_other_tenants_keep_full_approval():
    walmart = _signin("admin@walmart.com")
    r = client.post("/requests", json=_payload(10.0), headers=_auth(walmart))
    assert r.status_code == 201, r.text
    assert r.json()["approval_status"] == "pending"  # no threshold configured
