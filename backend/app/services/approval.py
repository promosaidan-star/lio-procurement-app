"""Auto-approval policy (bonus task).

"Requiring a buyer to sign off on every single request wastes too much of
their time. Anything below $2,000 shouldn't need approval at all."

The threshold is an organization setting (``auto_approve_below``); ``None``
keeps the original behaviour where every request waits for a buyer. A request
auto-approved by the policy has ``approved_by = None``, which is how the UI
and this module tell it apart from a buyer's decision. If such a request is
later edited above the threshold it goes back to pending; a buyer's decision
is never undone by the policy.
"""
from __future__ import annotations  # forward references in type hints

import uuid  # organization id type
from datetime import datetime, timezone  # approved_at timestamps

from sqlalchemy.orm import Session  # DB session type

from app.models import Organization, ProcurementRequest  # ORM rows read / updated here


def auto_approve_threshold(db: Session, organization_id: uuid.UUID | None) -> float | None:
    """The org's threshold, or None when every request needs a buyer."""
    if organization_id is None:  # no organization context
        return None
    org = db.get(Organization, organization_id)  # primary-key lookup
    if org is None:
        return None
    value = (org.settings or {}).get("auto_approve_below")  # absent on orgs that never set it
    return float(value) if value is not None else None  # JSON may hold int or float


def apply_approval_policy(request: ProcurementRequest, threshold: float | None) -> str | None:
    """Set the request's approval state from its total. Returns an audit summary or None.

    Only requests the policy itself decided (``approved_by is None``) are ever
    changed here; anything a buyer approved or rejected stays as decided.
    """
    if threshold is None:  # policy disabled for this organization
        return None
    if request.approved_by is not None:  # a human decided: never override
        return None
    total = float(request.total_cost or 0)  # None total counts as zero
    below = total < threshold  # strictly below: exactly $2,000 still needs a buyer

    if below and request.approval_status == "pending":  # new or re-edited request under the limit
        request.approval_status = "approved"
        request.approved_at = datetime.now(timezone.utc)  # when the policy approved it
        return f"Auto-approved: total ${total:,.2f} is below the ${threshold:,.2f} threshold"

    if not below and request.approval_status == "approved":  # auto-approved earlier, now edited above the limit
        request.approval_status = "pending"
        request.approved_at = None
        return f"Approval required again: total ${total:,.2f} reached the ${threshold:,.2f} threshold"

    return None  # nothing to change
