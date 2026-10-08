from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sqlalchemy import select  # look up the caller's membership

from app.core.deps import get_current_user  # auth: any signed-in user
from app.db.session import get_db
from app.models import OrganizationMember, User  # resolve user -> organization for the rules
from app.schemas.extraction import ExtractionRequest, ExtractionResponse
from app.services.extraction import extract_vendor_data

router = APIRouter(prefix="/extraction", tags=["extraction"])


@router.post("", response_model=ExtractionResponse, response_model_by_alias=True)
def extract(
    payload: ExtractionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ExtractionResponse:
    """Extract structured vendor data from PDF text via OpenAI.

    Returns ``{success, data?, error?, missingFields?}`` — the same envelope the
    original server action produced (errors are part of the payload, not HTTP
    errors, so the UI can show partial-extraction warnings).
    """
    # Users without an organization can still extract; they just get no org rules.
    membership = db.scalar(
        select(OrganizationMember).where(OrganizationMember.user_id == user.id)
    )
    organization_id = membership.organization_id if membership else None
    return extract_vendor_data(payload.text, db, organization_id)
