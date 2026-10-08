from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.core.deps import get_current_membership
from app.db.session import get_db
from app.models import Article, OrganizationMember
from app.schemas.articles import (
    ArticleOut,
    ArticlePage,
    ArticleSuggestion,
    SuggestRequest,
    SuggestResponse,
)
from app.services.catalog import CatalogIndex

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("", response_model=ArticlePage)
def list_articles(
    search: str | None = Query(
        default=None, description="Filter by description or article number (case-insensitive)"
    ),
    supplier_id: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    membership: OrganizationMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
) -> ArticlePage:
    filters = [Article.organization_id == membership.organization_id]
    if search:
        term = f"%{search}%"
        filters.append(
            or_(Article.description.ilike(term), Article.article_number.ilike(term))
        )
    if supplier_id:
        filters.append(Article.supplier_id == supplier_id)

    total = db.scalar(
        select(func.count()).select_from(Article).where(*filters)
    ) or 0

    stmt = (
        select(Article)
        .where(*filters)
        .options(joinedload(Article.supplier))
        .order_by(Article.article_number)
        .offset(offset)
        .limit(limit)
    )
    items = [
        ArticleOut(
            id=a.id,
            article_number=a.article_number,
            supplier_id=a.supplier_id,
            supplier_name=a.supplier.name if a.supplier else None,
            description=a.description,
            unit_price=a.unit_price,
            currency=a.currency,
            unit=a.unit,
            quantity=a.quantity,
            created_at=a.created_at,
        )
        for a in db.scalars(stmt)
    ]
    return ArticlePage(items=items, total=total, limit=limit, offset=offset)


@router.get("/count")
def count_articles(
    membership: OrganizationMember = Depends(get_current_membership),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    total = (
        db.scalar(
            select(func.count())
            .select_from(Article)
            .where(Article.organization_id == membership.organization_id)
        )
        or 0
    )
    return {"count": total}


@router.post("/suggest", response_model=SuggestResponse, response_model_by_alias=True)  # camelCase on the wire
def suggest_articles(
    payload: SuggestRequest,  # {"lines": [{"positionDescription": ...}], "limit": 3}
    membership: OrganizationMember = Depends(get_current_membership),  # resolves the caller's organization
    db: Session = Depends(get_db),
) -> SuggestResponse:
    """Match free-text order lines against the organization's article catalog.

    Returns, per line, the best-matching articles with their negotiated
    prices (best match first, cheapest first among equal matches). Used by the
    request form to offer "use the negotiated price" inline.
    """
    articles = list(  # the whole org catalog, a few thousand rows at most
        db.scalars(
            select(Article)
            .where(Article.organization_id == membership.organization_id)  # tenant isolation
            .options(joinedload(Article.supplier))  # supplier name in the same query
        )
    )
    index = CatalogIndex(articles)  # tokenise + IDF once per request, reused for every line
    suggestions = [
        [
            ArticleSuggestion(  # ORM row + score -> wire format
                article_id=m.article.id,
                article_number=m.article.article_number,
                description=m.article.description,
                supplier_id=m.article.supplier_id,
                supplier_name=m.article.supplier.name if m.article.supplier else None,
                unit_price=float(m.article.unit_price),
                currency=m.article.currency,
                unit=m.article.unit,
                score=m.score,
                matched_terms=m.matched_terms,
            )
            for m in index.suggest(line.position_description, limit=payload.limit)  # best matches for this line
        ]
        for line in payload.lines  # one inner list per input line, same order
    ]
    return SuggestResponse(suggestions=suggestions)
