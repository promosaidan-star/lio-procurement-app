import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ArticleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    article_number: str
    supplier_id: uuid.UUID
    supplier_name: str | None = None
    description: str
    unit_price: Decimal
    currency: str
    unit: str
    quantity: Decimal
    created_at: datetime


class ArticlePage(BaseModel):
    """A page of articles plus the totals the UI needs to render pagination."""

    items: list[ArticleOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Catalog suggestions for order lines (camelCase: consumed by the request form)
# ---------------------------------------------------------------------------
class SuggestLineIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    position_description: str = Field(alias="positionDescription")


class SuggestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lines: list[SuggestLineIn] = Field(max_length=100)
    limit: int = Field(default=3, ge=1, le=10)


class ArticleSuggestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    article_id: uuid.UUID = Field(alias="articleId")
    article_number: str = Field(alias="articleNumber")
    description: str
    supplier_id: uuid.UUID = Field(alias="supplierId")
    supplier_name: str | None = Field(default=None, alias="supplierName")
    unit_price: float = Field(alias="unitPrice")
    currency: str
    unit: str
    score: float
    matched_terms: list[str] = Field(alias="matchedTerms")


class SuggestResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # One list per input line, in the same order (empty when nothing matched).
    suggestions: list[list[ArticleSuggestion]]
