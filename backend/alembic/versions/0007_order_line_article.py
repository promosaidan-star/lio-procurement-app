"""link order lines to catalog articles

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-08

Adds a nullable ``article_id`` to ``order_lines`` so a request records which
negotiated catalog article (if any) a line was taken from. Deleting an article
keeps the line and clears the link.
"""
from collections.abc import Sequence  # type for the optional branch/dependency labels below

import sqlalchemy as sa  # column and type definitions
from alembic import op  # schema operations (add column, constraints, indexes)

# revision identifiers, used by Alembic.
revision: str = "0007"  # this migration's id
down_revision: str | None = "0006"  # runs after the master-data scoping migration
branch_labels: str | Sequence[str] | None = None  # single linear history, no branches
depends_on: str | Sequence[str] | None = None  # no cross-branch dependencies


def upgrade() -> None:
    op.add_column("order_lines", sa.Column("article_id", sa.Uuid(), nullable=True))  # nullable: lines without a catalog article stay valid
    op.create_foreign_key(
        "fk_order_lines_article_id",  # constraint name, needed again in downgrade()
        "order_lines",  # child table
        "articles",  # parent table
        ["article_id"],  # child column
        ["id"],  # parent column
        ondelete="SET NULL",  # deleting an article keeps the line, just drops the link
    )
    op.create_index("ix_order_lines_article_id", "order_lines", ["article_id"])  # fast "which requests used article X" lookups


def downgrade() -> None:
    op.drop_index("ix_order_lines_article_id", table_name="order_lines")  # reverse in the opposite order
    op.drop_constraint("fk_order_lines_article_id", "order_lines", type_="foreignkey")  # constraint before column
    op.drop_column("order_lines", "article_id")  # finally the column itself
