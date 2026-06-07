"""baseline schema

Revision ID: 0001_baseline_schema
Revises: None
Create Date: 2026-06-04
"""

from pathlib import Path

from alembic import context
from alembic import op


revision = "0001_baseline_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = Path(__file__).resolve().parents[2] / "backend" / "db" / "init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    if context.is_offline_mode():
        op.execute(sql)
    else:
        op.get_bind().exec_driver_sql(sql)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chat_contexts CASCADE")
    op.execute("DROP TABLE IF EXISTS chat_messages CASCADE")
    op.execute("DROP TABLE IF EXISTS chat_sessions CASCADE")
    op.execute("DROP TABLE IF EXISTS document_profile_embeddings CASCADE")
    op.execute("DROP TABLE IF EXISTS document_profiles CASCADE")
    op.execute("DROP TABLE IF EXISTS figures CASCADE")
    op.execute("DROP TABLE IF EXISTS images CASCADE")
    op.execute("DROP TABLE IF EXISTS chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
