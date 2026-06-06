"""add pinned flag to chat sessions

Revision ID: 0003_chat_session_pinned
Revises: 0002_profile_cols
Create Date: 2026-06-06
"""

from alembic import op


revision = "0003_chat_session_pinned"
down_revision = "0002_profile_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute(
        "CREATE INDEX IF NOT EXISTS chat_sessions_owner_pinned_updated_idx "
        "ON chat_sessions (owner_id, pinned DESC, updated_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chat_sessions_owner_pinned_updated_idx")
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS pinned")
