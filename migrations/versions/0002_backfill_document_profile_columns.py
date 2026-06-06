"""backfill document profile columns

Revision ID: 0002_profile_cols
Revises: 0001_baseline_schema
Create Date: 2026-06-04
"""

from alembic import op


revision = "0002_profile_cols"
down_revision = "0001_baseline_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS conclusion TEXT")
    op.execute("ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo'")
    op.execute("ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS profile_json JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS outline_json JSONB NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE document_profiles ADD COLUMN IF NOT EXISTS source_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE document_profile_embeddings ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'demo'")
    op.execute("CREATE INDEX IF NOT EXISTS document_profiles_owner_document_idx ON document_profiles (owner_id, document_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS document_profile_embeddings_owner_document_idx "
        "ON document_profile_embeddings (owner_id, document_id, field_name)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS document_profile_embeddings_owner_document_idx")
    op.execute("DROP INDEX IF EXISTS document_profiles_owner_document_idx")
    op.execute("ALTER TABLE document_profile_embeddings DROP COLUMN IF EXISTS owner_id")
    op.execute("ALTER TABLE document_profiles DROP COLUMN IF EXISTS source_refs_json")
    op.execute("ALTER TABLE document_profiles DROP COLUMN IF EXISTS outline_json")
    op.execute("ALTER TABLE document_profiles DROP COLUMN IF EXISTS profile_json")
    op.execute("ALTER TABLE document_profiles DROP COLUMN IF EXISTS owner_id")
    op.execute("ALTER TABLE document_profiles DROP COLUMN IF EXISTS conclusion")
