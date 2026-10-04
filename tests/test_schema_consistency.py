"""
Repo/database consistency guards.

A database restart once exposed a mismatch between the repo and the live
database: the stored embeddings were 1536-dimensional while the app embedded
queries at 1536, and row-level security allowed only the service role while the
app queries with the anon key (so retrieval silently returned zero rows).
These checks keep the pieces that must agree in agreement.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _code_default_dim():
    m = re.search(r'os\.getenv\("EMBEDDING_DIM",\s*"(\d+)"\)', _read("src/embeddings/embedder.py"))
    assert m, "EMBEDDING_DIM default not found in embedder.py"
    return int(m.group(1))


def test_schema_vector_dimensions_match_the_code_default():
    dims = {int(n) for n in re.findall(r"vector\((\d+)\)", _read("infra/supabase_schema.sql"))}
    assert dims == {_code_default_dim()}, f"schema {dims} vs code default {_code_default_dim()}"


def test_render_blueprint_sets_the_same_dimension():
    m = re.search(r'key:\s*EMBEDDING_DIM\s*\n\s*value:\s*"?(\d+)"?', _read("render.yaml"))
    assert m, "render.yaml must set EMBEDDING_DIM"
    assert int(m.group(1)) == _code_default_dim()


def test_schema_lets_the_anon_role_read_the_document_tables():
    sql = _read("infra/supabase_schema.sql")
    for table in ("documents", "document_embeddings"):
        assert re.search(rf"alter table {table} enable row level security", sql)
        assert re.search(
            rf"create policy \"[\w]+\"\s+on public\.{table} for select to anon", sql
        ), f"no anon SELECT policy for {table}: retrieval would silently return nothing"
