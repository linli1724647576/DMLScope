import random

from init_env.core.dialect import dialect_option, normalize_dialect
from init_env.core.row_builder import build_row
from init_env.core.schema import get_schema
from init_env.core.schema_utils import get_table_order
from init_env.updates.delete_expansions import build_delete_variants
from init_env.updates.delete_tests import test_delete_variants
from init_env.updates.replace_expansions import build_replace_variants
from init_env.updates.replace_tests import test_replace_variants
from init_env.updates.update_expansions import build_update_variants
from init_env.updates.update_tests import test_update_variants


def insert_stmt(table: str, cols: list, row: dict) -> str:
    values = ", ".join(row[c] for c in cols)
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({values});"


def build_dml_with_meta(dialect: str) -> tuple[str, dict]:
    d = normalize_dialect(dialect)
    schema = get_schema()
    rows_per_table = random.randint(3, 8)
    table_ids = {t: list(range(1, rows_per_table + 1)) for t in schema}

    statements = []

    for table in get_table_order(schema):
        cols = list(schema[table]["columns"].keys())
        for pk in table_ids[table]:
            row = build_row(schema, table, pk, table_ids, d)
            statements.append(insert_stmt(table, cols, row))

    update_variants = build_update_variants(schema, table_ids, d)
    test_update_variants(update_variants)
    statements.extend(update_variants)

    if dialect_option(d, "supports_replace_statement", True):
        replace_variants = build_replace_variants(schema, table_ids, d)
        test_replace_variants(replace_variants)
        statements.extend(replace_variants)

    delete_variants = build_delete_variants(schema, table_ids, d)
    test_delete_variants(delete_variants)
    statements.extend(delete_variants)

    return "\n".join(statements) + "\n", table_ids


def build_dml(dialect: str) -> str:
    sql, _ = build_dml_with_meta(dialect)
    return sql
