import copy
import os
import random
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from init_env.core.dialect import dialect_option
from init_env.updates.delete_expansions import delete_unified
from init_env.updates.delete_tests import test_delete_variant
from init_env.updates.replace_expansions import replace_unified
from init_env.updates.replace_tests import test_replace_variant
from init_env.updates.update_expansions import (
    update_unified,
)
from init_env.updates.update_tests import test_update_variant
from mutate_DML.original_insert import (
    DUCKDB_INSERT_FALLBACK_VALUES_PROB,
    DUCKDB_INSERT_ON_DUPLICATE_PROB,
    DUCKDB_INSERT_VALUES_PROB,
    INSERT_FALLBACK_VALUES_PROB,
    INSERT_ON_DUPLICATE_PROB,
    INSERT_ON_DUPLICATE_VALUES_PROB,
    INSERT_VALUES_PROB,
    _insert_on_duplicate_values,
    _insert_select,
    _insert_values,
)


def _log_generation_error(error_log_path: str | None, *, attempt: int, exc: Exception) -> None:
    if not error_log_path:
        return
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"-- GENERATION ERROR attempt={attempt}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc().rstrip())
        f.write("\n\n")


def _generate_update(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    join_where_target_only: bool = False,
) -> str:
    stmt = update_unified(
        schema,
        table_ids,
        dialect,
        join_where_target_only=join_where_target_only,
    )
    test_update_variant(stmt)
    return stmt


def _generate_delete(schema: dict, table_ids: dict, dialect: str) -> str:
    stmt = delete_unified(schema, table_ids, dialect)
    test_delete_variant(stmt)
    return stmt


def _generate_replace_or_insert_fallback(schema: dict, table_ids: dict, dialect: str) -> str:
    if dialect == "duckdb":
        if random.random() < DUCKDB_INSERT_FALLBACK_VALUES_PROB:
            return _insert_values(schema, table_ids, dialect, replace=True)
        return _insert_select(schema, table_ids, dialect, replace=True)

    if not dialect_option(dialect, "supports_replace_statement", True):
        if random.random() < INSERT_FALLBACK_VALUES_PROB:
            return _insert_values(schema, table_ids, dialect)
        return _insert_select(schema, table_ids, dialect)

    stmt = replace_unified(schema, table_ids, dialect if dialect != "duckdb" else "mysql")
    test_replace_variant(stmt)
    return stmt


def _generate_insert(schema: dict, table_ids: dict, dialect: str) -> str:
    r_ins = random.random()
    upsert_prob = INSERT_ON_DUPLICATE_PROB
    values_prob = INSERT_VALUES_PROB
    if dialect == "duckdb":
        upsert_prob = DUCKDB_INSERT_ON_DUPLICATE_PROB
        values_prob = DUCKDB_INSERT_VALUES_PROB

    if dialect != "duckdb":
        if dialect_option(dialect, "supports_on_duplicate_key_update", True) and r_ins < upsert_prob:
            if random.random() < INSERT_ON_DUPLICATE_VALUES_PROB:
                return _insert_on_duplicate_values(schema, table_ids, dialect)
            return _insert_select(schema, table_ids, dialect, upsert=True)
        if r_ins < upsert_prob + values_prob:
            return _insert_values(schema, table_ids, dialect)
        return _insert_select(schema, table_ids, dialect)

    if r_ins < upsert_prob:
        if random.random() < INSERT_ON_DUPLICATE_VALUES_PROB:
            return _insert_on_duplicate_values(schema, table_ids, dialect)
        return _insert_select(schema, table_ids, dialect, upsert=True)
    if r_ins < upsert_prob + values_prob:
        return _insert_values(schema, table_ids, dialect)
    return _insert_select(schema, table_ids, dialect)


def _generate_one_original_dml(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    join_where_target_only: bool = False,
) -> str:
    r = random.random()
    if r < 0.65:
        return _generate_update(
            schema,
            table_ids,
            dialect,
            join_where_target_only=join_where_target_only,
        )
    if r < 0.85:
        return _generate_delete(schema, table_ids, dialect)
    if r < 0.95:
        return _generate_insert(schema, table_ids, dialect)
    return _generate_replace_or_insert_fallback(schema, table_ids, dialect)


def generate_original_dml(
    schema: dict,
    table_ids: dict,
    dialect: str,
    count: int = 1000,
    error_log_path: str | None = None,
    max_attempts: int | None = None,
    join_where_target_only: bool = False,
) -> str:
    statements = []
    attempts = 0
    failures = 0
    max_attempts = max_attempts or max(count * 3, count + 20)
    if error_log_path:
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("")

    while len(statements) < count and attempts < max_attempts:
        attempts += 1
        snapshot = copy.deepcopy(table_ids)
        try:
            stmt = _generate_one_original_dml(
                schema,
                table_ids,
                dialect,
                join_where_target_only=join_where_target_only,
            )
        except Exception as exc:
            failures += 1
            table_ids.clear()
            table_ids.update(snapshot)
            _log_generation_error(error_log_path, attempt=attempts, exc=exc)
            print(f"[ori-dml] skip failed statement attempt={attempts}: {type(exc).__name__}: {exc}")
            continue
        statements.append(stmt)

    if failures:
        print(f"[ori-dml] generation failures={failures} generated={len(statements)} attempts={attempts}")
    if len(statements) < count:
        print(f"[ori-dml] warning: requested={count} generated={len(statements)} max_attempts={max_attempts}")
    return "\n".join(statements) + "\n"
