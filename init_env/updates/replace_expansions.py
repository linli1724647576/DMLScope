import random

import sqlglot
from sqlglot import exp

from init_env.core.dialect import dialect_option
from init_env.core.rand_utils import rand_str
from init_env.core.row_builder import build_row
from init_env.core.schema_utils import table_partition_names
from init_env.updates.replace_tests import test_replace_variant
from mutate_DML.original_insert import (
    INSERT_FALLBACK_VALUES_PROB,
    INSERT_SELECT_CTE_PROB,
    INSERT_SELECT_SOURCE_TYPES,
    INSERT_TOPLEVEL_FROM_SUBQUERY_PROB,
    INSERT_TOPLEVEL_SETOP_PROB,
    _fresh_insert_pk_expr,
    _parse_query,
    _insert_select_expr,
    _insert_select_source,
    _limit_insert_target_rows,
    _maybe_group_insert_select,
    _render_sql,
    _select_stmt_for_setop,
    _type_tag_for_col_type,
    _wrap_insert_top_level_from_subquery,
    _wrap_insert_top_level_setop,
    _wrap_source_with_stable_pk,
)


REPLACE_VALUES_PROB = INSERT_FALLBACK_VALUES_PROB


def _pick_table(schema: dict) -> str:
    return random.choice(list(schema.keys()))


def _pick_safe_table(schema: dict) -> str:
    inbound = {t: 0 for t in schema}
    for info in schema.values():
        for fk in info["foreign_keys"]:
            inbound[fk["ref_table"]] += 1
    candidates = [t for t, count in inbound.items() if count == 0]
    return random.choice(candidates) if candidates else _pick_table(schema)


def _replace_prefix(dialect: str) -> str:
    allow_low = dialect_option(dialect, "supports_replace_low_priority", True)
    allow_delayed = dialect_option(dialect, "supports_replace_delayed", True)
    r = random.random()
    if allow_low and r < 0.2:
        opt = " LOW_PRIORITY"
    elif allow_delayed and r < 0.4:
        opt = " DELAYED"
    else:
        opt = ""
    into = " INTO" if random.random() < 0.8 else ""
    return f"REPLACE{opt}{into}"


def _partition_clause(schema: dict, table: str, dialect: str) -> str:
    if not dialect_option(dialect, "supports_dml_partition_clause", False):
        return ""
    names = table_partition_names(schema, table)
    if not names or random.random() >= 0.3:
        return ""
    picked = random.sample(names, k=random.randint(1, min(2, len(names))))
    return f" PARTITION ({', '.join(picked)})"


def _replace_target(schema: dict, table: str, dialect: str) -> str:
    return f"{table}{_partition_clause(schema, table, dialect)}"


def _next_replace_row(schema: dict, table_ids: dict, dialect: str) -> tuple[str, list[str], dict]:
    table = _pick_safe_table(schema)
    cols = list(schema[table]["columns"].keys())
    pk_base = max(table_ids[table]) if table_ids.get(table) else 0
    pk = pk_base + random.randint(1, 3)
    row = build_row(schema, table, pk, table_ids, dialect)
    table_ids.setdefault(table, []).append(pk)
    return table, cols, row


def _replace_values_sql(prefix: str, target: str, cols: list[str], values: list[str]) -> str:
    return f"{prefix} {target} ({', '.join(cols)}) VALUES ({', '.join(values)});"


def _apply_replace_top_level_wrappers(
    query: exp.Expression,
    cols: list[str],
    dialect: str,
    *,
    schema: dict,
    target_table: str,
    table_ids: dict,
) -> exp.Expression:
    roll = random.random()
    if roll < INSERT_TOPLEVEL_SETOP_PROB:
        return _wrap_insert_top_level_setop(
            query,
            cols,
            dialect,
            schema=schema,
            target_table=target_table,
            table_ids=table_ids,
        )
    if roll < INSERT_TOPLEVEL_SETOP_PROB + INSERT_TOPLEVEL_FROM_SUBQUERY_PROB:
        return _wrap_insert_top_level_from_subquery(query, cols)
    return query


def _maybe_replace_statement_cte(
    select_ast: exp.Select,
    *,
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
) -> tuple[str, exp.Select]:
    if (
        not dialect_option(dialect, "supports_dml_with", True)
        or not dialect_option(dialect, "supports_replace_with", False)
        or random.random() >= INSERT_SELECT_CTE_PROB
    ):
        return "", select_ast
    type_tag = random.choice(INSERT_SELECT_SOURCE_TYPES)
    cte_stmt = _select_stmt_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth=1,
        target_table=table,
        protected_tables={table},
        column_count=1,
    )
    cte_name = f"cte_rep_{rand_str(3, 5)}"
    cte_select = sqlglot.select(
        exp.column("k", table="cte_sub").as_("c1")
    ).from_(_parse_query(cte_stmt, dialect).subquery("cte_sub"))
    select_ast = select_ast.where(
        exp.Exists(this=sqlglot.select(exp.Literal.number(1)).from_(exp.to_table(cte_name))),
        copy=False,
    )
    return f"WITH {cte_name} AS ({_render_sql(cte_select, dialect)}) ", select_ast


def replace_values(schema: dict, table_ids: dict, dialect: str) -> str:
    table, cols, row = _next_replace_row(schema, table_ids, dialect)
    prefix = _replace_prefix(dialect)
    target = _replace_target(schema, table, dialect)
    return _replace_values_sql(prefix, target, cols, [row[col] for col in cols])


def replace_set(schema: dict, table_ids: dict, dialect: str) -> str:
    return replace_values(schema, table_ids, dialect)


def replace_select(schema: dict, table_ids: dict, dialect: str) -> str:
    table, cols, row = _next_replace_row(schema, table_ids, dialect)
    prefix = _replace_prefix(dialect)
    target = _replace_target(schema, table, dialect)

    source = _insert_select_source(schema, table, table_ids, dialect)
    source, stable_pk_expr = _wrap_source_with_stable_pk(source, dialect)
    expr_types = [_type_tag_for_col_type(schema[table]["columns"].get(col)) for col in cols]
    pk_col = schema[table]["pk"]

    select_items: list[exp.Expression] = []
    for col in cols:
        if col == pk_col:
            select_items.append(_fresh_insert_pk_expr(row[col], stable_pk_expr, dialect))
        else:
            select_items.append(_insert_select_expr(schema, table, col, row[col], table_ids, dialect, source))

    select_ast = sqlglot.select(*select_items)
    if source.from_expression is not None:
        select_ast.set("from_", source.from_expression)
    if source.joins:
        select_ast.set("joins", source.joins)
    for filt in source.filters:
        select_ast = select_ast.where(filt, copy=False)

    select_ast = _maybe_group_insert_select(
        select_ast,
        select_items=select_items,
        expr_types=expr_types,
        dialect=dialect,
    )
    if dialect_option(dialect, "supports_select_distinct", True) and random.random() < 0.25:
        select_ast.set("distinct", exp.Distinct())

    with_sql, select_ast = _maybe_replace_statement_cte(
        select_ast,
        schema=schema,
        table=table,
        table_ids=table_ids,
        dialect=dialect,
    )

    select_ast = _apply_replace_top_level_wrappers(
        select_ast,
        cols,
        dialect,
        schema=schema,
        target_table=table,
        table_ids=table_ids,
    )
    select_ast = _limit_insert_target_rows(select_ast, dialect=dialect)

    select_sql = _render_sql(select_ast, dialect)
    return f"{with_sql}{prefix} {target} ({', '.join(cols)}) {select_sql};"


def replace_unified(schema: dict, table_ids: dict, dialect: str) -> str:
    if random.random() < REPLACE_VALUES_PROB:
        return replace_values(schema, table_ids, dialect)
    return replace_select(schema, table_ids, dialect)


def build_replace_variants(schema: dict, table_ids: dict, dialect: str) -> list:
    variants = []
    replace_count = random.randint(1, max(2, len(schema) // 2))
    for _ in range(replace_count):
        stmt = replace_unified(schema, table_ids, dialect)
        test_replace_variant(stmt)
        variants.append(stmt)
    return variants
