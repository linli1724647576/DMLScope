import random

import re

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from init_env.core.dialect import (
    dialect_option,
    get_sqlglot_dialect,
    get_supported_join_types,
    get_supported_set_ops,
    supports_limit_offset,
    supports_update_tail,
)
from init_env.core.rand_utils import rand_date, rand_datetime, rand_str, rand_time, sql_str
from init_env.core.schema_utils import column_meta, columns_by_type, non_pk_columns, table_partition_names
from init_env.core.type_compat import available_general_type_tags, column_type_tag, columns_by_type_tag
from init_env.core.value_gen import DATETIME_TYPES, DATE_TYPES, JSON_TYPES, NUMERIC_TYPES, STRING_TYPES, gen_value
from init_env.utils.sqlglot_compat import normalize_sqlglot_input
from init_env.updates.function_catalog import FUNCTION_SPECS
from init_env.updates.update_tests import test_update_variant

SETOP_DERIVED_FROM_PROB = 0.6
SUBQUERY_DERIVED_FROM_PROB = 0.3
SETOP_COMBINE_PROB = 0.7
SETOP_SUBQUERY_PROB = 0.7
UPDATE_SET_SELECT_PROB = 0.75
UPDATE_WHERE_SELECT_PROB = 0.8
WINDOW_FUNCTION_PROB = 0.2
WINDOW_SELECTOR_FUNCTIONS = ("ROW_NUMBER", "RANK", "DENSE_RANK")
WINDOW_VALUE_FUNCTIONS = ("LAG", "LEAD")
COND_NOT_PROB = 0.10
COND_AND_PROB = 0.10
COND_OR_PROB = 0.06
JOIN_ON_EXTRA_TERM1_PROB = 0.45
JOIN_ON_EXTRA_TERM2_PROB = 0.12
WHERE_OP_COMPOSITE_PROB = 0.35
UPDATE_RHS_COMPLEX_PROB = 0.75
UPDATE_RHS_TOPLEVEL_FROM_SUBQUERY_PROB = 0.2
UPDATE_RHS_TOPLEVEL_SETOP_PROB = 0.15
UPDATE_RHS_TOPLEVEL_CTE_PROB = 0.2
UPDATE_RHS_CTE_PROB = 0.35
UPDATE_UNIFIED_JOIN_PROB = 0.85
UPDATE_RHS_SOURCE_FROM_SUBQUERY_PROB = 0.3
UPDATE_RETURNING_PROB = 0.18
DUCKDB_UPDATE_VALUES_SOURCE_PROB = 0.3

ConditionColSpec = tuple[str, str, str]


@dataclass
class UpdateQuerySource:
    from_expression: exp.From | None
    joins: list[exp.Join]
    filters: list[exp.Expression]
    limit: int | None = None
    source_table: str | None = None
    source_alias: str | None = None


def _sqlglot_dialect(dialect: str) -> str:
    return get_sqlglot_dialect(dialect)


def _supports_update_from(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_update_from", False))


def _uses_update_from(dialect: str) -> bool:
    return _supports_update_from(dialect) and not dialect_option(dialect, "supports_update_join", True)


def _date_shift_sql(base_sql: str, amount: int, unit: str, dialect: str) -> str:
    if dialect.lower().strip() == "monetdb":
        return f"({base_sql} + INTERVAL '{amount}' {unit})"
    return f"DATE_ADD({base_sql}, INTERVAL '{amount}' {unit})"


def _normalize_monetdb_sql(sql: str, dialect: str) -> str:
    if dialect.lower().strip() != "monetdb":
        return sql

    def _rewrite_format_literals(text: str) -> str:
        def _rewrite_literal(match: re.Match[str]) -> str:
            content = match.group(1)
            if not (
                any(ch in content for ch in "-:/ %")
                and (
                    re.search(r"%(?:Y|y|m|d|H|M|S|i|s)", content)
                    or re.search(r"(?i)(YYYY|YY|HH24|MM|DD|MI|SS)", content)
                )
            ):
                return match.group(0)
            rewritten = content
            for pattern, repl in (
                (r"(?i)HH24", "%H"),
                (r"(?i)YYYY", "%Y"),
                (r"(?i)YY", "%y"),
                (r"(?i)MI", "%M"),
                (r"(?i)SS", "%S"),
                (r"(?i)DD", "%d"),
                (r"(?i)MM", "%m"),
            ):
                rewritten = re.sub(pattern, repl, rewritten)
            rewritten = rewritten.replace("%i", "%M").replace("%s", "%S")
            return f"'{rewritten}'"

        return re.sub(r"'((?:''|[^'])*)'", _rewrite_literal, text)

    sql = re.sub(
        r"(?i)\bINTERVAL\s+'([+-]?\d+)\s+([A-Z]+)'",
        lambda m: f"INTERVAL '{m.group(1)}' {m.group(2).upper()}",
        sql,
    )
    rewrites = (
        (r"(?i)\bTO_DATE\s*\(", "STR_TO_DATE("),
        (r"(?i)\bTO_TIMESTAMP\s*\(", "STR_TO_TIMESTAMP("),
        (r"(?i)\bDAY_OF_MONTH\s*\(", "DAYOFMONTH("),
        (r"(?i)\bDAY_OF_WEEK\s*\(", "DAYOFWEEK("),
        (r"(?i)\bDAY_OF_YEAR\s*\(", "DAYOFYEAR("),
        (r"(?i)\bSTDDEV\s*\(", "STDDEV_POP("),
        (r"(?i)\bVARIANCE\s*\(", "VAR_POP("),
    )
    for pattern, repl in rewrites:
        sql = re.sub(pattern, repl, sql)
    return _rewrite_format_literals(sql)


def _parse_update_shell(sql: str, dialect: str) -> exp.Update:
    parsed = sqlglot.parse_one(normalize_sqlglot_input(sql, dialect), read=_sqlglot_dialect(dialect))
    if not isinstance(parsed, exp.Update):
        raise ValueError(f"Expected UPDATE statement, got {type(parsed).__name__}: {sql}")
    return parsed


def _parse_update_target(target_sql: str, dialect: str) -> exp.Expression:
    stmt = _parse_update_shell(f"UPDATE {target_sql} SET __ast_col = 1", dialect)
    return stmt.args["this"]


def _parse_update_assignments(set_clause: str, dialect: str) -> list[exp.Expression]:
    stmt = _parse_update_shell(f"UPDATE __ast_target SET {set_clause}", dialect)
    return [expr.copy() for expr in stmt.args.get("expressions") or []]


def _parse_update_where(where_clause: str, dialect: str) -> exp.Where | None:
    stmt = _parse_update_shell(f"UPDATE __ast_target SET __ast_col = 1 WHERE {where_clause}", dialect)
    where = stmt.args.get("where")
    return where.copy() if where else None


def _parse_update_from(from_sql: str, dialect: str) -> exp.From | None:
    if not from_sql:
        return None
    stmt = _parse_update_shell(f"UPDATE __ast_target SET __ast_col = 1 FROM {from_sql} WHERE 1 = 1", dialect)
    from_ = stmt.args.get("from_")
    return from_.copy() if from_ else None


def _parse_update_tail(tail_sql: str, dialect: str) -> tuple[exp.Order | None, exp.Expression | None]:
    if not tail_sql:
        return None, None
    stmt = _parse_update_shell(
        f"UPDATE __ast_target SET __ast_col = 1 WHERE 1 = 1{tail_sql}",
        dialect,
    )
    order = stmt.args.get("order")
    limit = stmt.args.get("limit")
    return order.copy() if order else None, limit.copy() if limit else None


def _parse_update_with(with_clause: str, dialect: str) -> exp.With | None:
    if not with_clause:
        return None
    stmt = _parse_update_shell(f"{with_clause}UPDATE __ast_target SET __ast_col = 1 WHERE 1 = 1", dialect)
    with_ = stmt.args.get("with_")
    return with_.copy() if with_ else None


def _parse_update_returning(returning_sql: str, dialect: str) -> exp.Returning | None:
    if not returning_sql:
        return None
    stmt = _parse_update_shell(
        f"UPDATE __ast_target SET __ast_col = 1 RETURNING {returning_sql}",
        dialect,
    )
    returning = stmt.args.get("returning")
    return returning.copy() if returning else None


def _parse_query_ast(sql: str, dialect: str) -> exp.Query:
    parsed = sqlglot.parse_one(normalize_sqlglot_input(sql, dialect), read=_sqlglot_dialect(dialect))
    if not isinstance(parsed, exp.Query):
        raise ValueError(f"Expected query expression, got {type(parsed).__name__}: {sql}")
    return parsed


def _apply_update_prefix(sql: str, prefix: str) -> str:
    modifiers = prefix.split()[1:]
    if not modifiers:
        return sql
    return re.sub(r"\bUPDATE\b", f"UPDATE {' '.join(modifiers)}", sql, count=1)


def _strip_qualified_update_set(sql: str) -> str:
    sql = re.sub(
        r"(?i)(\bSET\s+)([A-Za-z_][A-Za-z0-9_]*\.)(([A-Za-z_][A-Za-z0-9_]*))\s*=",
        r"\1\3 =",
        sql,
        count=1,
    )
    sql = re.sub(
        r"(?i)(,\s*)([A-Za-z_][A-Za-z0-9_]*\.)(([A-Za-z_][A-Za-z0-9_]*))\s*=",
        r"\1\3 =",
        sql,
    )
    return sql


def _render_update_ast(
    *,
    with_clause: str,
    prefix: str,
    target_sql: str,
    set_clause: str,
    where_clause: str,
    tail_sql: str,
    dialect: str,
    from_sql: str | None = None,
    returning_sql: str | None = None,
) -> str:
    target = _parse_update_target(target_sql, dialect)
    assignments = _parse_update_assignments(set_clause, dialect)
    where = _parse_update_where(where_clause, dialect)
    order, limit = _parse_update_tail(tail_sql, dialect)
    with_ = _parse_update_with(with_clause, dialect)
    from_ = _parse_update_from(from_sql or "", dialect)
    returning = _parse_update_returning(returning_sql or "", dialect)
    stmt = exp.Update(
        this=target.copy(),
        expressions=assignments,
        from_=from_.copy() if from_ else None,
        where=where.copy() if where else None,
        order=order.copy() if order else None,
        limit=limit.copy() if limit else None,
        with_=with_.copy() if with_ else None,
        returning=returning.copy() if returning else None,
    )
    rendered = _normalize_monetdb_sql(
        _restore_mysql_regexp_like(stmt.sql(dialect=_sqlglot_dialect(dialect))),
        dialect,
    )
    if not dialect_option(dialect, "supports_update_join", True):
        rendered = _strip_qualified_update_set(rendered)
    return _apply_update_prefix(rendered, prefix) + ";"


def _maybe_update_returning(schema: dict, table: str, dialect: str) -> str:
    if not dialect_option(dialect, "supports_dml_returning", False):
        return ""
    if random.random() >= UPDATE_RETURNING_PROB:
        return ""
    cols = list(schema[table]["columns"].keys())
    chosen = [schema[table]["pk"]]
    extras = [col for col in cols if col != chosen[0]]
    if extras and random.random() < 0.35:
        chosen.append(random.choice(extras))
    return ", ".join(chosen)


def _rhs_alias_projection_expr(expr: exp.Expression, alias: str) -> exp.Expression:
    base = expr.this.copy() if isinstance(expr, exp.Alias) else expr.copy()
    return base.as_(alias, copy=False)


def _rhs_query_with_output_aliases(query: exp.Expression, cols: list[str]) -> exp.Select:
    if isinstance(query, exp.Select):
        wrapped = query.copy()
        wrapped.set(
            "expressions",
            [_rhs_alias_projection_expr(expr, col) for expr, col in zip(wrapped.expressions, cols, strict=False)],
        )
        return wrapped

    alias = f"upd_wrap_{rand_str(3, 5)}"
    return sqlglot.select(
        *[exp.column(col, table=alias).as_(col, copy=False) for col in cols]
    ).from_(query.copy().subquery(alias), copy=False)


def _wrap_update_rhs_from_subquery(query: exp.Expression, cols: list[str]) -> exp.Select:
    alias = f"upd_top_{rand_str(3, 5)}"
    return sqlglot.select(
        *[exp.column(col, table=alias) for col in cols]
    ).from_(_rhs_query_with_output_aliases(query, cols).subquery(alias), copy=False)


def _is_intersect_safe_update_rhs_query(query: exp.Select) -> bool:
    return query.args.get("distinct") is not None or query.args.get("group") is not None


def _available_update_rhs_setops(dialect: str, left: exp.Select) -> list[str]:
    options: list[str] = []
    if dialect_option(dialect, "supports_except", True):
        options.append("EXCEPT")
    if dialect_option(dialect, "supports_intersect", True) and _is_intersect_safe_update_rhs_query(left):
        options.append("INTERSECT")
    return options


def _update_rhs_value_expr_ast(
    schema: dict,
    source_table: str,
    source_alias: str,
    type_tag: str,
    dialect: str,
) -> exp.Expression:
    source_col = _pick_col_by_tag(schema, source_table, type_tag)
    if not source_col:
        return _literal_ast(type_tag)

    base = exp.column(source_col, table=source_alias)
    if type_tag == "numeric":
        roll = random.random()
        if roll < 0.4:
            return base
        if roll < 0.7:
            return exp.Abs(this=base.copy())
        op = random.choice(["+", "-"])
        return sqlglot.parse_one(
            f"{_expr_sql(base, dialect)} {op} {random.randint(1, 20)}",
            read=_sqlglot_dialect(dialect),
        )
    if type_tag == "string":
        roll = random.random()
        if roll < 0.35:
            return base
        if roll < 0.6:
            return exp.Upper(this=base.copy())
        if roll < 0.8:
            return exp.Lower(this=base.copy())
        return exp.Concat(expressions=[base.copy(), exp.Literal.string(rand_str(2, 6))])
    if type_tag in {"date", "datetime"}:
        if random.random() < 0.5:
            return base
        unit = "DAY" if type_tag == "date" else random.choice(["DAY", "HOUR"])
        return sqlglot.parse_one(
            _date_shift_sql(_expr_sql(base, dialect), random.randint(1, 12), unit, dialect),
            read=_sqlglot_dialect(dialect),
        )
    return base


def _update_anchor_source_clause(
    schema: dict,
    source_table: str,
    target_table: str,
    table_ids: dict,
    dialect: str,
) -> UpdateQuerySource:
    alias = f"upd_src_{random.randint(100, 999)}"
    from_expression: exp.From
    if random.random() < UPDATE_RHS_SOURCE_FROM_SUBQUERY_PROB:
        inner_alias = f"{alias}_inner"
        projected_cols = _all_projectable_columns(schema, source_table)
        subquery = sqlglot.select(
            *[exp.column(col, table=inner_alias).as_(col, copy=False) for col in projected_cols]
        ).from_(exp.to_table(source_table).as_(inner_alias, copy=False), copy=False)
        from_expression = exp.From(this=subquery.subquery(alias))
    else:
        from_expression = exp.From(this=exp.to_table(source_table).as_(alias, copy=False))
    source = UpdateQuerySource(
        from_expression=from_expression,
        joins=[],
        filters=[],
        limit=None,
        source_table=source_table,
        source_alias=alias,
    )
    if random.random() < 0.45:
        source.filters.append(
            _build_condition_ast(
                schema,
                source_table,
                table_ids,
                dialect,
                allow_subqueries=True,
                target_table=target_table,
                protected_tables={target_table},
                table_alias=alias,
            )
        )
    return source


def _update_join_source_clause(
    schema: dict,
    source_table: str,
    target_table: str,
    table_ids: dict,
    dialect: str,
) -> UpdateQuerySource:
    source = _update_anchor_source_clause(schema, source_table, target_table, table_ids, dialect)
    type_tag = random.choice([t for t in _available_types(schema, source_table) if t in ("numeric", "string", "date", "datetime")] or ["numeric"])
    stmt = _select_stmt_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth=1,
        target_table=target_table,
        protected_tables={target_table},
        column_count=random.randint(1, 2),
    )
    alias = f"upd_join_{random.randint(100, 999)}"
    source.filters.append(
        exp.Exists(
            this=sqlglot.select(exp.Literal.number(1)).from_(_parse_query_ast(stmt, dialect).subquery(alias))
        )
    )
    return source


def _update_exists_source_clause(
    schema: dict,
    source_table: str,
    target_table: str,
    table_ids: dict,
    dialect: str,
) -> UpdateQuerySource:
    source = _update_anchor_source_clause(schema, source_table, target_table, table_ids, dialect)
    type_tag = random.choice([t for t in _available_types(schema, source_table) if t in ("numeric", "string", "date", "datetime")] or ["numeric"])
    stmt = _select_stmt_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth=2,
        target_table=target_table,
        protected_tables={target_table},
        column_count=random.randint(1, 3),
    )
    alias = f"upd_filter_{random.randint(100, 999)}"
    exists_query = sqlglot.select(exp.Literal.number(1)).from_(_parse_query_ast(stmt, dialect).subquery(alias))
    source.filters.append(exp.Exists(this=exists_query))
    return source


def _update_rhs_select_source(
    schema: dict,
    source_table: str,
    target_table: str,
    table_ids: dict,
    dialect: str,
) -> UpdateQuerySource:
    if random.random() >= UPDATE_RHS_COMPLEX_PROB:
        return _update_anchor_source_clause(schema, source_table, target_table, table_ids, dialect)
    roll = random.random()
    if roll < 0.4:
        return _update_join_source_clause(schema, source_table, target_table, table_ids, dialect)
    if roll < 0.75:
        return _update_anchor_source_clause(schema, source_table, target_table, table_ids, dialect)
    return _update_exists_source_clause(schema, source_table, target_table, table_ids, dialect)


def _build_update_rhs_query(
    schema: dict,
    target_table: str,
    source_table: str,
    join_col: str,
    value_type: str,
    table_ids: dict,
    dialect: str,
) -> exp.Select:
    source = _update_rhs_select_source(schema, source_table, target_table, table_ids, dialect)
    source_alias = source.source_alias or source_table
    select_items = [
        exp.column(join_col, table=source_alias).as_("k", copy=False),
        _update_rhs_value_expr_ast(schema, source_table, source_alias, value_type, dialect).as_("v", copy=False),
    ]
    select_ast = sqlglot.select(*select_items)
    if source.from_expression is not None:
        select_ast.set("from_", source.from_expression)
    if source.joins:
        select_ast.set("joins", source.joins)
    for filt in source.filters:
        select_ast = select_ast.where(filt, copy=False)
    if source.limit is not None:
        select_ast = select_ast.limit(source.limit, copy=False)
    if dialect_option(dialect, "supports_select_distinct", True) and random.random() < 0.25:
        select_ast.set("distinct", exp.Distinct())
    return select_ast


def _independent_update_rhs_setop_branch(
    schema: dict,
    target_table: str,
    source_table: str,
    join_col: str,
    value_type: str,
    table_ids: dict,
    dialect: str,
) -> exp.Select:
    return _build_update_rhs_query(
        schema,
        target_table,
        source_table,
        join_col,
        value_type,
        table_ids,
        dialect,
    )


def _wrap_update_rhs_top_level_setop(
    query: exp.Expression,
    cols: list[str],
    dialect: str,
    *,
    schema: dict,
    target_table: str,
    source_table: str,
    join_col: str,
    value_type: str,
    table_ids: dict,
) -> exp.Expression:
    left = _wrap_update_rhs_from_subquery(query, cols)
    right = _rhs_query_with_output_aliases(
        _independent_update_rhs_setop_branch(
            schema,
            target_table,
            source_table,
            join_col,
            value_type,
            table_ids,
            dialect,
        ),
        cols,
    )
    available_setops = _available_update_rhs_setops(dialect, left)
    if not available_setops:
        return query
    setop_kind = random.choice(available_setops)
    if setop_kind == "UNION_ALL":
        return exp.Union(this=left, expression=right, distinct=False)
    if setop_kind == "UNION":
        return exp.Union(this=left, expression=right, distinct=True)
    if setop_kind == "EXCEPT":
        return exp.Except(this=left, expression=right)
    return exp.Intersect(this=left, expression=right)


def _apply_update_rhs_top_level_wrappers(
    query: exp.Expression,
    cols: list[str],
    dialect: str,
    *,
    schema: dict,
    target_table: str,
    source_table: str,
    join_col: str,
    value_type: str,
    table_ids: dict,
    force_setop: bool = False,
) -> exp.Expression:
    if force_setop:
        return _wrap_update_rhs_top_level_setop(
            query,
            cols,
            dialect,
            schema=schema,
            target_table=target_table,
            source_table=source_table,
            join_col=join_col,
            value_type=value_type,
            table_ids=table_ids,
        )

    roll = random.random()
    if roll < UPDATE_RHS_TOPLEVEL_SETOP_PROB:
        return _wrap_update_rhs_top_level_setop(
            query,
            cols,
            dialect,
            schema=schema,
            target_table=target_table,
            source_table=source_table,
            join_col=join_col,
            value_type=value_type,
            table_ids=table_ids,
        )
    if roll < UPDATE_RHS_TOPLEVEL_SETOP_PROB + UPDATE_RHS_TOPLEVEL_FROM_SUBQUERY_PROB:
        return _wrap_update_rhs_from_subquery(query, cols)
    return query


def _pick_table(
    schema: dict,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> str:
    blocked = _normalize_protected_tables(target_table, protected_tables)
    candidates = [table for table in schema.keys() if table not in blocked]
    if candidates:
        return random.choice(candidates)
    return random.choice(list(schema.keys()))


def _pick_numeric_col(schema: dict, table: str) -> str:
    num_cols = columns_by_type_tag(schema, table, "numeric")
    if num_cols:
        return random.choice(num_cols)
    cols = non_pk_columns(schema, table)
    return random.choice(cols) if cols else schema[table]["pk"]


def _pick_aggregate_numeric_col(schema: dict, table: str, dialect: str) -> str:
    num_cols = columns_by_type_tag(schema, table, "numeric")
    if num_cols:
        return random.choice(num_cols)
    cols = non_pk_columns(schema, table)
    return random.choice(cols) if cols else schema[table]["pk"]


def _pick_string_col(schema: dict, table: str) -> str:
    str_cols = columns_by_type(schema, table, STRING_TYPES)
    if str_cols:
        return random.choice(str_cols)
    cols = non_pk_columns(schema, table)
    return random.choice(cols) if cols else schema[table]["pk"]


def _pick_json_col(schema: dict, table: str) -> str | None:
    json_cols = columns_by_type(schema, table, JSON_TYPES)
    if json_cols:
        return random.choice(json_cols)
    return None


def _pick_date_col(schema: dict, table: str) -> str:
    date_cols = columns_by_type(schema, table, DATE_TYPES)
    if date_cols:
        return random.choice(date_cols)
    return None


def _pick_datetime_col(schema: dict, table: str) -> str:
    dt_cols = columns_by_type(schema, table, DATETIME_TYPES)
    if dt_cols:
        return random.choice(dt_cols)
    return None


def _pick_any_col(schema: dict, table: str) -> str:
    cols = [
        col for col in non_pk_columns(schema, table)
        if schema[table]["columns"].get(col) != "geom"
    ]
    return random.choice(cols) if cols else schema[table]["pk"]


def _qualify_col(col: str | None, table: str, table_alias: str | None = None) -> str | None:
    if not col:
        return col
    if "." in col:
        return col
    qualifier = table_alias or table
    return f"{qualifier}.{col}"


def _maybe_update_ignore(dialect: str) -> str:
    if not dialect_option(dialect, "supports_update_ignore", True):
        return ""
    return " IGNORE" if random.random() < 0.5 else ""


def _maybe_update_low_priority(dialect: str) -> str:
    if not dialect_option(dialect, "supports_update_low_priority", True):
        return ""
    return " LOW_PRIORITY" if random.random() < 0.2 else ""


def _update_prefix(dialect: str) -> str:
    return f"UPDATE{_maybe_update_low_priority(dialect)}{_maybe_update_ignore(dialect)}"


def _join_type(dialect: str) -> str:
    return random.choice(get_supported_join_types(dialect))

def _join_type_with_on(dialect: str) -> str:
    supported = [jt for jt in get_supported_join_types(dialect) if jt != "CROSS JOIN"]
    return random.choice(supported or ["JOIN"])


def _join_non_equi_ops(type_tag: str) -> list[str]:
    if type_tag in {"numeric", "string", "date", "datetime", "time"}:
        return ["<>", ">", "<", ">=", "<="]
    return ["<>"]


def _pick_other_col_for_type(
    schema: dict,
    table: str,
    type_tag: str,
    exclude: set[str] | None = None,
) -> str | None:
    excluded = set(exclude or set())
    cols = [
        col
        for col, col_type in schema[table]["columns"].items()
        if col not in excluded
        and col_type != "geom"
        and _type_tag_for_col_type(col_type) == type_tag
    ]
    return random.choice(cols) if cols else None


def _pick_join_on_term(pool: list[str], prefer_subquery: bool = True) -> str:
    if not prefer_subquery:
        return random.choice(pool)
    subquery_pool = [
        cand
        for cand in pool
        if "EXISTS (" in cand.upper() or " IN (SELECT " in cand.upper()
    ]
    if subquery_pool and random.random() < 0.75:
        return random.choice(subquery_pool)
    return random.choice(pool)


def _compose_join_on_sql(candidates: list[str], required_term: str | None = None) -> str:
    pool = [cand for cand in candidates if cand]
    required = (required_term or "").strip()
    if required and required not in pool:
        pool.insert(0, required)
    if not pool:
        return "1 = 1"

    terms: list[str] = []
    remaining = list(pool)
    if required:
        terms.append(required)
        remaining = [cand for cand in remaining if cand != required]
    else:
        first = _pick_join_on_term(remaining)
        terms.append(first)
        remaining = [cand for cand in remaining if cand != first]

    if remaining and random.random() < JOIN_ON_EXTRA_TERM1_PROB:
        extra = _pick_join_on_term(remaining)
        terms.append(extra)
        remaining = [cand for cand in remaining if cand != extra]
    if remaining and random.random() < JOIN_ON_EXTRA_TERM2_PROB:
        terms.append(_pick_join_on_term(remaining))
    return " AND ".join(f"({term})" for term in terms)


def _partition_clause(schema: dict, table: str, dialect: str) -> str:
    if not dialect_option(dialect, "supports_dml_partition_clause", False):
        return ""
    names = table_partition_names(schema, table)
    if not names or random.random() >= 0.3:
        return ""
    picked = random.sample(names, k=random.randint(1, min(2, len(names))))
    return f" PARTITION ({', '.join(picked)})"


def _requirement_columns(schema: dict, table: str) -> list[str]:
    return [
        col
        for col, col_type in schema[table]["columns"].items()
        if col_type not in {"geom", "json"}
    ]


def _requirement_type_family(col_type: str) -> str:
    return _type_tag_for_col_type(col_type)


def _requirement_source_candidates(schema: dict, col_type: str) -> list[tuple[str, str]]:
    family = _requirement_type_family(col_type)
    candidates: list[tuple[str, str]] = []
    for table_name, info in schema.items():
        for col_name, other_type in info["columns"].items():
            if other_type in {"geom", "json"}:
                continue
            if _requirement_type_family(other_type) != family:
                continue
            candidates.append((table_name, col_name))
    return candidates


def _requirement_subquery_predicate(
    schema: dict,
    table: str,
    col: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> str | None:
    col_type = schema[table]["columns"][col]
    sources = _requirement_source_candidates(schema, col_type)
    if not sources:
        return None

    same_table = [(src_table, src_col) for src_table, src_col in sources if src_table == table]
    source_table, source_col = random.choice(same_table or sources)
    wrapped_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        source_table,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_req_sub",
        required_cols=_predicate_visible_subquery_cols(schema, source_table),
    )
    sub_cond = _build_condition(
        schema,
        source_table,
        table_ids,
        dialect,
        allow_subqueries=False,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=qualifier,
    )
    outer_ref = f"{table}.{col}"
    source_ref = f"{qualifier}.{source_col}"
    if random.random() < 0.5:
        return f"{outer_ref} IN (SELECT {source_ref} FROM {wrapped_ref} WHERE {sub_cond})"
    return f"EXISTS (SELECT {source_ref} FROM {wrapped_ref} WHERE {source_ref} = {outer_ref} AND {sub_cond})"


def _requirement_predicate(
    schema: dict,
    table: str,
    col: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> str:
    col_ref = f"{table}.{col}"
    if col == schema[table]["pk"]:
        return f"{col_ref} = {random.choice(table_ids[table])}"

    col_type = schema[table]["columns"][col]
    type_tag = _type_tag_for_col_type(col_type)
    if random.random() < 0.3:
        subquery_pred = _requirement_subquery_predicate(
            schema,
            table,
            col,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
        )
        if subquery_pred:
            return subquery_pred

    value = gen_value(col_type, dialect, column_meta(schema, table, col))
    if value.upper() == "NULL":
        return f"{col_ref} IS NULL"

    if col_type == "bool":
        return f"{col_ref} = {value}"

    if type_tag == "string":
        if random.random() < 0.25:
            return f"{col_ref} LIKE {sql_str('%' + rand_str(1, 3) + '%')}"
        return f"{col_ref} = {value}"

    if type_tag in {"numeric", "date", "datetime"} and random.random() < 0.25:
        op = random.choice([">", "<", ">=", "<="])
        return f"{col_ref} {op} {value}"

    return f"{col_ref} = {value}"


def _row_requirement_clause(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> str:
    cols = _requirement_columns(schema, table)
    if not cols:
        pk = schema[table]["pk"]
        return f"{table}.{pk} = {random.choice(table_ids[table])}"

    non_pk = [col for col in cols if col != schema[table]["pk"]]
    chosen: list[str] = []
    target_count = random.randint(1, min(2, len(cols)))

    if non_pk and random.random() < 0.7:
        chosen.append(random.choice(non_pk))
    if len(chosen) < target_count:
        remaining = [col for col in cols if col not in chosen]
        extra = random.sample(remaining, k=min(target_count - len(chosen), len(remaining)))
        chosen.extend(extra)

    predicates = [
        _requirement_predicate(
            schema,
            table,
            col,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
        )
        for col in chosen
    ]
    return " AND ".join(predicates)


def _normalize_protected_tables(
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> set[str]:
    tables = set(protected_tables or set())
    if target_table:
        tables.add(target_table)
    return tables


def _is_protected_table(
    table: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> bool:
    return table in _normalize_protected_tables(target_table, protected_tables)


def _projected_columns(
    schema: dict,
    table: str,
    required_cols: list[str] | None = None,
) -> list[str]:
    cols = []
    seen = set()
    source = required_cols or list(schema[table]["columns"].keys())
    for col in source:
        if not col or col in seen or col not in schema[table]["columns"]:
            continue
        if schema[table]["columns"][col] == "geom":
            continue
        cols.append(col)
        seen.add(col)
    if cols:
        return cols
    pk = schema[table]["pk"]
    return [pk]


def _all_projectable_columns(schema: dict, table: str) -> list[str]:
    return _projected_columns(schema, table, required_cols=list(schema[table]["columns"].keys()))


def _predicate_visible_subquery_cols(schema: dict, table: str) -> list[str]:
    # Wrapped table aliases reused by the generic predicate builder must expose
    # every projectable column, otherwise later predicates can reference
    # columns that were not projected into the derived table.
    return _all_projectable_columns(schema, table)


def _exists_projection_col(schema: dict, table: str) -> str:
    cols = _all_projectable_columns(schema, table)
    return cols[0] if cols else schema[table]["pk"]


def _setop_input_aliases(column_count: int) -> list[str]:
    return ["k"] + [f"k{i}" for i in range(2, column_count + 1)]


def _setop_visible_aliases(column_count: int, prefix: str = "c") -> list[str]:
    stem = f"{prefix}_{rand_str(3, 5)}"
    return [f"{stem}_{i}" for i in range(1, column_count + 1)]


def _wrap_setop_stmt_with_aliases(
    stmt: str,
    column_count: int,
    *,
    output_aliases: list[str] | None = None,
    alias_prefix: str = "setop",
) -> tuple[str, list[str]]:
    inner_alias = f"{alias_prefix}_{rand_str(3, 5)}"
    output_cols = output_aliases or _setop_visible_aliases(column_count, prefix=alias_prefix)
    input_cols = _setop_input_aliases(column_count)
    projection = ", ".join(
        f"{inner_alias}.{src} AS {dst}" for src, dst in zip(input_cols, output_cols, strict=False)
    )
    return f"(SELECT {projection} FROM ({stmt}) AS {inner_alias})", output_cols


def _double_wrapped_table_ref(
    schema: dict,
    table: str,
    alias: str = "_sub",
    required_cols: list[str] | None = None,
) -> str:
    inner_alias = f"{alias}_inner"
    projected_cols = _projected_columns(schema, table, required_cols=required_cols)
    inner_projection = ", ".join(projected_cols)
    outer_projection = ", ".join(f"{inner_alias}.{col} AS {col}" for col in projected_cols)
    return f"(SELECT {outer_projection} FROM (SELECT {inner_projection} FROM {table}) AS {inner_alias}) AS {alias}"


def _safe_table_ref_for_subquery(
    schema: dict,
    table: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    alias: str = "_sub",
    required_cols: list[str] | None = None,
) -> tuple[str, str]:
    if _is_protected_table(table, target_table=target_table, protected_tables=protected_tables):
        return _double_wrapped_table_ref(schema, table, alias, required_cols=required_cols), alias
    if random.random() < SUBQUERY_DERIVED_FROM_PROB:
        return _double_wrapped_table_ref(schema, table, alias, required_cols=required_cols), alias
    return table, table


def _maybe_with_clause(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> tuple[str, str, list[ConditionColSpec]]:
    if not dialect_option(dialect, "supports_dml_with", True):
        return "", "", []
    if random.random() >= 0.2:
        return "", "", []

    available_types = _available_types(schema, table)
    if not available_types:
        return "", "", []
    type_tag = random.choice(available_types)
    target_col = _pick_col_by_tag(schema, table, type_tag)
    if not target_col:
        return "", "", []
    cte_cols = random.randint(2, 4)

    stmt = _select_stmt_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth=2,
        target_table=table,
        protected_tables=protected_tables,
        column_count=cte_cols,
    )

    cte_name = f"cte_{rand_str(3, 6)}"
    wrapped_stmt, visible_cols = _wrap_setop_stmt_with_aliases(stmt, cte_cols, alias_prefix="cte")
    chosen_col = random.choice(visible_cols)
    cte_sql = f"WITH {cte_name} AS {wrapped_stmt} "
    extra_pred = random.choice(
        [
            f"{table}.{target_col} IN (SELECT {chosen_col} FROM {cte_name})",
            f"EXISTS (SELECT {chosen_col} FROM {cte_name} WHERE {cte_name}.{chosen_col} = {table}.{target_col})",
        ]
    )
    cte_specs = [(cte_name, col, type_tag) for col in visible_cols]
    return cte_sql, extra_pred, cte_specs


def _type_tag_for_col_type(col_type: str) -> str:
    return column_type_tag(col_type)


def _pick_col_by_tag(schema: dict, table: str, type_tag: str) -> str | None:
    if type_tag in {"numeric", "string", "date", "datetime", "json", "boolean", "binary"}:
        cols = columns_by_type_tag(schema, table, type_tag)
    else:
        cols = non_pk_columns(schema, table)
    return random.choice(cols) if cols else None


def _key_columns(schema: dict, table: str) -> list[str]:
    cols: list[str] = []
    pk = schema[table]["pk"]
    if pk in schema[table]["columns"]:
        cols.append(pk)
    for fk in schema[table]["foreign_keys"]:
        col = fk["column"]
        if col in schema[table]["columns"] and col not in cols:
            cols.append(col)
    return cols


def _pick_key_col_by_tag(schema: dict, table: str, type_tag: str) -> str | None:
    pk = schema[table]["pk"]
    if pk not in schema[table]["columns"]:
        return None
    if _type_tag_for_col_type(schema[table]["columns"][pk]) == type_tag:
        return pk
    return None


def _cols_by_tag(schema: dict, table: str, type_tag: str) -> list[str]:
    if type_tag in {"numeric", "string", "date", "datetime", "json", "boolean", "binary"}:
        return list(columns_by_type_tag(schema, table, type_tag))
    return list(non_pk_columns(schema, table))


def _condition_available_types(
    schema: dict,
    table: str,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> list[str]:
    if isinstance(table_alias, list) and extra_col_specs is None:
        extra_col_specs = table_alias
        table_alias = None
    types = {type_tag for _, _, type_tag in _condition_col_specs(schema, table, table_alias, extra_col_specs)}
    return list(types) or ["numeric"]


def _condition_col_specs(
    schema: dict,
    table: str,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> list[ConditionColSpec]:
    if isinstance(table_alias, list) and extra_col_specs is None:
        extra_col_specs = table_alias
        table_alias = None
    alias = table_alias or table
    specs: list[ConditionColSpec] = []
    for type_tag in _available_types(schema, table):
        specs.extend((alias, col, type_tag) for col in _cols_by_tag(schema, table, type_tag))
    if extra_col_specs:
        specs.extend(extra_col_specs)

    deduped: list[ConditionColSpec] = []
    seen: set[ConditionColSpec] = set()
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        deduped.append(spec)
    return deduped


def _merge_condition_col_specs(*groups: list[ConditionColSpec] | None) -> list[ConditionColSpec]:
    merged: list[ConditionColSpec] = []
    seen: set[ConditionColSpec] = set()
    for group in groups:
        if not group:
            continue
        for spec in group:
            if spec in seen:
                continue
            seen.add(spec)
            merged.append(spec)
    return merged


def _joined_update_where_specs(
    cte_specs: list[ConditionColSpec],
    rhs_specs: list[ConditionColSpec],
    *,
    target_only: bool = False,
) -> list[ConditionColSpec]:
    if target_only:
        return []
    return _merge_condition_col_specs(cte_specs, rhs_specs)


def _pick_condition_col_ast(
    schema: dict,
    table: str,
    type_tag: str,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression | None:
    candidates = [
        spec for spec in _condition_col_specs(schema, table, table_alias, extra_col_specs)
        if spec[2] == type_tag
    ]
    if not candidates:
        return None
    qualifier, col, _ = random.choice(candidates)
    if qualifier.startswith("cte_"):
        return sqlglot.parse_one(f"(SELECT {qualifier}.{col} FROM {qualifier} ORDER BY {qualifier}.{col} DESC LIMIT 1)")
    return exp.column(col, table=qualifier)


def _compatible_column_pairs(
    schema: dict,
    left_table: str,
    right_table: str,
    allowed_type_tags: set[str] | None = None,
    left_exclude: set[str] | None = None,
    right_exclude: set[str] | None = None,
) -> list[tuple[str, str]]:
    left_blocked = set(left_exclude or set())
    right_blocked = set(right_exclude or set())
    pairs: list[tuple[str, str]] = []
    for left_col, left_type in schema[left_table]["columns"].items():
        if left_col in left_blocked or left_type in {"geom", "json"}:
            continue
        left_tag = _type_tag_for_col_type(left_type)
        if left_tag in {"binary", "boolean", "json", "geom"}:
            continue
        if allowed_type_tags and left_tag not in allowed_type_tags:
            continue
        for right_col, right_type in schema[right_table]["columns"].items():
            if right_col in right_blocked or right_type in {"geom", "json"}:
                continue
            right_tag = _type_tag_for_col_type(right_type)
            if right_tag in {"binary", "boolean", "json", "geom"}:
                continue
            if right_tag != left_tag:
                continue
            pairs.append((left_col, right_col))
    return pairs


def _pick_source_column_for_type(
    schema: dict,
    type_tag: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    blocked = _normalize_protected_tables(target_table, protected_tables)
    for table_name in schema:
        if table_name in blocked:
            continue
        col = _pick_col_by_tag(schema, table_name, type_tag)
        if col:
            candidates.append((table_name, col))
    if not candidates:
        for table_name in schema:
            col = _pick_col_by_tag(schema, table_name, type_tag)
            if col:
                candidates.append((table_name, col))
    return random.choice(candidates) if candidates else None


def _pick_related_table(
    schema: dict,
    table: str,
    allowed_type_tags: set[str] | None = None,
    left_exclude: set[str] | None = None,
    right_exclude: set[str] | None = None,
    blocked_tables: set[str] | None = None,
) -> tuple[str, str, str] | None:
    candidates: list[tuple[str, str, str]] = []
    blocked = set(blocked_tables or set())

    for fk in schema[table]["foreign_keys"]:
        fk_type = schema[table]["columns"].get(fk["column"])
        ref_type = schema[fk["ref_table"]]["columns"].get(fk["ref_column"])
        if not fk_type or not ref_type:
            continue
        fk_tag = _type_tag_for_col_type(fk_type)
        ref_tag = _type_tag_for_col_type(ref_type)
        if fk_tag != ref_tag:
            continue
        if allowed_type_tags and fk_tag not in allowed_type_tags:
            continue
        if fk["column"] in set(left_exclude or set()) or fk["ref_column"] in set(right_exclude or set()):
            continue
        if fk["ref_table"] in blocked:
            continue
        candidates.append((fk["ref_table"], fk["column"], fk["ref_column"]))

    for other, info in schema.items():
        if other == table:
            continue
        if other in blocked:
            continue
        for fk in info["foreign_keys"]:
            if fk["ref_table"] != table:
                continue
            ref_type = schema[table]["columns"].get(fk["ref_column"])
            other_type = schema[other]["columns"].get(fk["column"])
            if not ref_type or not other_type:
                continue
            ref_tag = _type_tag_for_col_type(ref_type)
            other_tag = _type_tag_for_col_type(other_type)
            if ref_tag != other_tag:
                continue
            if allowed_type_tags and ref_tag not in allowed_type_tags:
                continue
            if fk["ref_column"] in set(left_exclude or set()) or fk["column"] in set(right_exclude or set()):
                continue
            candidates.append((other, fk["ref_column"], fk["column"]))

    for other in schema:
        if other == table:
            continue
        if other in blocked:
            continue
        for left_col, right_col in _compatible_column_pairs(
            schema,
            table,
            other,
            allowed_type_tags=allowed_type_tags,
            left_exclude=left_exclude,
            right_exclude=right_exclude,
        ):
            candidates.append((other, left_col, right_col))

    return random.choice(candidates) if candidates else None


def _pick_fk_to_pk_related_table(
    schema: dict,
    table: str,
    allowed_type_tags: set[str] | None = None,
    left_exclude: set[str] | None = None,
    right_exclude: set[str] | None = None,
    blocked_tables: set[str] | None = None,
) -> tuple[str, str, str] | None:
    candidates: list[tuple[str, str, str]] = []
    blocked = set(blocked_tables or set())

    for fk in schema[table]["foreign_keys"]:
        ref_table = fk["ref_table"]
        ref_column = fk["ref_column"]
        if ref_table in blocked:
            continue
        if ref_column != schema[ref_table]["pk"]:
            continue
        fk_type = schema[table]["columns"].get(fk["column"])
        ref_type = schema[ref_table]["columns"].get(ref_column)
        if not fk_type or not ref_type:
            continue
        fk_tag = _type_tag_for_col_type(fk_type)
        ref_tag = _type_tag_for_col_type(ref_type)
        if fk_tag != ref_tag:
            continue
        if allowed_type_tags and fk_tag not in allowed_type_tags:
            continue
        if fk["column"] in set(left_exclude or set()) or ref_column in set(right_exclude or set()):
            continue
        candidates.append((ref_table, fk["column"], ref_column))

    return random.choice(candidates) if candidates else None



def _available_types(schema: dict, table: str) -> list:
    return available_general_type_tags(schema, table)


def _pick_col_for_type(schema: dict, table: str, type_tag: str) -> str:
    if type_tag == "string":
        return _pick_string_col(schema, table)
    if type_tag == "numeric":
        return _pick_numeric_col(schema, table)
    if type_tag == "date":
        return _pick_date_col(schema, table)
    if type_tag == "datetime":
        return _pick_datetime_col(schema, table)
    if type_tag == "time":
        return None
    if type_tag == "json":
        return _pick_json_col(schema, table) or _pick_any_col(schema, table)
    if type_tag == "boolean":
        cols = columns_by_type_tag(schema, table, "boolean")
        return random.choice(cols) if cols else _pick_any_col(schema, table)
    if type_tag == "binary":
        cols = columns_by_type_tag(schema, table, "binary")
        return random.choice(cols) if cols else _pick_any_col(schema, table)
    return _pick_any_col(schema, table)


def _literal_for_type(type_tag: str) -> str:
    if type_tag == "string":
        return sql_str(rand_str(3, 8))
    if type_tag == "numeric":
        return str(random.randint(1, 1000))
    if type_tag == "boolean":
        return str(random.randint(0, 1))
    if type_tag == "date":
        return f"CAST({sql_str(rand_date())} AS DATE)"
    if type_tag == "datetime":
        return f"CAST({sql_str(rand_datetime())} AS DATETIME)"
    if type_tag == "time":
        return f"CAST({sql_str(rand_time())} AS TIME)"
    if type_tag == "json":
        return sql_str('{"k": "' + rand_str(3, 6) + '"}')
    return sql_str(rand_str(3, 8))


def _pattern_literal(op: str) -> str:
    base = rand_str(3, 6)
    if op == "LIKE":
        variants = [
            base,
            f"{base}%",
            f"%{base}",
            f"%{base}%",
            f"{base}_{rand_str(1, 2)}",
        ]
    else:
        variants = [
            base,
            f"{base}.*",
            f".*{base}",
            f".*{base}.*",
            f"{base}[a-z]",
        ]
    return sql_str(random.choice(variants))


def _expr_sql(expr: exp.Expression, dialect: str) -> str:
    return _normalize_monetdb_sql(
        _restore_mysql_regexp_like(expr.sql(dialect=_sqlglot_dialect(dialect))),
        dialect,
    )


def _query_sql(query: exp.Expression, dialect: str) -> str:
    return _normalize_monetdb_sql(
        _restore_mysql_regexp_like(query.sql(dialect=_sqlglot_dialect(dialect))),
        dialect,
    )


def _restore_mysql_regexp_like(sql: str) -> str:
    marker = "REGEXP_LIKE("
    upper_sql = sql.upper()
    out = []
    i = 0
    n = len(sql)

    while i < n:
        pos = upper_sql.find(marker, i)
        if pos < 0:
            out.append(sql[i:])
            break
        out.append(sql[i:pos])

        j = pos + len(marker)
        depth = 1
        in_single = False
        in_double = False
        in_backtick = False
        while j < n and depth > 0:
            ch = sql[j]
            if ch == "'" and not in_double and not in_backtick:
                in_single = not in_single
            elif ch == '"' and not in_single and not in_backtick:
                in_double = not in_double
            elif ch == "`" and not in_single and not in_double:
                in_backtick = not in_backtick
            elif not in_single and not in_double and not in_backtick:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            j += 1

        if depth != 0:
            out.append(sql[pos:j])
            i = j
            continue

        inner = sql[pos + len(marker): j - 1]
        depth2 = 0
        in_single = False
        in_double = False
        in_backtick = False
        comma = -1
        for idx, ch in enumerate(inner):
            if ch == "'" and not in_double and not in_backtick:
                in_single = not in_single
            elif ch == '"' and not in_single and not in_backtick:
                in_double = not in_double
            elif ch == "`" and not in_single and not in_double:
                in_backtick = not in_backtick
            elif not in_single and not in_double and not in_backtick:
                if ch == "(":
                    depth2 += 1
                elif ch == ")":
                    depth2 -= 1
                elif ch == "," and depth2 == 0:
                    comma = idx
                    break

        if comma < 0:
            out.append(sql[pos:j])
        else:
            left = inner[:comma].strip()
            right = inner[comma + 1:].strip()
            out.append(f"{left} REGEXP {right}")
        i = j

    return "".join(out)


def _literal_ast(type_tag: str) -> exp.Expression:
    if type_tag == "numeric":
        return exp.Literal.number(random.randint(1, 1000))
    if type_tag == "boolean":
        return exp.Literal.number(random.randint(0, 1))
    if type_tag == "string":
        return exp.Literal.string(rand_str(3, 8))
    if type_tag == "date":
        return exp.Cast(this=exp.Literal.string(rand_date()), to=exp.DataType.build("DATE"))
    if type_tag == "datetime":
        return exp.Cast(this=exp.Literal.string(rand_datetime()), to=exp.DataType.build("DATETIME"))
    if type_tag == "time":
        return exp.Cast(this=exp.Literal.string(rand_time()), to=exp.DataType.build("TIME"))
    if type_tag == "json":
        return exp.Literal.string('{"k": "' + rand_str(3, 6) + '"}')
    return exp.Literal.string(rand_str(3, 8))


def _pattern_literal_ast(op: str) -> exp.Expression:
    base = rand_str(3, 6)
    if op == "LIKE":
        variants = [
            base,
            f"{base}%",
            f"%{base}",
            f"%{base}%",
            f"{base}_{rand_str(1, 2)}",
        ]
    else:
        variants = [
            base,
            f"{base}.*",
            f".*{base}",
            f".*{base}.*",
            f"{base}[a-z]",
        ]
    return exp.Literal.string(random.choice(variants))


def _qualify_col_ast(col: str | None, table: str, table_alias: str | None = None) -> exp.Expression | None:
    if not col:
        return None
    if "." in col:
        table_name, col_name = col.split(".", 1)
        return exp.column(col_name, table=table_name)
    return exp.column(col, table=table_alias or table)


def _data_type_ast(type_sql: str, dialect: str) -> exp.DataType:
    parsed = sqlglot.parse_one(
        normalize_sqlglot_input(type_sql, dialect),
        into=exp.DataType,
        read=_sqlglot_dialect(dialect),
    )
    if not isinstance(parsed, exp.DataType):
        raise ValueError(f"Expected data type, got {type(parsed).__name__}: {type_sql}")
    return parsed


def _build_args_ast(
    schema: dict,
    table: str,
    dialect: str,
    spec: dict,
    table_alias: str | None = None,
) -> list[exp.Expression]:
    count = _variadic_arg_count(spec)
    arg_types = spec.get("arg_types") or ["any"]
    args: list[exp.Expression] = []
    for i in range(count):
        type_tag = arg_types[i] if i < len(arg_types) else arg_types[-1]
        if type_tag == "any":
            col = _pick_any_col(schema, table)
            node = _qualify_col_ast(col, table, table_alias)
            args.append(node if node is not None else _literal_ast(random.choice(["string", "numeric"])))
        else:
            col = _pick_aggregate_numeric_col(schema, table, dialect) if type_tag == "numeric" else _pick_col_for_type(schema, table, type_tag)
            if col and random.random() < 0.7:
                node = _qualify_col_ast(col, table, table_alias)
                args.append(node if node is not None else _literal_ast(type_tag))
            else:
                args.append(_literal_ast(type_tag))
    return args


def _simple_expr_ast(schema: dict, table: str, type_tag: str, table_alias: str | None = None) -> exp.Expression:
    if type_tag == "numeric":
        col = _pick_numeric_col(schema, table)
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("numeric")
    if type_tag == "boolean":
        col = _pick_col_by_tag(schema, table, "boolean")
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("boolean")
    if type_tag == "string":
        col = _pick_string_col(schema, table)
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("string")
    if type_tag == "date":
        col = _pick_date_col(schema, table)
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("date")
    if type_tag == "datetime":
        col = _pick_datetime_col(schema, table)
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("datetime")
    if type_tag == "time":
        return _literal_ast("time")
    if type_tag == "json":
        col = _pick_json_col(schema, table)
        if col and random.random() < 0.7:
            node = _qualify_col_ast(col, table, table_alias)
            if node is not None:
                return node
        return _literal_ast("json")
    col = _pick_any_col(schema, table)
    if col and random.random() < 0.7:
        node = _qualify_col_ast(col, table, table_alias)
        if node is not None:
            return node
    return _literal_ast(random.choice(["string", "numeric"]))


def _boolean_expr_ast(schema: dict, table: str, table_alias: str | None = None) -> exp.Expression:
    col = _pick_col_by_tag(schema, table, "boolean")
    if col and random.random() < 0.45:
        node = _qualify_col_ast(col, table, table_alias)
        if node is not None:
            return node

    roll = random.random()
    if roll < 0.35:
        return _literal_ast("boolean")

    cmp_left = _simple_expr_ast(schema, table, "numeric", table_alias=table_alias)
    cmp_right = _simple_expr_ast(schema, table, "numeric", table_alias=table_alias)
    cmp_op = random.choice(
        [
            exp.EQ,
            exp.NEQ,
            exp.GT,
            exp.LT,
            exp.GTE,
            exp.LTE,
        ]
    )
    return exp.Case(
        ifs=[
            exp.If(
                this=cmp_op(this=cmp_left, expression=cmp_right),
                true=exp.Literal.number(1),
            )
        ],
        default=exp.Literal.number(0),
    )


def _string_expr_ast(schema: dict, table: str, table_alias: str | None = None) -> exp.Expression:
    col = _pick_string_col(schema, table)
    if col and random.random() < 0.8:
        node = _qualify_col_ast(col, table, table_alias)
        if node is not None:
            return node
    return exp.Literal.string(rand_str(3, 8))


def _any_expr_ast(
    schema: dict,
    table: str,
    dialect: str,
    table_alias: str | None = None,
) -> exp.Expression:
    col = _pick_any_col(schema, table)
    if col and random.random() < 0.7:
        node = _qualify_col_ast(col, table, table_alias)
        if node is not None:
            return node
    return _literal_ast(random.choice(["string", "numeric"]))


def _function_expr_ast(
    schema: dict,
    table: str,
    dialect: str,
    desired_type: str,
    allow_aggregate: bool = True,
    table_alias: str | None = None,
) -> exp.Expression:
    category = None if allow_aggregate else "scalar"
    spec = _pick_function_spec(desired_type, dialect, category=category)
    name = spec["name"]

    if not allow_aggregate and spec.get("category") == "aggregate":
        return _simple_expr_ast(schema, table, desired_type, table_alias=table_alias)

    if name in ("MIN", "MAX"):
        arg = _simple_expr_ast(schema, table, desired_type, table_alias=table_alias)
        klass = exp.Min if name == "MIN" else exp.Max
        return klass(this=arg)

    if name in ("COUNT_DISTINCT", "SUM_DISTINCT"):
        arg = _simple_expr_ast(schema, table, "any", table_alias=table_alias)
        distinct = exp.Distinct(expressions=[arg])
        return exp.Count(this=distinct) if name == "COUNT_DISTINCT" else exp.Sum(this=distinct)

    if name == "CAST":
        cast_type = random.choice(["CHAR(10)", "SIGNED", "DECIMAL(10,2)", "DATE", "DATETIME"])
        source_type = "string" if cast_type.startswith("CHAR") else ("datetime" if cast_type in ("DATE", "DATETIME") else "numeric")
        expr = _simple_expr_ast(schema, table, source_type, table_alias=table_alias)
        return exp.Cast(this=expr, to=_data_type_ast(cast_type, dialect))

    if name == "CONVERT":
        cast_type = random.choice(["CHAR(10)", "SIGNED", "DECIMAL(10,2)", "DATE", "DATETIME"])
        source_type = "string" if cast_type.startswith("CHAR") else ("datetime" if cast_type in ("DATE", "DATETIME") else "numeric")
        expr = _simple_expr_ast(schema, table, source_type, table_alias=table_alias)
        # sqlglot renders exp.Convert as CAST(type AS expr) in mysql dialect,
        # which produces invalid SQL. Use CAST syntax instead.
        return exp.Cast(this=expr, to=_data_type_ast(cast_type, dialect))

    if name == "DIV":
        left = _simple_expr_ast(schema, table, "numeric", table_alias=table_alias)
        right = _simple_expr_ast(schema, table, "numeric", table_alias=table_alias)
        return sqlglot.parse_one(f"{_expr_sql(left, dialect)} DIV {_expr_sql(right, dialect)}", read=_sqlglot_dialect(dialect))

    if name == "SUBSTRING":
        return exp.Substring(
            this=_string_expr_ast(schema, table, table_alias=table_alias),
            start=exp.Literal.number(random.randint(1, 10)),
            length=exp.Literal.number(random.randint(1, 5)),
        )

    if name == "SUBSTRING_INDEX":
        return exp.SubstringIndex(
            this=_string_expr_ast(schema, table, table_alias=table_alias),
            delimiter=exp.Literal.string(random.choice([",", "-", "_"])),
            count=exp.Literal.number(random.randint(1, 5)),
        )

    if name == "TRIM":
        src = _string_expr_ast(schema, table, table_alias=table_alias)
        if random.random() < 0.5:
            return exp.Trim(this=src)
        return exp.Trim(
            this=src,
            expression=exp.Literal.string(random.choice([" ", "-", "_"])),
            position=exp.var(random.choice(["BOTH", "LEADING", "TRAILING"])),
        )

    if name == "CONCAT":
        arg_count = 2 if dialect == "monetdb" else random.randint(2, 5)
        args = [_string_expr_ast(schema, table, table_alias=table_alias) for _ in range(arg_count)]
        return exp.Anonymous(this="CONCAT", expressions=args)

    if name == "CONCAT_WS":
        args = [exp.Literal.string(random.choice([",", "-", "_"]))] + [
            _string_expr_ast(schema, table, table_alias=table_alias) for _ in range(random.randint(2, 5))
        ]
        return exp.Anonymous(this="CONCAT_WS", expressions=args)

    if name == "DATE_FORMAT":
        dt = _pick_datetime_col(schema, table) or _pick_date_col(schema, table)
        dt_expr = _qualify_col_ast(dt, table, table_alias) or exp.Literal.string(rand_datetime())
        fmt_choices = ["%Y-%m-%d", "%Y-%m-%d %H:%i:%s", "%H:%i:%s"]
        if dialect == "duckdb":
            fmt_choices = [choice.replace("%i", "%M").replace("%s", "%S") for choice in fmt_choices]
        fmt = exp.Literal.string(random.choice(fmt_choices))
        if dialect == "duckdb":
            return exp.TimeToStr(this=dt_expr, format=fmt)
        return exp.Anonymous(this="DATE_FORMAT", expressions=[dt_expr, fmt])

    if name == "STR_TO_DATE":
        fmt_choices = ["%Y-%m-%d", "%Y-%m-%d %H:%i:%s"]
        if dialect == "duckdb":
            fmt_choices = [choice.replace("%i", "%M").replace("%s", "%S") for choice in fmt_choices]
        fmt = random.choice(fmt_choices)
        value = rand_datetime() if "%H" in fmt else rand_date()
        if dialect == "duckdb":
            return exp.StrToTime(this=exp.Literal.string(value), format=exp.Literal.string(fmt))
        return exp.Anonymous(this="STR_TO_DATE", expressions=[exp.Literal.string(value), exp.Literal.string(fmt)])

    if name == "DATEDIFF":
        d1 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or _qualify_col_ast(_pick_date_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        d2 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or _qualify_col_ast(_pick_date_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        return exp.DateDiff(this=d1, expression=d2)

    if name == "TIMEDIFF":
        d1 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        d2 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        if dialect == "duckdb":
            return exp.Cast(
                this=exp.Sub(
                    this=exp.Cast(this=d1, to=exp.DataType.build("TIMESTAMP")),
                    expression=exp.Cast(this=d2, to=exp.DataType.build("TIMESTAMP")),
                ),
                to=exp.DataType.build("TEXT"),
            )
        return exp.Anonymous(this="TIMEDIFF", expressions=[d1, d2])

    if name == "TIMESTAMPDIFF":
        unit = exp.var(random.choice(["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]))
        d1 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or _qualify_col_ast(_pick_date_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        d2 = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or _qualify_col_ast(_pick_date_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        return exp.TimestampDiff(this=d2, expression=d1, unit=unit)

    if name == "TIME":
        dt = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        if dialect in {"duckdb", "monetdb"}:
            return exp.Cast(this=dt, to=exp.DataType.build("TIME"))
        return exp.Anonymous(this="TIME", expressions=[dt])

    if name == "DATE":
        dt = _qualify_col_ast(_pick_datetime_col(schema, table), table, table_alias) or _qualify_col_ast(_pick_date_col(schema, table), table, table_alias) or exp.Literal.string(rand_datetime())
        if dialect in {"duckdb", "monetdb"}:
            return exp.Cast(this=dt, to=exp.DataType.build("DATE"))
        return exp.Anonymous(this="DATE", expressions=[dt])

    if name in ("DATE_ADD", "DATE_SUB", "ADDDATE", "SUBDATE"):
        dt = _pick_datetime_col(schema, table) or _pick_date_col(schema, table)
        dt_expr = _qualify_col_ast(dt, table, table_alias) or exp.Literal.string(rand_datetime())
        unit = exp.var(random.choice(["DAY", "MONTH", "YEAR", "HOUR", "MINUTE", "SECOND"]))
        num = exp.Literal.number(random.randint(1, 30))
        klass = exp.DateAdd if name in ("DATE_ADD", "ADDDATE") else exp.DateSub
        return klass(this=dt_expr, expression=num, unit=unit)

    if name == "JSON_VALUE":
        jcol = _qualify_col_ast(_pick_json_col(schema, table), table, table_alias) or _literal_ast("json")
        return exp.Anonymous(this="JSON_VALUE", expressions=[jcol, exp.Literal.string(random.choice(["$.k", "$.key", "$.a", "$.b", "$.items[0]"]))])

    if name in ("JSON_EXTRACT", "JSON_REMOVE"):
        jcol = _qualify_col_ast(_pick_json_col(schema, table), table, table_alias) or _literal_ast("json")
        paths = [exp.Literal.string(random.choice(["$.k", "$.key", "$.a", "$.b", "$.items[0]"])) for _ in range(random.randint(1, 4))]
        return exp.Anonymous(this=name, expressions=[jcol] + paths)

    if name in ("JSON_SET", "JSON_INSERT", "JSON_REPLACE"):
        jcol = _qualify_col_ast(_pick_json_col(schema, table), table, table_alias) or _literal_ast("json")
        parts: list[exp.Expression] = [jcol]
        for _ in range(random.randint(1, 3)):
            parts.append(exp.Literal.string(random.choice(["$.k", "$.key", "$.a", "$.b", "$.items[0]"])))
            parts.append(_any_expr_ast(schema, table, dialect, table_alias=table_alias))
        return exp.Anonymous(this=name, expressions=parts)

    if name == "JSON_ARRAY":
        args = [_any_expr_ast(schema, table, dialect, table_alias=table_alias) for _ in range(random.randint(0, 4))]
        return exp.Anonymous(this="JSON_ARRAY", expressions=args)

    return exp.Anonymous(this=name, expressions=_build_args_ast(schema, table, dialect, spec, table_alias=table_alias))


def _expr_for_type_ast(
    schema: dict,
    table: str,
    dialect: str,
    type_tag: str,
    allow_aggregate: bool = True,
    allow_subqueries: bool = False,
    table_ids: dict | None = None,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    if type_tag == "boolean":
        return _boolean_expr_ast(schema, table, table_alias=table_alias)
    if allow_subqueries and table_ids is not None and type_tag in {"numeric", "string", "date", "datetime"} and random.random() < 0.3:
        return _scalar_subquery_expr_ast(
            schema,
            table,
            table_ids,
            dialect,
            type_tag,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table_alias,
        )
    col_expr = _pick_condition_col_ast(
        schema,
        table,
        type_tag,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )
    if col_expr is not None and random.random() < 0.55:
        return col_expr
    if random.random() < 0.4:
        return _function_expr_ast(
            schema,
            table,
            dialect,
            type_tag,
            allow_aggregate=allow_aggregate,
            table_alias=table_alias,
        )
    if col_expr is not None:
        return col_expr
    return _literal_ast(type_tag)


def _scalar_subquery_expr_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
) -> exp.Expression:
    candidate_tables = [name for name in schema if _pick_col_by_tag(schema, name, type_tag)]
    if not candidate_tables:
        return _literal_ast(type_tag)

    source_table = random.choice(candidate_tables)
    source_col = _pick_col_by_tag(schema, source_table, type_tag)
    if not source_col:
        return _literal_ast(type_tag)

    rel = _pick_related_table(schema, table) if random.random() < 0.6 else None
    where_sql = ""
    required_cols = [source_col]
    if rel:
        other, t_join, s_join = rel
        if other == source_table and s_join in schema[source_table]["columns"]:
            required_cols.append(s_join)
            source_ref, qualifier = _safe_table_ref_for_subquery(
                schema,
                source_table,
                target_table=target_table,
                protected_tables=protected_tables,
                alias="_case_sub",
                required_cols=required_cols,
            )
            qualified_join = f"{qualifier}.{s_join}"
            where_sql = f" WHERE {qualified_join} = {(table_alias or table)}.{t_join}"
        else:
            source_ref, qualifier = _safe_table_ref_for_subquery(
                schema,
                source_table,
                target_table=target_table,
                protected_tables=protected_tables,
                alias="_case_sub",
                required_cols=required_cols,
            )
    else:
        source_ref, qualifier = _safe_table_ref_for_subquery(
            schema,
            source_table,
            target_table=target_table,
            protected_tables=protected_tables,
            alias="_case_sub",
            required_cols=required_cols,
        )

    qualified_src = f"{qualifier}.{source_col}"
    if type_tag == "numeric":
        if random.random() < 0.65:
            agg = random.choice(["MAX", "MIN", "SUM", "AVG"])
            stmt = f"SELECT {agg}({qualified_src}) FROM {source_ref}{where_sql}"
        else:
            stmt = f"SELECT {qualified_src} FROM {source_ref}{where_sql} ORDER BY {qualified_src} DESC LIMIT 1"
    else:
        if random.random() < 0.5:
            agg = random.choice(["MAX", "MIN"])
            stmt = f"SELECT {agg}({qualified_src}) FROM {source_ref}{where_sql}"
        else:
            stmt = f"SELECT {qualified_src} FROM {source_ref}{where_sql} ORDER BY {qualified_src} DESC LIMIT 1"

    return exp.Subquery(this=_parse_query_ast(stmt, dialect))


def _cmp_ast(op: str, left: exp.Expression, right: exp.Expression) -> exp.Expression:
    return {
        "=": exp.EQ,
        "<>": exp.NEQ,
        ">": exp.GT,
        "<": exp.LT,
        ">=": exp.GTE,
        "<=": exp.LTE,
    }[op](this=left, expression=right)


def _function_supported(name: str, dialect: str) -> bool:
    if name.startswith("JSON_") and not dialect_option(dialect, "supports_json", True):
        return False
    if name == "ADDDATE" and not dialect_option(dialect, "supports_adddate", True):
        return False
    if name == "SUBDATE" and not dialect_option(dialect, "supports_subdate", True):
        return False
    if dialect == "monetdb" and name in {
        "BIT_AND",
        "BIT_OR",
        "BIT_XOR",
        "CONCAT_WS",
        "SUBSTRING_INDEX",
        "DIV",
        "IF",
        "STD",
        "STDDEV",
        "VARIANCE",
        "DATE_ADD",
        "DATE_SUB",
        "ADDDATE",
        "SUBDATE",
        "DATEDIFF",
        "TIMEDIFF",
        "TIMESTAMPDIFF",
        "DATE_FORMAT",
        "DAYOFMONTH",
        "DAYOFWEEK",
        "DAYOFYEAR",
        "YEAR",
        "MONTH",
        "DAY",
        "HOUR",
        "MINUTE",
        "SECOND",
    }:
        return False
    if dialect == "duckdb" and name in {"SUBSTRING_INDEX"}:
        return False
    return True


def _pick_function_spec(desired_type: str, dialect: str, category: str | None = None) -> dict:
    if desired_type == "numeric":
        candidates = [
            f
            for f in FUNCTION_SPECS
            if f["return_type"] in ("numeric", "INT", "DOUBLE")
            or f["name"] in ("MIN", "MAX")
        ]
    elif desired_type == "string":
        candidates = [
            f for f in FUNCTION_SPECS if f["return_type"] in ("string",) or f["name"] in ("MIN", "MAX")
        ]
    elif desired_type == "date":
        candidates = [
            f
            for f in FUNCTION_SPECS
            if f["return_type"] == "date" or f["name"] in ("MIN", "MAX")
        ]
    elif desired_type == "datetime":
        candidates = [
            f
            for f in FUNCTION_SPECS
            if f["return_type"] == "datetime" or f["name"] in ("MIN", "MAX")
        ]
    elif desired_type == "time":
        candidates = [
            f
            for f in FUNCTION_SPECS
            if f["return_type"] == "time" or f["name"] in ("MIN", "MAX")
        ]
    elif desired_type == "json":
        candidates = [f for f in FUNCTION_SPECS if f["return_type"] == "json"]
    else:
        candidates = [f for f in FUNCTION_SPECS]

    if desired_type != "json":
        candidates = [f for f in candidates if not f["name"].startswith("JSON_")]

    candidates = [f for f in candidates if _function_supported(f["name"], dialect)]

    if category is not None:
        candidates = [f for f in candidates if f.get("category") == category]

    if not candidates:
        candidates = [f for f in FUNCTION_SPECS if _function_supported(f["name"], dialect)] or FUNCTION_SPECS
    return random.choice(candidates)


def _variadic_arg_count(spec: dict) -> int:
    min_args = spec["min_args"]
    max_args = spec["max_args"]
    if max_args is not None:
        return random.randint(min_args, max_args)

    name = spec["name"]
    if name == "JSON_ARRAY":
        return random.randint(0, 4)
    if name in ("CONCAT", "GROUP_CONCAT"):
        return random.randint(max(2, min_args), max(2, min_args) + 3)
    if name == "CONCAT_WS":
        return random.randint(max(3, min_args), max(3, min_args) + 3)
    if name in ("JSON_EXTRACT", "JSON_REMOVE"):
        return random.randint(min_args, min_args + 3)
    return random.randint(min_args, min_args + 3)


def _build_args(
    schema: dict,
    table: str,
    dialect: str,
    spec: dict,
    table_alias: str | None = None,
) -> list:
    min_args = spec["min_args"]
    count = _variadic_arg_count(spec)
    arg_types = spec.get("arg_types") or ["any"]

    args = []
    for i in range(count):
        if i < len(arg_types):
            type_tag = arg_types[i]
        else:
            type_tag = arg_types[-1]
        if type_tag == "any":
            col = _pick_any_col(schema, table)
            args.append(_qualify_col(col, table, table_alias))
        else:
            col = _pick_col_for_type(schema, table, type_tag)
            if col and random.random() < 0.7:
                args.append(_qualify_col(col, table, table_alias))
            else:
                args.append(_literal_for_type(type_tag))
    return args


def _function_expr(
    schema: dict,
    table: str,
    dialect: str,
    desired_type: str,
    allow_aggregate: bool = True,
    table_alias: str | None = None,
) -> str:
    return _expr_sql(
        _function_expr_ast(
            schema,
            table,
            dialect,
            desired_type,
            allow_aggregate=allow_aggregate,
            table_alias=table_alias,
        ),
        dialect,
    )


def _expr_for_type(
    schema: dict,
    table: str,
    dialect: str,
    type_tag: str,
    allow_aggregate: bool = True,
    table_alias: str | None = None,
) -> str:
    return _expr_sql(
        _expr_for_type_ast(
            schema,
            table,
            dialect,
            type_tag,
            allow_aggregate=allow_aggregate,
            table_alias=table_alias,
        ),
        dialect,
    )


def _string_expr(schema: dict, table: str, table_alias: str | None = None) -> str:
    return _expr_sql(_string_expr_ast(schema, table, table_alias=table_alias), "mysql")


def _any_expr(schema: dict, table: str, dialect: str, table_alias: str | None = None) -> str:
    return _expr_sql(_any_expr_ast(schema, table, dialect, table_alias=table_alias), dialect)




def _simple_expr_for_type(schema: dict, table: str, type_tag: str, table_alias: str | None = None) -> str:
    return _expr_sql(_simple_expr_ast(schema, table, type_tag, table_alias=table_alias), "mysql")


def _supports_window_functions(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_window_functions", False))


def _window_projected_cols(column_count: int) -> list[str]:
    return ["k"] + [f"k{i}" for i in range(2, column_count + 1)]


def _window_over_clause(order_expr: str, direction: str, partition_expr: str | None = None) -> str:
    if partition_expr:
        return f"PARTITION BY {partition_expr} ORDER BY {order_expr} {direction}"
    return f"ORDER BY {order_expr} {direction}"


def _window_selector_expr(
    order_expr: str,
    *,
    alias: str = "rn",
    partition_expr: str | None = None,
    force_row_number: bool = False,
) -> str:
    direction = random.choice(["ASC", "DESC"])
    func = "ROW_NUMBER" if force_row_number else random.choice(WINDOW_SELECTOR_FUNCTIONS)
    over_clause = _window_over_clause(order_expr, direction, partition_expr)
    return f"{func}() OVER ({over_clause}) AS {alias}"


def _window_value_expr(
    value_expr: str,
    order_expr: str,
    *,
    alias: str = "v",
    partition_expr: str | None = None,
) -> str:
    if random.random() < 0.5:
        return f"{value_expr} AS {alias}"
    direction = random.choice(["ASC", "DESC"])
    func = random.choice(WINDOW_VALUE_FUNCTIONS)
    over_clause = _window_over_clause(order_expr, direction, partition_expr)
    return f"COALESCE({func}({value_expr}) OVER ({over_clause}), {value_expr}) AS {alias}"


def _wrap_stmt_with_window_filter(
    stmt: str,
    column_count: int,
    *,
    order_col: str = "k",
) -> str:
    src_alias = f"win_src_{rand_str(3, 5)}"
    win_alias = f"win_{rand_str(3, 5)}"
    projected_cols = _window_projected_cols(column_count)
    if order_col not in projected_cols:
        order_col = projected_cols[0]
    order_expr = f"{src_alias}.{order_col}"
    shifted_col = random.choice(projected_cols)
    inner_projection_parts = []
    for col in projected_cols:
        value_expr = f"{src_alias}.{col}"
        if col == shifted_col:
            inner_projection_parts.append(_window_value_expr(value_expr, order_expr, alias=col))
        else:
            inner_projection_parts.append(f"{value_expr} AS {col}")
    inner_projection = ", ".join(inner_projection_parts)
    outer_projection = ", ".join(f"{win_alias}.{col} AS {col}" for col in projected_cols)
    return (
        f"SELECT {outer_projection} FROM ("
        f"SELECT {inner_projection}, "
        f"{_window_selector_expr(order_expr, alias='rn')} "
        f"FROM ({stmt}) AS {src_alias}"
        f") AS {win_alias} WHERE {win_alias}.rn = 1 ORDER BY {win_alias}.{order_col}"
    )

# SET strategies

def set_assign_one(schema: dict, table: str, dialect: str) -> str:
    col = _pick_any_col(schema, table)
    return f"{table}.{col} = {gen_value(schema[table]['columns'][col], dialect, column_meta(schema, table, col))}"


def set_assign_two(schema: dict, table: str, dialect: str) -> str:
    cols = non_pk_columns(schema, table)
    if len(cols) < 2:
        return set_assign_one(schema, table, dialect)
    c1, c2 = random.sample(cols, 2)
    return (
        f"{table}.{c1} = {gen_value(schema[table]['columns'][c1], dialect, column_meta(schema, table, c1))}, "
        f"{table}.{c2} = {gen_value(schema[table]['columns'][c2], dialect, column_meta(schema, table, c2))}"
    )


def set_numeric_expr(schema: dict, table: str, dialect: str) -> str:
    col = _pick_numeric_col(schema, table)
    return f"{table}.{col} = {table}.{col} * 1.05"


def set_string_concat(schema: dict, table: str, dialect: str) -> str:
    col = _pick_string_col(schema, table)
    meta = column_meta(schema, table, col)
    if meta.get("enum_values") or meta.get("set_values"):
        return set_assign_one(schema, table, dialect)
    suffix = rand_str(3, 5)
    max_len = meta.get("max_len")
    if max_len:
        return f"{table}.{col} = LEFT(CONCAT({table}.{col}, {sql_str(suffix)}), {int(max_len)})"
    return f"{table}.{col} = CONCAT({table}.{col}, {sql_str(suffix)})"


def set_case_expr(schema: dict, table: str, dialect: str) -> str:
    col = _pick_numeric_col(schema, table)
    return f"{table}.{col} = CASE WHEN {table}.{col} IS NULL THEN 0 ELSE {table}.{col} + 1 END"


def set_subquery_agg(
    schema: dict,
    table: str,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> str:
    col = _pick_aggregate_numeric_col(schema, table, dialect)
    spec = _pick_function_spec("numeric", dialect, category="aggregate")
    name = spec["name"]
    wrapped_ref = _double_wrapped_table_ref(schema, table, alias="_sub", required_cols=[col])
    qualified_col = f"_sub.{col}"
    if name == "COUNT_DISTINCT":
        expr = f"COUNT(DISTINCT {qualified_col})"
    elif name == "SUM_DISTINCT":
        expr = f"SUM(DISTINCT {qualified_col})"
    else:
        expr = f"{name}({qualified_col})"
    return f"{table}.{col} = (SELECT {expr} FROM {wrapped_ref})"


def set_subquery_agg_other(
    schema: dict,
    table: str,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> str:
    target_col = _pick_aggregate_numeric_col(schema, table, dialect)
    rel = _pick_related_table(schema, table)
    if not rel or not target_col:
        return set_assign_one(schema, table, dialect)
    other, t_join, s_join = rel
    other_col = _pick_aggregate_numeric_col(schema, other, dialect)
    spec = _pick_function_spec("numeric", dialect, category="aggregate")
    name = spec["name"]
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=[other_col, s_join],
    )
    qualified_col = f"{qualifier}.{other_col}"
    join_col = f"{qualifier}.{s_join}"
    if name == "COUNT_DISTINCT":
        expr = f"COUNT(DISTINCT {qualified_col})"
    elif name == "SUM_DISTINCT":
        expr = f"SUM(DISTINCT {qualified_col})"
    else:
        expr = f"{name}({qualified_col})"
    return (
        f"{table}.{target_col} = (SELECT {expr} FROM {other_ref} "
        f"WHERE {join_col} = {table}.{t_join})"
    )


def set_subquery_limit_other(
    schema: dict,
    table: str,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> str:
    allowed = [t for t in _available_types(schema, table) if t in ("numeric", "string", "date", "datetime")]
    if not allowed:
        return set_assign_one(schema, table, dialect)
    type_tag = random.choice(allowed)
    target_col = _pick_col_by_tag(schema, table, type_tag)
    rel = _pick_related_table(schema, table)
    if not rel or not target_col:
        return set_assign_one(schema, table, dialect)
    other, t_join, s_join = rel
    source_col = _pick_col_by_tag(schema, other, type_tag)
    if not source_col:
        return set_assign_one(schema, table, dialect)
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=[source_col, s_join],
    )
    qualified_src = f"{qualifier}.{source_col}"
    join_col = f"{qualifier}.{s_join}"
    return (
        f"{table}.{target_col} = (SELECT {qualified_src} FROM {other_ref} "
        f"WHERE {join_col} = {table}.{t_join} "
        f"ORDER BY {qualified_src} DESC LIMIT 1)"
    )


def set_subquery_window_other(
    schema: dict,
    table: str,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> str:
    if not _supports_window_functions(dialect):
        return set_subquery_limit_other(schema, table, dialect, protected_tables=protected_tables)

    allowed = [t for t in _available_types(schema, table) if t in ("numeric", "string", "date", "datetime")]
    if not allowed:
        return set_assign_one(schema, table, dialect)
    type_tag = random.choice(allowed)
    target_col = _pick_col_by_tag(schema, table, type_tag)
    rel = _pick_related_table(schema, table)
    if not rel or not target_col:
        return set_assign_one(schema, table, dialect)
    other, t_join, s_join = rel
    source_col = _pick_col_by_tag(schema, other, type_tag)
    if not source_col:
        return set_assign_one(schema, table, dialect)

    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=[source_col, s_join],
    )
    win_alias = f"w_{rand_str(3, 5)}"
    partition_expr = f"{qualifier}.{s_join}"
    order_expr = f"{qualifier}.{source_col}"
    return (
        f"{table}.{target_col} = (SELECT {win_alias}.v FROM ("
        f"SELECT {qualifier}.{s_join} AS k, "
        f"{_window_value_expr(order_expr, order_expr, alias='v', partition_expr=partition_expr)}, "
        f"{_window_selector_expr(order_expr, alias='rn', partition_expr=partition_expr, force_row_number=True)} "
        f"FROM {other_ref}"
        f") AS {win_alias} WHERE {win_alias}.k = {table}.{t_join} AND {win_alias}.rn = 1)"
    )


def set_time_add(schema: dict, table: str, dialect: str) -> str:
    col = _pick_datetime_col(schema, table) or _pick_date_col(schema, table)
    if not col:
        return set_assign_one(schema, table, dialect)
    days = random.randint(1, 30)
    return f"{table}.{col} = {_date_shift_sql(f'{table}.{col}', days, 'DAY', dialect)}"


def set_time_literal(schema: dict, table: str, dialect: str) -> str:
    col = _pick_datetime_col(schema, table) or _pick_date_col(schema, table)
    if not col:
        return set_assign_one(schema, table, dialect)
    value = rand_datetime() if _pick_datetime_col(schema, table) else rand_date()
    return f"{table}.{col} = {sql_str(value)}"


# WHERE strategies

def _case_expr_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    type_tag: str | None = None,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    type_tag = type_tag or random.choice(_condition_available_types(schema, table, extra_col_specs))
    ifs = []
    for _ in range(random.randint(1, 2)):
        ifs.append(
            exp.If(
                this=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=False,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    extra_col_specs=extra_col_specs,
                ),
                true=_expr_for_type_ast(
                    schema,
                    table,
                    dialect,
                    type_tag,
                    allow_aggregate=False,
                    allow_subqueries=allow_subqueries,
                    table_ids=table_ids,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    extra_col_specs=extra_col_specs,
                ),
            )
        )
    default = None
    if random.random() < 0.7:
        default = _expr_for_type_ast(
            schema,
            table,
            dialect,
            type_tag,
            allow_aggregate=False,
            allow_subqueries=allow_subqueries,
            table_ids=table_ids,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table_alias,
            extra_col_specs=extra_col_specs,
        )
    return exp.Case(ifs=ifs, default=default)


def _condition_expr_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    allow_case: bool = True,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    expr = _expr_for_type_ast(
        schema,
        table,
        dialect,
        type_tag,
        allow_aggregate=False,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )
    if allow_case and type_tag != "json" and random.random() < 0.2:
        expr = _case_expr_ast(
            schema,
            table,
            table_ids,
            dialect,
            type_tag=type_tag,
            allow_subqueries=allow_subqueries,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table_alias,
            extra_col_specs=extra_col_specs,
        )
    return expr


def _predicate_ops(type_tag: str, dialect: str) -> list[str]:
    if type_tag == "json":
        return ["EXISTS", "IS", "IS NOT", "=", "<>"]
    if type_tag == "boolean":
        return ["EXISTS", "=", "<>", "IS", "IS NOT"]
    if type_tag == "string":
        ops = ["EXISTS", "=", "<>", ">", "<", ">=", "<=", "LIKE", "IN", "IS", "IS NOT"]
        if dialect_option(dialect, "supports_regexp", True):
            ops.append("REGEXP")
        return ops
    if type_tag in {"date", "datetime", "time", "numeric"}:
        return ["EXISTS", "=", "<>", ">", "<", ">=", "<=", "BETWEEN", "IN", "IS", "IS NOT"]
    return ["EXISTS", "=", "<>", "IN", "IS", "IS NOT"]


def _predicate_value_ops(type_tag: str, dialect: str) -> list[str]:
    return [op for op in _predicate_ops(type_tag, dialect) if op != "EXISTS"]


def _predicate_non_subquery_ops(type_tag: str, dialect: str) -> list[str]:
    return [op for op in _predicate_value_ops(type_tag, dialect) if op != "IN"]


def _predicate_exists_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    allow_case: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> exp.Expression:
    if random.random() < 0.45:
        type_tag = _select_type_tag(schema, table)
        stmt = _select_stmt_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth=2,
            target_table=target_table,
            protected_tables=protected_tables,
            column_count=random.randint(1, 4),
        )
        return exp.Exists(this=_parse_query_ast(stmt, dialect))
    other_table = _pick_table(
        schema,
        target_table=target_table,
        protected_tables=protected_tables,
    )
    exists_col = _exists_projection_col(schema, other_table)
    ref_sql, qualifier = _safe_table_ref_for_subquery(
        schema,
        other_table,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=_predicate_visible_subquery_cols(schema, other_table),
    )
    query = sqlglot.select(exp.column(exists_col, table=qualifier)).from_(
        sqlglot.parse_one(ref_sql, read=_sqlglot_dialect(dialect))
    )
    if random.random() < 0.7:
        inner_type_tag = random.choice(_available_types(schema, other_table))
        inner = _predicate_value_leaf_ast(
            schema,
            other_table,
            table_ids,
            dialect,
            type_tag=inner_type_tag,
            op=random.choice(_predicate_value_ops(inner_type_tag, dialect)),
            allow_case=allow_case,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=qualifier,
        )
        query = query.where(inner)
    return exp.Exists(this=query)


def _predicate_in_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    left: exp.Expression,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> exp.Expression:
    if random.random() < 0.45:
        stmt = _select_stmt_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth=2,
            target_table=target_table,
            protected_tables=protected_tables,
            column_count=1,
        )
        return exp.In(this=left, query=exp.Subquery(this=_parse_query_ast(stmt, dialect)))
    if random.random() < 0.25:
        return exp.In(this=left, expressions=[_literal_ast(type_tag) for _ in range(random.randint(1, 4))])

    source = _pick_source_column_for_type(
        schema,
        type_tag,
        target_table=target_table,
        protected_tables=protected_tables,
    )
    if source:
        source_table, source_col = source
        wrapped_ref, qualifier = _safe_table_ref_for_subquery(
            schema,
            source_table,
            target_table=target_table,
            protected_tables=protected_tables,
            alias="_sub",
            required_cols=[source_col],
        )
        query = sqlglot.select(exp.column(source_col, table=qualifier)).from_(
            sqlglot.parse_one(wrapped_ref, read=_sqlglot_dialect(dialect))
        )
        return exp.In(this=left, query=exp.Subquery(this=query))
    return exp.In(this=left, expressions=[_literal_ast(type_tag) for _ in range(random.randint(1, 4))])


def _predicate_leaf_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    allow_case: bool = True,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    type_tag = random.choice(_condition_available_types(schema, table, extra_col_specs))
    op = random.choice(_predicate_ops(type_tag, dialect) if allow_subqueries else _predicate_non_subquery_ops(type_tag, dialect))
    if op == "EXISTS":
        return _predicate_exists_ast(
            schema,
            table,
            table_ids,
            dialect,
            allow_case=allow_case,
            target_table=target_table,
            protected_tables=protected_tables,
        )
    return _predicate_value_leaf_ast(
        schema,
        table,
        table_ids,
        dialect,
        type_tag=type_tag,
        op=op,
        allow_case=allow_case,
        allow_subqueries=allow_subqueries,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )


def _predicate_value_leaf_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    op: str,
    allow_case: bool = True,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    table_alias = table_alias or table

    left = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        type_tag,
        allow_case=allow_case,
        allow_subqueries=allow_subqueries,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )

    if op == "IS":
        return exp.Is(this=left, expression=exp.Null())
    if op == "IS NOT":
        return exp.Not(this=exp.Is(this=left, expression=exp.Null()))
    if op == "BETWEEN":
        return exp.Between(this=left, low=_literal_ast(type_tag), high=_literal_ast(type_tag))
    if op == "IN":
        return _predicate_in_ast(
            schema,
            table,
            table_ids,
            dialect,
            type_tag,
            left,
            target_table=target_table,
            protected_tables=protected_tables,
        )
    if op == "LIKE":
        return exp.Like(this=left, expression=_pattern_literal_ast("LIKE"))
    if op == "REGEXP":
        return exp.RegexpLike(this=left, expression=_pattern_literal_ast("REGEXP"))

    right = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        type_tag,
        allow_case=allow_case,
        allow_subqueries=allow_subqueries,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )
    return _cmp_ast(op, left, right)


def _build_condition_ast(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    allow_case: bool = True,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    depth: int = 0,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> exp.Expression:
    max_depth = 2
    if depth < max_depth:
        roll = random.random()
        if roll < COND_NOT_PROB:
            return exp.Not(
                this=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=allow_case,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    depth=depth + 1,
                    extra_col_specs=extra_col_specs,
                )
            )
        if roll < COND_NOT_PROB + COND_AND_PROB:
            return exp.And(
                this=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=allow_case,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    depth=depth + 1,
                    extra_col_specs=extra_col_specs,
                ),
                expression=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=allow_case,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    depth=depth + 1,
                    extra_col_specs=extra_col_specs,
                ),
            )
        if roll < COND_NOT_PROB + COND_AND_PROB + COND_OR_PROB:
            return exp.Or(
                this=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=allow_case,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    depth=depth + 1,
                    extra_col_specs=extra_col_specs,
                ),
                expression=_build_condition_ast(
                    schema,
                    table,
                    table_ids,
                    dialect,
                    allow_case=allow_case,
                    allow_subqueries=allow_subqueries,
                    target_table=target_table,
                    protected_tables=protected_tables,
                    table_alias=table_alias,
                    depth=depth + 1,
                    extra_col_specs=extra_col_specs,
                ),
            )
    return _predicate_leaf_ast(
        schema,
        table,
        table_ids,
        dialect,
        allow_case=allow_case,
        allow_subqueries=allow_subqueries,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table_alias,
        extra_col_specs=extra_col_specs,
    )

def _case_expr(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
) -> str:
    return _expr_sql(
        _case_expr_ast(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table_alias,
        ),
        dialect,
    )


def _build_condition(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    allow_case: bool = True,
    allow_subqueries: bool = True,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    table_alias: str | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    return _expr_sql(
        _build_condition_ast(
            schema,
            table,
            table_ids,
            dialect,
            allow_case=allow_case,
            allow_subqueries=allow_subqueries,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table_alias,
            extra_col_specs=extra_col_specs,
        ),
        dialect,
    )


def _select_type_tag(schema: dict, table: str) -> str:
    types = _available_types(schema, table)
    return random.choice(types) if types else "numeric"


def _from_item_for_setop(
    schema: dict,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    depth: int,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    *,
    force_pk_key: bool = False,
) -> tuple[str, str, str]:
    if depth > 0 and random.random() < SETOP_DERIVED_FROM_PROB:
        column_count = random.randint(2, 4)
        inner = _select_stmt_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth - 1,
            target_table,
            protected_tables=protected_tables,
            column_count=column_count,
            force_pk_key=force_pk_key,
        )
        wrapped_inner, visible_cols = _wrap_setop_stmt_with_aliases(inner, column_count)
        alias = f"d_{rand_str(3, 6)}"
        key_expr = f"{alias}.{visible_cols[0]}"
        value_expr = f"{alias}.{visible_cols[1]}" if len(visible_cols) > 1 else key_expr
        return f"{wrapped_inner} {alias}", key_expr, value_expr

    source = None
    if force_pk_key:
        source = _pick_key_source_column_for_type(
            schema,
            type_tag,
            target_table=target_table,
            protected_tables=protected_tables,
        )
    if source:
        table, col = source
    else:
        source = _pick_source_column_for_type(
            schema,
            type_tag,
            target_table=target_table,
            protected_tables=protected_tables,
        )
        if source:
            table, col = source
        else:
            blocked = _normalize_protected_tables(target_table, protected_tables)
            candidates = [table_name for table_name in schema if table_name not in blocked and schema[table_name]["pk"] in schema[table_name]["columns"]]
            table = random.choice(candidates) if candidates else _pick_table(schema)
            col = schema[table]["pk"]
    required_cols = [col]
    if force_pk_key and schema[table]["pk"] not in required_cols:
        required_cols.append(schema[table]["pk"])
    table_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        table,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=required_cols,
    )
    if force_pk_key:
        key_expr = f"{qualifier}.{schema[table]['pk']}"
    else:
        key_expr = f"{qualifier}.{col}"
    value_expr = f"{qualifier}.{col}"
    return table_ref, key_expr, value_expr


def _pick_key_source_column_for_type(
    schema: dict,
    type_tag: str,
    *,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
) -> tuple[str, str] | None:
    blocked = _normalize_protected_tables(target_table, protected_tables)
    pk_candidates: list[tuple[str, str]] = []
    for table_name in schema:
        if table_name in blocked:
            continue
        col = _pick_key_col_by_tag(schema, table_name, type_tag)
        if not col:
            continue
        pk_candidates.append((table_name, col))
    if pk_candidates:
        return random.choice(pk_candidates)
    return None


def _setop_extra_expr(col_expr: str, type_tag: str, dialect: str) -> str:
    if type_tag == "numeric":
        choices = [
            col_expr,
            f"ABS({col_expr})",
            f"{col_expr} + {random.randint(1, 9)}",
            f"COALESCE({col_expr}, {_literal_for_type(type_tag)})",
        ]
        return random.choice(choices)

    if type_tag == "string":
        suffix = _literal_for_type("string")
        choices = [
            col_expr,
            f"UPPER({col_expr})",
            f"LOWER({col_expr})",
            f"CONCAT({col_expr}, {suffix})",
            f"COALESCE({col_expr}, {_literal_for_type(type_tag)})",
        ]
        return random.choice(choices)

    if type_tag in {"date", "datetime"}:
        delta = random.randint(1, 30)
        unit = random.choice(["DAY", "MONTH"])
        choices = [
            col_expr,
            _date_shift_sql(col_expr, delta, unit, dialect),
            f"COALESCE({col_expr}, {_literal_for_type(type_tag)})",
        ]
        return random.choice(choices)

    return random.choice([col_expr, f"COALESCE({col_expr}, {_literal_for_type(type_tag)})"])


def _setop_projection_items(col_expr: str, type_tag: str, column_count: int, dialect: str) -> list[str]:
    items = [col_expr]
    for _ in range(2, column_count + 1):
        expr = _setop_extra_expr(col_expr, type_tag, dialect) if random.random() < 0.7 else _literal_for_type(type_tag)
        items.append(expr)
    return items


def _setop_projection(items: list[str]) -> str:
    cols = []
    for i, expr in enumerate(items, 1):
        alias = "k" if i == 1 else f"k{i}"
        cols.append(f"{expr} AS {alias}")
    return ", ".join(cols)


def _expr_has_column_ref(expr_sql: str, dialect: str) -> bool:
    try:
        node = sqlglot.parse_one(expr_sql, read=_sqlglot_dialect(dialect))
    except Exception:
        return True
    return any(True for _ in node.find_all(exp.Column))


def _having_clause_for_grouped_exprs(exprs: list[str], type_tag: str, dialect: str) -> str:
    probes = [sqlglot.parse_one(expr, read=_sqlglot_dialect("mysql")) for expr in exprs]
    monetdb_safe = dialect.lower().strip() == "monetdb"

    def having_operand() -> exp.Expression:
        probe = random.choice(probes).copy()
        options: list[exp.Expression] = [] if monetdb_safe else [probe]
        if type_tag in {"numeric", "string", "date", "datetime"}:
            options.extend([
                exp.Min(this=probe.copy()),
                exp.Max(this=probe.copy()),
            ])
        if type_tag == "numeric":
            options.extend([
                exp.Count(this=probe.copy()),
                exp.Sum(this=probe.copy()),
                exp.Avg(this=probe.copy()),
                exp.Count(this=probe.copy()),
                exp.Count(this=exp.Distinct(expressions=[probe.copy()])),
            ])
        if not options:
            options.append(exp.Count(this=exp.Star()))
        return random.choice(options)

    def having_leaf() -> exp.Expression:
        if type_tag == "string":
            op = random.choice(
                ["=", "<>", ">", "<", ">=", "<=", "LIKE", "IS", "IS NOT"]
                + (["REGEXP"] if dialect_option(dialect, "supports_regexp", True) else [])
            )
        else:
            op = random.choice(["=", "<>", ">", "<", ">=", "<=", "BETWEEN", "IS", "IS NOT"])
        left = having_operand()
        if op == "IS":
            return exp.Is(this=left, expression=exp.Null())
        if op == "IS NOT":
            return exp.Not(this=exp.Is(this=left, expression=exp.Null()))
        if op == "BETWEEN":
            return exp.Between(this=left, low=_literal_ast(type_tag), high=_literal_ast(type_tag))
        if op == "LIKE":
            return exp.Like(this=left, expression=_pattern_literal_ast("LIKE"))
        if op == "REGEXP":
            return exp.RegexpLike(this=left, expression=_pattern_literal_ast("REGEXP"))
        return _cmp_ast(op, left, having_operand())

    def having_predicate(depth: int = 0) -> exp.Expression:
        if depth < 2:
            roll = random.random()
            if roll < COND_NOT_PROB:
                return exp.Not(this=having_predicate(depth + 1))
            if roll < COND_NOT_PROB + COND_AND_PROB:
                return exp.And(this=having_predicate(depth + 1), expression=having_predicate(depth + 1))
            if roll < COND_NOT_PROB + COND_AND_PROB + COND_OR_PROB:
                return exp.Or(this=having_predicate(depth + 1), expression=having_predicate(depth + 1))
        return having_leaf()

    return _expr_sql(having_predicate(), dialect)


def _select_core_for_setop(
    schema: dict,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    depth: int,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    column_count: int = 1,
    force_pk_key: bool = False,
) -> str:
    from_sql, key_expr, col_expr = _from_item_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth,
        target_table,
        protected_tables=protected_tables,
        force_pk_key=force_pk_key,
    )
    projection_items = _setop_projection_items(col_expr, type_tag, column_count, dialect)
    if projection_items:
        projection_items[0] = key_expr
    projection = _setop_projection(projection_items)
    select_prefix = "SELECT DISTINCT" if dialect_option(dialect, "supports_select_distinct", True) and random.random() < 0.2 else "SELECT"
    sql = f"{select_prefix} {projection} FROM {from_sql}"
    if (
        dialect_option(dialect, "supports_select_having", True)
        and type_tag in {"numeric", "string", "date", "datetime"}
        and random.random() < 0.2
    ):
        if dialect.lower().strip() == "monetdb":
            group_cols = ", ".join("k" if i == 1 else f"k{i}" for i in range(1, len(projection_items) + 1))
        else:
            group_cols = ", ".join(str(i) for i in range(1, len(projection_items) + 1))
        having_pred = _having_clause_for_grouped_exprs(projection_items, type_tag, dialect)
        sql += f" GROUP BY {group_cols} HAVING {having_pred}"
    return sql


def _select_stmt_for_setop(
    schema: dict,
    table_ids: dict,
    dialect: str,
    type_tag: str,
    depth: int,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    column_count: int = 1,
    force_pk_key: bool = False,
) -> str:
    if random.random() < SETOP_COMBINE_PROB:
        left = _select_core_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth,
            target_table,
            protected_tables=protected_tables,
            column_count=column_count,
            force_pk_key=force_pk_key,
        )
        right = _select_core_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth,
            target_table,
            protected_tables=protected_tables,
            column_count=column_count,
            force_pk_key=force_pk_key,
        )
        op = random.choice(get_supported_set_ops(dialect))
        return f"{left} {op} {right}"
    return _select_core_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth,
        target_table,
        protected_tables=protected_tables,
        column_count=column_count,
        force_pk_key=force_pk_key,
    )


def where_op_condition(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    return _build_condition(
        schema,
        table,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )


def where_op_compound(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    c1 = _build_condition(
        schema,
        table,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    c2 = _build_condition(
        schema,
        table,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    if random.random() >= WHERE_OP_COMPOSITE_PROB:
        return c1
    joiner = random.choice(["AND", "OR"])
    return f"({c1}) {joiner} ({c2})"


def where_exists_subquery(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    if random.random() < SETOP_SUBQUERY_PROB:
        type_tag = random.choice(_condition_available_types(schema, table, extra_col_specs))
        stmt = _select_stmt_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth=2,
            target_table=target_table,
            protected_tables=protected_tables,
            column_count=random.randint(2, 4),
        )
        return f"EXISTS ({stmt})"
    rel = _pick_related_table(
        schema,
        table,
        blocked_tables=_normalize_protected_tables(target_table, protected_tables),
    )
    if not rel:
        return _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            extra_col_specs=extra_col_specs,
        )
    other, t_join, s_join = rel
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
    )
    cond = _build_condition(
        schema,
        other,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=qualifier,
    )
    other_col = f"{qualifier}.{s_join}"
    return f"EXISTS (SELECT {other_col} FROM {other_ref} WHERE {other_col} = {table}.{t_join} AND {cond})"


def where_in_subquery(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    if random.random() < SETOP_SUBQUERY_PROB:
        type_tag = random.choice(_condition_available_types(schema, table, extra_col_specs))
        left_expr = _condition_expr_ast(
            schema,
            table,
            table_ids,
            dialect,
            type_tag,
            allow_case=False,
            allow_subqueries=False,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table,
            extra_col_specs=extra_col_specs,
        )
        if left_expr is None:
            return _build_condition(
                schema,
                table,
                table_ids,
                dialect,
                target_table=target_table,
                protected_tables=protected_tables,
                extra_col_specs=extra_col_specs,
            )
        stmt = _select_stmt_for_setop(
            schema,
            table_ids,
            dialect,
            type_tag,
            depth=2,
            target_table=target_table,
            protected_tables=protected_tables,
            column_count=1,
        )
        return f"{_expr_sql(left_expr, dialect)} IN ({stmt})"
    rel = _pick_related_table(
        schema,
        table,
        allowed_type_tags={"numeric", "string", "date", "datetime"},
        blocked_tables=_normalize_protected_tables(target_table, protected_tables),
    )
    if not rel:
        return _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            extra_col_specs=extra_col_specs,
        )
    other, t_join, s_join = rel
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
    )
    cond = _build_condition(
        schema,
        other,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=qualifier,
    )
    other_col = f"{qualifier}.{s_join}"
    left_expr = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        _type_tag_for_col_type(schema[table]["columns"][t_join]),
        allow_case=False,
        allow_subqueries=False,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    return f"{_expr_sql(left_expr, dialect)} IN (SELECT {other_col} FROM {other_ref} WHERE {cond})"


def where_compare_subquery(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    rel = _pick_related_table(
        schema,
        table,
        allowed_type_tags={"numeric"},
        blocked_tables=_normalize_protected_tables(target_table, protected_tables),
    )
    if not rel:
        return _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table,
            extra_col_specs=extra_col_specs,
        )
    other, t_join, s_join = rel
    other_num = _pick_aggregate_numeric_col(schema, other, dialect)
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=[s_join, other_num],
    )
    other_col = f"{qualifier}.{s_join}"
    avg_col = f"{qualifier}.{other_num}"
    left_expr = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        "numeric",
        allow_case=False,
        allow_subqueries=False,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    left_sql = _expr_sql(left_expr, dialect)
    return f"{left_sql} > (SELECT AVG({avg_col}) FROM {other_ref} WHERE {other_col} = {table}.{t_join})"


def where_derived_filter(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    rel = _pick_related_table(schema, table)
    if not rel:
        return _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table,
            extra_col_specs=extra_col_specs,
        )
    other, t_join, s_join = rel
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
    )
    cond = _build_condition(
        schema,
        other,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=qualifier,
    )
    col_ref = f"{qualifier}.{s_join}"
    left_expr = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        _type_tag_for_col_type(schema[table]["columns"][t_join]),
        allow_case=False,
        allow_subqueries=False,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    return f"{_expr_sql(left_expr, dialect)} IN (SELECT d.k FROM (SELECT {col_ref} AS k FROM {other_ref} WHERE {cond}) d)"


def where_window_filter(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
) -> str:
    if not _supports_window_functions(dialect):
        return where_derived_filter(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            extra_col_specs=extra_col_specs,
        )
    rel = _pick_related_table(schema, table)
    if not rel:
        return _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            table_alias=table,
            extra_col_specs=extra_col_specs,
        )
    other, t_join, s_join = rel
    other_ref, qualifier = _safe_table_ref_for_subquery(
        schema,
        other,
        target_table=target_table,
        protected_tables=protected_tables,
        alias="_sub",
        required_cols=_all_projectable_columns(schema, other),
    )
    cond = _build_condition(
        schema,
        other,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=qualifier,
    )
    alias = f"d_{rand_str(3, 5)}"
    order_expr = f"{qualifier}.{s_join}"
    left_expr = _condition_expr_ast(
        schema,
        table,
        table_ids,
        dialect,
        _type_tag_for_col_type(schema[table]["columns"][t_join]),
        allow_case=False,
        allow_subqueries=False,
        target_table=target_table,
        protected_tables=protected_tables,
        table_alias=table,
        extra_col_specs=extra_col_specs,
    )
    return (
        f"{_expr_sql(left_expr, dialect)} IN (SELECT {alias}.k FROM ("
        f"SELECT {qualifier}.{s_join} AS k, "
        f"{_window_selector_expr(order_expr, alias='rn')} "
        f"FROM {other_ref} WHERE {cond}"
        f") AS {alias} WHERE {alias}.rn = 1)"
    )


def _order_expr(schema: dict, table: str) -> str:
    col = _pick_any_col(schema, table)
    return f"{table}.{col}"


# tail strategies

def tail_none(schema: dict, table: str, dialect: str) -> str:
    return ""



def tail_order_limit(schema: dict, table: str, dialect: str) -> str:
    return ""



def tail_order_expr_limit(schema: dict, table: str, dialect: str) -> str:
    return ""



def tail_limit_offset(schema: dict, table: str, dialect: str) -> str:
    return ""



def _choose_update_set_clause(
    schema: dict,
    table: str,
    dialect: str,
    protected_tables: set[str] | None = None,
) -> str:
    basic_set = [
        set_assign_one,
        set_assign_two,
        set_numeric_expr,
        set_string_concat,
        set_case_expr,
        set_time_add,
        set_time_literal,
    ]
    select_set = [
        set_subquery_agg,
        set_subquery_agg_other,
        set_subquery_limit_other,
    ]
    if _supports_window_functions(dialect) and random.random() < WINDOW_FUNCTION_PROB:
        select_set.append(set_subquery_window_other)
    if random.random() < UPDATE_SET_SELECT_PROB:
        strategy = random.choice(select_set)
        return strategy(schema, table, dialect, protected_tables=protected_tables)
    return random.choice(basic_set)(schema, table, dialect)


def _rhs_assignable_specs(rhs_specs: list[ConditionColSpec]) -> list[ConditionColSpec]:
    return [
        spec
        for spec in rhs_specs
        if spec[2] in {"numeric", "string", "date", "datetime", "time", "boolean"}
    ]


def _join_set_target_cols(schema: dict, table: str, type_tag: str) -> list[str]:
    cols = [col for col in columns_by_type_tag(schema, table, type_tag) if col != schema[table]["pk"]]
    if cols:
        return cols
    fallback = [col for col in non_pk_columns(schema, table) if _type_tag_for_col_type(schema[table]["columns"][col]) == type_tag]
    return fallback


def _rhs_value_expr(spec: ConditionColSpec, dialect: str, *, target_meta: dict | None = None) -> str:
    ref = _spec_ref(spec)
    _, _, type_tag = spec
    target_meta = target_meta or {}

    if type_tag == "numeric":
        options = [
            ref,
            f"ABS({ref})",
            f"{ref} + {random.randint(1, 9)}",
            f"{ref} - {random.randint(1, 9)}",
        ]
        return random.choice(options)

    if type_tag == "string":
        if target_meta.get("enum_values") or target_meta.get("set_values"):
            return ref
        options = [
            ref,
            f"UPPER({ref})",
            f"LOWER({ref})",
            f"CONCAT({ref}, {sql_str(rand_str(2, 4))})",
        ]
        max_len = target_meta.get("max_len")
        expr = random.choice(options)
        if max_len and "CONCAT(" in expr:
            return f"LEFT({expr}, {int(max_len)})"
        return expr

    if type_tag == "date":
        options = [
            ref,
            _date_shift_sql(ref, random.randint(1, 15), "DAY", dialect),
            _date_shift_sql(ref, random.randint(1, 6), "MONTH", dialect),
        ]
        return random.choice(options)

    if type_tag == "datetime":
        options = [
            ref,
            _date_shift_sql(ref, random.randint(1, 15), "DAY", dialect),
            _date_shift_sql(ref, random.randint(1, 12), "HOUR", dialect),
        ]
        return random.choice(options)

    if type_tag == "time":
        return ref

    if type_tag == "boolean":
        return random.choice([ref, f"NOT {ref}"])

    return ref


def _join_rhs_set_clause(
    schema: dict,
    table: str,
    dialect: str,
    rhs_specs: list[ConditionColSpec],
) -> str | None:
    usable_specs = _rhs_assignable_specs(rhs_specs)
    if not usable_specs:
        return None

    compatible_targets: list[tuple[str, str, ConditionColSpec]] = []
    for spec in usable_specs:
        for target_col in _join_set_target_cols(schema, table, spec[2]):
            compatible_targets.append((target_col, spec[2], spec))

    if not compatible_targets:
        return None

    target_col, _, spec = random.choice(compatible_targets)
    target_ref = f"{table}.{target_col}"
    target_meta = column_meta(schema, table, target_col)
    assignments = [f"{target_ref} = {_rhs_value_expr(spec, dialect, target_meta=target_meta)}"]

    remaining = [
        item
        for item in compatible_targets
        if item[0] != target_col and item[2] != spec
    ]
    if remaining and random.random() < 0.25:
        extra_target, _, extra_spec = random.choice(remaining)
        extra_ref = f"{table}.{extra_target}"
        extra_meta = column_meta(schema, table, extra_target)
        assignments.append(f"{extra_ref} = {_rhs_value_expr(extra_spec, dialect, target_meta=extra_meta)}")

    return ", ".join(assignments)


def _choose_update_join_set_clause(
    schema: dict,
    table: str,
    dialect: str,
    rhs_specs: list[ConditionColSpec] | None = None,
) -> str:
    local_clause = _choose_update_set_clause(schema, table, dialect, protected_tables={table})
    if not rhs_specs or random.random() >= 0.6:
        return local_clause
    rhs_clause = _join_rhs_set_clause(schema, table, dialect, rhs_specs)
    return rhs_clause or local_clause


def _choose_update_join_local_set_clause(
    schema: dict,
    table: str,
    dialect: str,
    rhs_specs: list[ConditionColSpec] | None = None,
) -> str:
    return _choose_update_join_set_clause(schema, table, dialect, rhs_specs=rhs_specs)


def _choose_update_where_clause(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    target_table: str | None = None,
    protected_tables: set[str] | None = None,
    extra_col_specs: list[ConditionColSpec] | None = None,
    target_only_columns: bool = True,
) -> str:
    basic_where = [where_op_condition, where_op_compound]
    if target_only_columns:
        return random.choice(basic_where)(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            extra_col_specs=None,
        )
    select_where = [
        where_exists_subquery,
        where_in_subquery,
        where_compare_subquery,
        where_derived_filter,
    ]
    if _supports_window_functions(dialect) and random.random() < WINDOW_FUNCTION_PROB:
        select_where.append(where_window_filter)
    if random.random() < UPDATE_WHERE_SELECT_PROB:
        return random.choice(select_where)(
            schema,
            table,
            table_ids,
            dialect,
            target_table=target_table,
            protected_tables=protected_tables,
            extra_col_specs=extra_col_specs,
        )
    return random.choice(basic_where)(
        schema,
        table,
        table_ids,
        dialect,
        target_table=target_table,
        protected_tables=protected_tables,
        extra_col_specs=extra_col_specs,
    )

def _schema_table_col_specs(
    schema: dict,
    table: str,
    alias: str,
    required_cols: list[str] | None = None,
) -> list[ConditionColSpec]:
    specs: list[ConditionColSpec] = []
    for col in _projected_columns(schema, table, required_cols=required_cols):
        col_type = schema[table]["columns"].get(col)
        if not col_type or col_type == "geom":
            continue
        specs.append((alias, col, _type_tag_for_col_type(col_type)))
    return specs


def _spec_ref(spec: ConditionColSpec) -> str:
    alias, col, _ = spec
    return f"{alias}.{col}"


def _sample_join_specs(
    specs: list[ConditionColSpec],
    *,
    max_count: int = 3,
    allowed_types: set[str] | None = None,
) -> list[ConditionColSpec]:
    pool = [
        spec
        for spec in specs
        if spec[2] not in {"json", "binary", "geom"}
        and (allowed_types is None or spec[2] in allowed_types)
    ]
    if not pool:
        return []
    return random.sample(pool, k=min(max_count, len(pool)))


def _cross_side_join_candidates(
    left_specs: list[ConditionColSpec],
    right_specs: list[ConditionColSpec],
) -> list[str]:
    compatible: list[tuple[ConditionColSpec, ConditionColSpec]] = []
    for left in left_specs:
        for right in right_specs:
            if left[2] != right[2]:
                continue
            if left[2] in {"json", "binary", "geom"}:
                continue
            compatible.append((left, right))
    if not compatible:
        return []

    candidates: list[str] = []
    for left, right in random.sample(compatible, k=min(4, len(compatible))):
        left_ref = _spec_ref(left)
        right_ref = _spec_ref(right)
        candidates.append(f"{left_ref} = {right_ref}")
        ops = _join_non_equi_ops(left[2])
        if ops:
            for op in random.sample(ops, k=min(2, len(ops))):
                candidates.append(f"{left_ref} {op} {right_ref}")
    return candidates


def _rhs_subquery_join_candidates(
    table: str,
    rhs_source: dict,
) -> list[str]:
    key_pair = rhs_source.get("key_pair")
    subquery_sql = rhs_source.get("subquery_sql")
    alias = rhs_source.get("alias")
    if not key_pair or not subquery_sql or not alias:
        return []

    left_key, right_key = key_pair
    corr_alias = f"{alias}_corr"
    return [
        f"{table}.{left_key} IN (SELECT {corr_alias}.{right_key} FROM {subquery_sql} AS {corr_alias})",
        f"EXISTS (SELECT {corr_alias}.{right_key} FROM {subquery_sql} AS {corr_alias} WHERE {corr_alias}.{right_key} = {table}.{left_key})",
        f"EXISTS (SELECT {corr_alias}.{right_key} FROM {subquery_sql} AS {corr_alias} WHERE {corr_alias}.{right_key} = {alias}.{right_key})",
    ]


def _join_rhs_table_source(
    schema: dict,
    table: str,
    dialect: str,
) -> dict | None:
    del dialect
    rel = _pick_fk_to_pk_related_table(schema, table)
    if not rel:
        return None
    other, t_join, s_join = rel
    alias = f"d_{rand_str(3, 6)}"
    return {
        "kind": "table",
        "alias": alias,
        "rhs_sql": other,
        "col_specs": _schema_table_col_specs(schema, other, alias),
        "key_pair": (t_join, s_join),
        "subquery_sql": other,
        "base_table": other,
    }


def _join_rhs_derived_source(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
) -> dict | None:
    rel = _pick_fk_to_pk_related_table(schema, table)
    if not rel:
        return None
    other, t_join, s_join = rel
    value_type = random.choice([t for t in _available_types(schema, other) if t in ("numeric", "string", "date", "datetime")] or ["numeric"])
    query = _build_update_rhs_query(
        schema,
        table,
        other,
        s_join,
        value_type,
        table_ids,
        dialect,
    )
    query = _apply_update_rhs_top_level_wrappers(
        query,
        ["k", "v"],
        dialect,
        schema=schema,
        target_table=table,
        source_table=other,
        join_col=s_join,
        value_type=value_type,
        table_ids=table_ids,
        force_setop=False,
    )
    alias = f"d_{rand_str(3, 6)}"
    rhs_sql = f"({_query_sql(query, dialect)})"
    join_key_type = _type_tag_for_col_type(schema[other]["columns"][s_join])
    return {
        "kind": "derived",
        "alias": alias,
        "rhs_sql": rhs_sql,
        "col_specs": [(alias, "k", join_key_type), (alias, "v", value_type)],
        "key_pair": (t_join, "k"),
        "subquery_sql": rhs_sql,
        "base_table": other,
    }

def _join_rhs_setop_source(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
) -> dict | None:
    rel = _pick_fk_to_pk_related_table(schema, table)
    if not rel:
        return None
    other, t_join, s_join = rel
    value_type = random.choice([t for t in _available_types(schema, other) if t in ("numeric", "string", "date", "datetime")] or ["numeric"])
    query = _build_update_rhs_query(
        schema,
        table,
        other,
        s_join,
        value_type,
        table_ids,
        dialect,
    )
    query = _apply_update_rhs_top_level_wrappers(
        query,
        ["k", "v"],
        dialect,
        schema=schema,
        target_table=table,
        source_table=other,
        join_col=s_join,
        value_type=value_type,
        table_ids=table_ids,
        force_setop=True,
    )
    alias = f"d_{rand_str(3, 6)}"
    rhs_sql = f"({_query_sql(query, dialect)})"
    join_key_type = _type_tag_for_col_type(schema[other]["columns"][s_join])
    return {
        "kind": "setop",
        "alias": alias,
        "rhs_sql": rhs_sql,
        "col_specs": [(alias, "k", join_key_type), (alias, "v", value_type)],
        "key_pair": (t_join, "k"),
        "subquery_sql": rhs_sql,
        "base_table": other,
    }


def _join_rhs_values_source(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
) -> dict | None:
    if dialect != "duckdb":
        return None
    rel = _pick_fk_to_pk_related_table(schema, table)
    if not rel:
        return None
    other, t_join, s_join = rel
    keys = list(table_ids.get(other) or [])
    if not keys:
        return None
    value_type_choices = [
        t for t in _available_types(schema, table)
        if t in ("numeric", "string", "date", "datetime")
    ]
    value_type = random.choice(value_type_choices or ["numeric"])
    row_count = min(len(keys), random.randint(1, 3))
    picked_keys = random.sample(keys, k=row_count)
    values_sql = ", ".join(
        f"({key}, {_literal_for_type(value_type)})"
        for key in picked_keys
    )
    alias = f"d_{rand_str(3, 6)}"
    join_key_type = _type_tag_for_col_type(schema[other]["columns"][s_join])
    return {
        "kind": "values",
        "alias": alias,
        "rhs_sql": f"(VALUES {values_sql})",
        "from_sql": f"(VALUES {values_sql}) AS {alias}(k, v)",
        "col_specs": [(alias, "k", join_key_type), (alias, "v", value_type)],
        "key_pair": (t_join, "k"),
        "base_table": other,
    }


def _joined_update_protected_tables(table: str, rhs_source: dict | None) -> set[str]:
    protected = {table}
    if rhs_source:
        base_table = rhs_source.get("base_table")
        if base_table:
            protected.add(base_table)
    return protected


def _pick_join_rhs_source(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    source_kind: str | None = None,
) -> dict | None:
    builders = {
        "table": lambda: _join_rhs_table_source(schema, table, dialect),
        "derived": lambda: _join_rhs_derived_source(schema, table, table_ids, dialect),
        "setop": lambda: _join_rhs_setop_source(schema, table, table_ids, dialect),
        "values": lambda: _join_rhs_values_source(schema, table, table_ids, dialect),
    }
    supported = ["table", "derived", "setop"]
    if dialect == "duckdb":
        supported.append("values")

    if source_kind in supported:
        order = [source_kind] + [kind for kind in supported if kind != source_kind]
    else:
        weighted = ["table", "table", "derived", "setop", "setop"]
        if dialect == "duckdb" and random.random() < DUCKDB_UPDATE_VALUES_SOURCE_PROB:
            weighted.extend(["values", "values"])
        first = random.choice([kind for kind in weighted if kind in supported])
        rest = [kind for kind in supported if kind != first]
        random.shuffle(rest)
        order = [first] + rest

    for kind in order:
        source = builders[kind]()
        if source:
            return source
    return None


def _join_on_candidate_pool(
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    rhs_source: dict,
) -> list[str]:
    rhs_specs = list(rhs_source["col_specs"])
    left_specs = _schema_table_col_specs(schema, table, table)
    allow_on_subqueries = dialect_option(dialect, "supports_subquery_in_join_on", True)

    candidates = [
        _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            allow_subqueries=allow_on_subqueries,
            target_table=table,
            protected_tables={table},
            table_alias=table,
            extra_col_specs=rhs_specs,
        ),
        _build_condition(
            schema,
            table,
            table_ids,
            dialect,
            allow_subqueries=False,
            target_table=table,
            protected_tables={table},
            table_alias=table,
            extra_col_specs=rhs_specs,
        ),
    ]

    base_table = rhs_source.get("base_table")
    if base_table:
        candidates.append(
            _build_condition(
                schema,
                base_table,
                table_ids,
                dialect,
                allow_subqueries=allow_on_subqueries,
                target_table=table,
                protected_tables={table},
                table_alias=rhs_source["alias"],
            )
        )

    for spec in _sample_join_specs(left_specs + rhs_specs, max_count=4):
        candidates.append(f"{_spec_ref(spec)} IS NOT NULL")

    candidates.extend(_cross_side_join_candidates(left_specs, rhs_specs))

    if allow_on_subqueries:
        candidates.extend(_rhs_subquery_join_candidates(table, rhs_source))

    deduped: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        deduped.append(cand)
    return deduped


def _required_join_anchor(table: str, rhs_source: dict) -> str | None:
    key_pair = rhs_source.get("key_pair")
    alias = rhs_source.get("alias")
    if not key_pair or not alias:
        return None
    left_key, right_key = key_pair
    return f"{table}.{left_key} = {alias}.{right_key}"


def update_unified(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    source_kind: str | None = None,
    force_join: bool | None = None,
    join_where_target_only: bool = False,
) -> str:
    table = _pick_table(schema)
    effective_dialect = dialect
    uses_update_from = _uses_update_from(dialect)
    rhs_source = None
    want_join = force_join if force_join is not None else False
    from_sql = None

    if dialect_option(dialect, "supports_update_join", True) or uses_update_from:
        want_join = force_join if force_join is not None else (random.random() < UPDATE_UNIFIED_JOIN_PROB)
        if want_join:
            rhs_source = _pick_join_rhs_source(schema, table, table_ids, effective_dialect, source_kind=source_kind)

    prefix = _update_prefix(effective_dialect)
    with_clause, extra_pred, cte_specs = _maybe_with_clause(
        schema,
        table,
        table_ids,
        effective_dialect,
        protected_tables={table},
    )

    partition_clause = _partition_clause(schema, table, effective_dialect)
    if rhs_source:
        join_type = _join_type_with_on(effective_dialect)
        rhs_specs = list(rhs_source["col_specs"])
        where_protected_tables = _joined_update_protected_tables(table, rhs_source)
        where_specs = _joined_update_where_specs(
            cte_specs,
            rhs_specs,
            target_only=join_where_target_only,
        )
        set_clause = _choose_update_join_local_set_clause(schema, table, effective_dialect, rhs_specs=rhs_specs)
        where_clause = _choose_update_where_clause(
            schema,
            table,
            table_ids,
            effective_dialect,
            target_table=table,
            protected_tables=where_protected_tables,
            extra_col_specs=where_specs,
            target_only_columns=join_where_target_only,
        )
        join_predicate = _compose_join_on_sql(
            _join_on_candidate_pool(schema, table, table_ids, effective_dialect, rhs_source),
            required_term=_required_join_anchor(table, rhs_source),
        )
        if uses_update_from:
            from_sql = rhs_source.get("from_sql") or f"{rhs_source['rhs_sql']} AS {rhs_source['alias']}"
            where_clause = f"({where_clause}) AND ({join_predicate})"
            target_sql = f"{table}{partition_clause}"
        else:
            target_sql = f"{table}{partition_clause} {join_type} {rhs_source['rhs_sql']} AS {rhs_source['alias']} ON {join_predicate}"
    else:
        set_clause = _choose_update_set_clause(schema, table, effective_dialect, protected_tables={table})
        where_clause = _choose_update_where_clause(
            schema,
            table,
            table_ids,
            effective_dialect,
            target_table=table,
            protected_tables={table},
            extra_col_specs=cte_specs,
        )
        target_sql = f"{table}{partition_clause}"

    if extra_pred and not (rhs_source and join_where_target_only):
        where_clause = f"({where_clause}) AND {extra_pred}"

    return _render_update_ast(
        with_clause=with_clause,
        prefix=prefix,
        target_sql=target_sql,
        set_clause=set_clause,
        where_clause=where_clause,
        tail_sql="",
        dialect=effective_dialect,
        from_sql=from_sql,
        returning_sql=_maybe_update_returning(schema, table, effective_dialect),
    )


def update_random_single_table(schema: dict, table_ids: dict, dialect: str) -> str:
    return update_unified(schema, table_ids, dialect, force_join=False)


def update_join_general(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    source_kind: str | None = None,
) -> str:
    return update_unified(schema, table_ids, dialect, source_kind=source_kind, force_join=True)


def update_join_derived(schema: dict, table_ids: dict, dialect: str) -> str:
    return update_unified(schema, table_ids, dialect, source_kind="derived", force_join=True)


def update_multi_table(schema: dict, table_ids: dict, dialect: str) -> str:
    return update_unified(schema, table_ids, dialect, source_kind="table", force_join=True)


def update_join_setop(schema: dict, table_ids: dict, dialect: str) -> str:
    return update_unified(schema, table_ids, dialect, source_kind="setop", force_join=True)
def build_update_variants(schema: dict, table_ids: dict, dialect: str) -> list:
    variants = []
    update_count = random.randint(3, 10)
    for _ in range(update_count):
        stmt = update_unified(schema, table_ids, dialect)
        test_update_variant(stmt)
        variants.append(stmt)
    return variants
