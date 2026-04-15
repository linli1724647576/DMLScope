import random
import re
from dataclasses import dataclass

try:
    import sqlglot
    from sqlglot import exp
except Exception as exc:  # pragma: no cover
    raise ImportError("sqlglot is required for original insert generation") from exc

from init_env.core.dialect import dialect_option, get_sqlglot_dialect, get_supported_join_types
from init_env.core.rand_utils import rand_date, rand_datetime, rand_str, sql_str
from init_env.core.row_builder import build_row
from init_env.core.schema_utils import column_nullable
from init_env.updates.update_expansions import (
    _select_stmt_for_setop,
    _type_tag_for_col_type,
)
from init_env.utils.sqlglot_compat import normalize_sqlglot_input


INSERT_SELECT_SUBQUERY_TYPES = {"numeric", "string", "date", "datetime", "json"}
INSERT_SELECT_COMPLEX_PROB = 0.75
INSERT_SELECT_SOURCE_TYPES = ["numeric", "string", "date", "datetime"]
INSERT_SELECT_GROUP_SAFE_TYPES = {"numeric", "string", "date", "datetime"}
INSERT_SELECT_GROUP_PROB = 0.18
INSERT_SELECT_HAVING_PROB = 0.18
INSERT_SELECT_WINDOW_PROB = 0.2
INSERT_ON_DUPLICATE_PROB = 0.2
INSERT_VALUES_PROB = 0.05
INSERT_ON_DUPLICATE_VALUES_PROB = 0.05
INSERT_FALLBACK_VALUES_PROB = 0.05
INSERT_BY_NAME_PROB = 0.2
INSERT_RETURNING_PROB = 0.18
DUCKDB_INSERT_ON_DUPLICATE_PROB = 0.24
DUCKDB_INSERT_VALUES_PROB = 0.16
DUCKDB_INSERT_FALLBACK_VALUES_PROB = 0.14
INSERT_TOPLEVEL_FROM_SUBQUERY_PROB = 0.2
INSERT_TOPLEVEL_SETOP_PROB = 0.15
INSERT_TOPLEVEL_CTE_PROB = 0.2
INSERT_SELECT_CTE_PROB = 0.35
DUCKDB_INSERT_MULTI_VALUES_PROB = 0.7
DUCKDB_INSERT_MULTI_VALUES_MAX_ROWS = 4
DUCKDB_UPSERT_BATCH_PROB = 0.8
DUCKDB_UPSERT_BATCH_MAX_ROWS = 4
DUCKDB_BY_NAME_SUBSET_PROB = 0.55
DUCKDB_CONFLICT_DO_NOTHING_PROB = 0.35
DUCKDB_CONFLICT_COMPLEX_EXPR_PROB = 0.7
DUCKDB_UPSERT_SELECT_EXISTING_PK_PROB = 0.85
DUCKDB_PK_SUBQUERY_EXPR_PROB = 0.6
INSERT_TOPLEVEL_SETOP_TYPES = ("UNION_ALL", "UNION", "EXCEPT", "INTERSECT")
IN_SUBQUERY_SETOP_PROB = 0.45
IN_SUBQUERY_FROM_SUBQUERY_PROB = 0.2
IN_SUBQUERY_CTE_PROB = 0.2
INSERT_WINDOW_SELECTOR_FUNCTIONS = ("ROW_NUMBER", "RANK", "DENSE_RANK")
INSERT_WINDOW_VALUE_FUNCTIONS = ("LAG", "LEAD")
COND_NOT_PROB = 0.10
COND_AND_PROB = 0.10
COND_OR_PROB = 0.06


@dataclass
class QuerySource:
    from_expression: exp.From | None
    joins: list[exp.Join]
    filters: list[exp.Expression]
    limit: int | None = None
    row_order_exprs: list[exp.Expression] | None = None
    stable_pk_expr: exp.Expression | None = None
    source_table: str | None = None
    source_alias: str | None = None


def _sqlglot_dialect(dialect: str) -> str:
    return get_sqlglot_dialect(dialect)


def _normalize_sql_for_dialect(sql: str, dialect: str) -> str:
    sql = normalize_sqlglot_input(sql, dialect)
    if dialect == "duckdb":
        return re.sub(r"\bX'([0-9A-Fa-f]+)'", r"FROM_HEX('\1')", sql)
    if dialect == "monetdb":
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
    return sql


def _split_logic_leaf_budget(total_leaves: int) -> tuple[int, int]:
    if total_leaves <= 2:
        return 1, 1
    left = random.randint(1, total_leaves - 1)
    right = total_leaves - left
    return left, right


def _parse_expr(sql: str, dialect: str) -> exp.Expression:
    return sqlglot.parse_one(_normalize_sql_for_dialect(sql, dialect), read=_sqlglot_dialect(dialect))


def _parse_query(sql: str, dialect: str) -> exp.Query:
    query = sqlglot.parse_one(_normalize_sql_for_dialect(sql, dialect), read=_sqlglot_dialect(dialect))
    if not isinstance(query, exp.Query):
        raise ValueError(f"Expected query expression, got {type(query).__name__}: {sql}")
    return query


def _target_schema(table: str, cols: list[str]) -> exp.Schema:
    return exp.Schema(
        this=exp.to_table(table),
        expressions=[exp.to_identifier(col) for col in cols],
    )


def _seed_subquery(alias: str) -> exp.Subquery:
    return sqlglot.select(exp.alias_(exp.Literal.number(1), "anchor", copy=False)).subquery(alias)


def _join_type_with_on(dialect: str) -> str:
    supported = [jt for jt in get_supported_join_types(dialect) if jt != "CROSS JOIN"]
    return random.choice(supported or ["JOIN"])


def _bind_query_source_side(
    source: QuerySource,
    *,
    table: str,
    alias: str,
    schema: dict,
) -> None:
    source.source_table = table
    source.source_alias = alias
    source.stable_pk_expr = exp.column(schema[table]["pk"], table=alias)


def _true_expr(dialect: str) -> exp.Expression:
    return _parse_expr("TRUE", dialect)


def _join_key_op(type_tag: str) -> str:
    if type_tag in {"numeric", "string", "date", "datetime"}:
        return random.choice(["=", "=", "=", "<>", ">", "<", ">=", "<="])
    return random.choice(["=", "=", "<>"])


def _ordered(order_expr: exp.Expression, direction: str) -> exp.Ordered:
    is_desc = direction.upper() == "DESC"
    return exp.Ordered(
        this=order_expr.copy(),
        desc=is_desc,
        nulls_first=True if not is_desc else None,
    )


def _window_selector_ast(order_expr: exp.Expression, *, alias: str = "rn") -> exp.Alias:
    direction = random.choice(["ASC", "DESC"])
    func_name = random.choice(INSERT_WINDOW_SELECTOR_FUNCTIONS)
    func = {
        "ROW_NUMBER": exp.RowNumber(),
        "RANK": exp.Rank(),
        "DENSE_RANK": exp.DenseRank(),
    }[func_name]
    return exp.Window(
        this=func,
        order=exp.Order(expressions=[_ordered(order_expr, direction)]),
        over="OVER",
    ).as_(alias, copy=False)


def _window_value_ast(value_expr: exp.Expression, order_expr: exp.Expression, *, alias: str) -> exp.Alias:
    if random.random() < 0.5:
        return value_expr.copy().as_(alias, copy=False)
    direction = random.choice(["ASC", "DESC"])
    func_name = random.choice(INSERT_WINDOW_VALUE_FUNCTIONS)
    func = {
        "LAG": exp.Lag(this=value_expr.copy()),
        "LEAD": exp.Lead(this=value_expr.copy()),
    }[func_name]
    return exp.Coalesce(
        this=exp.Window(
            this=func,
            order=exp.Order(expressions=[_ordered(order_expr, direction)]),
            over="OVER",
        ),
        expressions=[value_expr.copy()],
    ).as_(alias, copy=False)


def _render_sql(node: exp.Expression, dialect: str) -> str:
    return _normalize_sql_for_dialect(node.sql(dialect=_sqlglot_dialect(dialect)), dialect)


def _simple_predicate_literal(type_tag: str) -> exp.Expression:
    if type_tag == "numeric":
        return exp.Literal.number(random.randint(0, 1000))
    if type_tag == "boolean":
        return exp.Literal.number(random.randint(0, 1))
    if type_tag == "date":
        return exp.Cast(this=exp.Literal.string(rand_date()), to=exp.DataType.build("DATE"))
    if type_tag == "datetime":
        return exp.Cast(this=exp.Literal.string(rand_datetime()), to=exp.DataType.build("DATETIME"))
    return exp.Literal.string(rand_str(1, 4))


def _simple_binary_predicate(type_tag: str, col_ref: exp.Expression) -> exp.Expression:
    left = col_ref.copy()
    if type_tag == "numeric":
        op = random.choice(["=", "<>", ">", "<", ">=", "<="])
        right = _simple_predicate_literal(type_tag)
    elif type_tag == "string":
        op = random.choice(["=", "<>", ">", "<", ">=", "<=", "LIKE"])
        if op == "LIKE":
            return exp.Like(
                this=left,
                expression=exp.Literal.string(f"%{rand_str(1, 3)}%"),
            )
        right = _simple_predicate_literal(type_tag)
    elif type_tag in {"date", "datetime"}:
        op = random.choice(["=", "<>", ">", "<", ">=", "<="])
        right = _simple_predicate_literal(type_tag)
    elif type_tag == "boolean":
        op = random.choice(["=", "<>"])
        right = _simple_predicate_literal(type_tag)
    else:
        op = random.choice(["=", "<>"])
        right = _simple_predicate_literal(type_tag)
    return {
        "=": exp.EQ,
        "<>": exp.NEQ,
        ">": exp.GT,
        "<": exp.LT,
        ">=": exp.GTE,
        "<=": exp.LTE,
    }[op](this=left, expression=right)


def _predicate_source_candidates(schema: dict, type_tag: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for source_table, info in schema.items():
        for source_col, source_col_type in info["columns"].items():
            if source_col_type in {"bin", "geom", "json"}:
                continue
            if _type_tag_for_col_type(source_col_type) == type_tag:
                candidates.append((source_table, source_col))
    return candidates


def _simple_in_subquery_base_query(
    schema: dict,
    type_tag: str,
    dialect: str,
) -> exp.Query | None:
    candidates = _predicate_source_candidates(schema, type_tag)
    if not candidates:
        return None
    source_table, source_col = random.choice(candidates)
    alias = f"ins_in_{random.randint(100, 999)}"
    source_ref = exp.column(source_col, table=alias)
    query = sqlglot.select(source_ref.copy()).from_(exp.to_table(source_table).as_(alias, copy=False), copy=False)
    if random.random() < 0.7:
        query = query.where(_simple_binary_predicate(type_tag, source_ref), copy=False)
    if dialect_option(dialect, "supports_select_distinct", True) and random.random() < 0.3:
        query.set("distinct", exp.Distinct())
    return query


def _wrap_single_column_from_subquery(query: exp.Expression, col_alias: str) -> exp.Select:
    alias = f"ins_in_wrap_{rand_str(3, 5)}"
    return sqlglot.select(
        exp.column(col_alias, table=alias).as_(col_alias, copy=False)
    ).from_(_query_with_output_aliases(query, [col_alias]).subquery(alias), copy=False)


def _wrap_single_column_cte(query: exp.Expression, col_alias: str) -> exp.Select:
    cte_name = f"cte_in_{rand_str(3, 5)}"
    alias = f"ins_in_cte_{rand_str(3, 5)}"
    outer = sqlglot.select(
        exp.column(col_alias, table=alias).as_(col_alias, copy=False)
    ).from_(exp.to_table(cte_name).as_(alias, copy=False), copy=False)
    return outer.with_(cte_name, as_=_query_with_output_aliases(query, [col_alias]), copy=False)


def _complex_in_subquery_query(
    schema: dict,
    type_tag: str,
    table_ids: dict | None,
    dialect: str,
) -> exp.Query | None:
    if table_ids is not None and random.random() < IN_SUBQUERY_SETOP_PROB:
        base: exp.Expression = _parse_query(
            _select_stmt_for_setop(
                schema,
                table_ids,
                dialect,
                type_tag,
                depth=2,
                column_count=1,
            ),
            dialect,
        )
    else:
        base = _simple_in_subquery_base_query(schema, type_tag, dialect)
        if base is None:
            return None

    base = _query_with_output_aliases(base, ["k"])
    roll = random.random()
    if roll < IN_SUBQUERY_FROM_SUBQUERY_PROB:
        return _wrap_single_column_from_subquery(base, "k")
    if (
        roll < IN_SUBQUERY_FROM_SUBQUERY_PROB + IN_SUBQUERY_CTE_PROB
        and dialect_option(dialect, "supports_dml_with", True)
        and base.args.get("with") is None
    ):
        return _wrap_single_column_cte(base, "k")
    return base


def _simple_in_subquery_predicate(
    schema: dict,
    type_tag: str,
    col_ref: exp.Expression,
    dialect: str,
    *,
    table_ids: dict | None = None,
) -> exp.Expression | None:
    query = _complex_in_subquery_query(schema, type_tag, table_ids, dialect)
    if query is None:
        return None
    return exp.In(this=col_ref.copy(), query=exp.Subquery(this=query))


def _simple_in_predicate(
    type_tag: str,
    col_ref: exp.Expression,
    *,
    schema: dict | None = None,
    dialect: str | None = None,
    table_ids: dict | None = None,
) -> exp.Expression:
    if schema is not None and dialect is not None and random.random() < 0.6:
        subquery_pred = _simple_in_subquery_predicate(schema, type_tag, col_ref, dialect, table_ids=table_ids)
        if subquery_pred is not None:
            return subquery_pred
    return exp.In(
        this=col_ref.copy(),
        expressions=[_simple_predicate_literal(type_tag) for _ in range(random.randint(2, 4))],
    )


def _simple_typed_predicate(
    type_tag: str,
    col_ref: exp.Expression,
    *,
    schema: dict | None = None,
    dialect: str | None = None,
    table_ids: dict | None = None,
) -> exp.Expression:
    if type_tag in {"numeric", "string", "date", "datetime", "boolean"} and random.random() < 0.25:
        return _simple_in_predicate(type_tag, col_ref, schema=schema, dialect=dialect, table_ids=table_ids)
    return _simple_binary_predicate(type_tag, col_ref)


def _simple_scalar_predicate(
    type_tag: str,
    col_ref: exp.Expression,
    *,
    schema: dict | None = None,
    dialect: str | None = None,
    table_ids: dict | None = None,
) -> exp.Expression:
    return _simple_typed_predicate(type_tag, col_ref, schema=schema, dialect=dialect, table_ids=table_ids)


def _simple_scalar_subquery(
    *,
    schema: dict,
    table_ids: dict | None,
    table_name: str,
    column_name: str,
    type_tag: str,
    dialect: str,
) -> exp.Expression:
    alias = f"ins_scalar_{random.randint(100, 999)}"
    col_ref = exp.column(column_name, table=alias)
    agg = random.choice(["MAX", "MIN"])
    agg_expr: exp.Expression = exp.Anonymous(this=agg, expressions=[col_ref.copy()])
    query = (
        sqlglot.select(agg_expr)
        .from_(exp.to_table(table_name).as_(alias, copy=False), copy=False)
        .where(_simple_scalar_predicate(type_tag, col_ref, schema=schema, dialect=dialect, table_ids=table_ids), copy=False)
    )
    return exp.Subquery(this=query)


def _insert_having_clause(
    select_items: list[exp.Expression],
    expr_types: list[str],
    dialect: str,
    *,
    allow_plain_probe: bool,
    plain_probe_items: list[exp.Expression] | None = None,
) -> exp.Expression:
    typed_safe_exprs = [
        (expr, type_tag)
        for expr, type_tag in zip(select_items, expr_types)
        if type_tag in INSERT_SELECT_GROUP_SAFE_TYPES
    ]
    safe_exprs = [expr for expr, _ in typed_safe_exprs]
    if not safe_exprs:
        return exp.GTE(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(1))
    type_tag = random.choice([t for _, t in typed_safe_exprs] or ["numeric"])
    typed_exprs = [expr for expr, expr_type in typed_safe_exprs if expr_type == type_tag] or safe_exprs
    typed_plain_exprs = [
        expr for expr, expr_type in typed_safe_exprs
        if expr_type == type_tag and not _insert_expr_has_aggregate(expr)
    ] or [expr for expr in (plain_probe_items or []) if not _insert_expr_has_aggregate(expr)]
    monetdb_safe = dialect.lower().strip() == "monetdb"

    def typed_literal() -> exp.Expression:
        if type_tag == "numeric":
            return exp.Literal.number(random.randint(1, 1000))
        if type_tag == "date":
            return exp.Cast(this=exp.Literal.string(rand_date()), to=exp.DataType.build("DATE"))
        if type_tag == "datetime":
            return exp.Cast(this=exp.Literal.string(rand_datetime()), to=exp.DataType.build("DATETIME"))
        return exp.Literal.string(rand_str(3, 8))

    def operand() -> exp.Expression:
        probe = random.choice(typed_exprs).copy()
        plain_probe = random.choice(typed_plain_exprs).copy() if typed_plain_exprs else probe.copy()
        options: list[exp.Expression] = [plain_probe] if allow_plain_probe and not monetdb_safe and typed_plain_exprs else []
        options.extend([
            exp.Min(this=probe.copy()),
            exp.Max(this=probe.copy()),
        ])
        if type_tag == "numeric":
            options.extend([
                exp.Sum(this=probe.copy()),
                exp.Avg(this=probe.copy()),
                exp.Count(this=probe.copy()),
                exp.Count(this=exp.Distinct(expressions=[probe.copy()])),
            ])
        return random.choice(options)

    def aggregate_operand() -> exp.Expression:
        probe = random.choice(typed_exprs).copy()
        options: list[exp.Expression] = [
            exp.Min(this=probe.copy()),
            exp.Max(this=probe.copy()),
        ]
        if type_tag == "numeric":
            options.extend([
                exp.Sum(this=probe.copy()),
                exp.Avg(this=probe.copy()),
                exp.Count(this=probe.copy()),
                exp.Count(this=exp.Distinct(expressions=[probe.copy()])),
            ])
        return random.choice(options)

    def leaf() -> exp.Expression:
        if type_tag == "string":
            op = random.choice(["=", "<>", ">", "<", ">=", "<=", "LIKE", "IS", "IS NOT"])
        else:
            op = random.choice(["=", "<>", ">", "<", ">=", "<=", "BETWEEN", "IS", "IS NOT"])
        literal_compare = op in {"=", "<>", ">", "<", ">=", "<="} and random.random() < 0.35
        left = aggregate_operand() if literal_compare else operand()
        if op == "IS":
            return exp.Is(this=left, expression=exp.Null())
        if op == "IS NOT":
            return exp.Not(this=exp.Is(this=left, expression=exp.Null()))
        if op == "BETWEEN":
            return exp.Between(this=left, low=typed_literal(), high=typed_literal())
        if op == "LIKE":
            return exp.Like(this=left, expression=exp.Literal.string(f"%{rand_str(2, 4)}%"))
        return {
            "=": exp.EQ,
            "<>": exp.NEQ,
            ">": exp.GT,
            "<": exp.LT,
            ">=": exp.GTE,
            "<=": exp.LTE,
        }[op](this=left, expression=typed_literal() if literal_compare else operand())

    def aggregate_leaf() -> exp.Expression:
        if type_tag == "string":
            op = random.choice(["=", "<>", ">", "<", ">=", "<=", "LIKE", "IS", "IS NOT"])
        else:
            op = random.choice(["=", "<>", ">", "<", ">=", "<=", "BETWEEN", "IS", "IS NOT"])
        left = aggregate_operand()
        if op == "IS":
            return exp.Is(this=left, expression=exp.Null())
        if op == "IS NOT":
            return exp.Not(this=exp.Is(this=left, expression=exp.Null()))
        if op == "BETWEEN":
            return exp.Between(this=left, low=typed_literal(), high=typed_literal())
        if op == "LIKE":
            return exp.Like(this=left, expression=exp.Literal.string(f"%{rand_str(2, 4)}%"))
        return {
            "=": exp.EQ,
            "<>": exp.NEQ,
            ">": exp.GT,
            "<": exp.LT,
            ">=": exp.GTE,
            "<=": exp.LTE,
        }[op](this=left, expression=typed_literal() if random.random() < 0.45 else operand())

    def predicate(leaf_budget: int = 3, logic_depth: int = 0) -> exp.Expression:
        expr = aggregate_leaf()
        used_leaves = 1
        depth = logic_depth
        while used_leaves < leaf_budget and depth < 3:
            if random.random() >= (COND_NOT_PROB + COND_AND_PROB + COND_OR_PROB):
                break
            next_expr = leaf()
            if random.random() < COND_NOT_PROB:
                next_expr = exp.Not(this=next_expr)
            expr = exp.And(this=expr, expression=next_expr) if random.random() < 0.5 else exp.Or(this=expr, expression=next_expr)
            used_leaves += 1
            depth += 1
        return expr

    return predicate()


def _is_scalar_subquery(subquery: exp.Subquery) -> bool:
    parent = subquery.parent
    if isinstance(parent, exp.In) and subquery.arg_key == "query":
        return False
    if isinstance(parent, (exp.Any, exp.All)) and subquery.arg_key == "this":
        return False
    if isinstance(parent, exp.Exists) and subquery.arg_key == "this":
        return False
    if isinstance(parent, exp.SetOperation):
        return False
    if isinstance(parent, (exp.From, exp.Join, exp.CTE, exp.Lateral)):
        return False
    return True


def _select_is_in_scalar_subquery(select: exp.Select) -> bool:
    current: exp.Expression | None = select
    while current is not None:
        if isinstance(current, exp.Subquery):
            return _is_scalar_subquery(current)
        current = current.parent
    return False


def _select_projection_exprs(select: exp.Select) -> list[exp.Expression]:
    exprs: list[exp.Expression] = []
    for item in select.expressions or []:
        base = item.this if isinstance(item, exp.Alias) else item
        if isinstance(base, exp.Expression):
            exprs.append(base)
    return exprs


def _infer_insert_expr_type_tag(node: exp.Expression | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, exp.Alias):
        return _infer_insert_expr_type_tag(node.this)
    if isinstance(node, exp.Paren):
        return _infer_insert_expr_type_tag(node.this)
    if isinstance(node, exp.Cast):
        cast_to = node.args.get("to")
        if isinstance(cast_to, exp.DataType):
            cast_type_tag = _type_tag_for_col_type(cast_to.sql(dialect="mysql"))
            if cast_type_tag in INSERT_SELECT_GROUP_SAFE_TYPES:
                return cast_type_tag
    if isinstance(node, exp.Literal):
        return "string" if node.is_string else "numeric"
    if isinstance(node, (exp.Count, exp.Sum, exp.Avg)):
        return "numeric"
    if isinstance(node, (exp.Min, exp.Max)):
        return _infer_insert_expr_type_tag(node.this)
    if isinstance(node, (exp.Add, exp.Sub)):
        if any(isinstance(child, exp.Interval) for child in node.iter_expressions()):
            return "datetime"
        return "numeric"
    if isinstance(node, (exp.Mul, exp.Div, exp.Mod)):
        return "numeric"
    if isinstance(node, exp.Anonymous):
        name = (node.name or "").upper()
        if name in {"COUNT", "SUM", "AVG", "LENGTH", "CHAR_LENGTH", "DATEDIFF", "TIMESTAMPDIFF"}:
            return "numeric"
        if name in {"LOWER", "UPPER", "CONCAT", "SUBSTR", "SUBSTRING", "TRIM"}:
            return "string"
        if name in {"DATE", "CURDATE", "CURRENT_DATE"}:
            return "date"
        if name in {"NOW", "CURRENT_TIMESTAMP", "SYSDATE"}:
            return "datetime"
    cls = node.__class__.__name__.lower()
    if cls in {"lower", "upper", "concat", "substr", "substring", "trim"}:
        return "string"
    if cls in {"length", "charlength", "datediff", "timestampdiff", "year", "month", "day", "hour", "minute", "second"}:
        return "numeric"
    if cls in {"currentdate", "date"}:
        return "date"
    if cls in {"currenttimestamp", "timestamp", "now"}:
        return "datetime"
    return None


def _build_select_having_expr(select: exp.Select, dialect: str) -> exp.Expression:
    projection_exprs = _select_projection_exprs(select) or [exp.Literal.number(1)]
    expr_types = [(_infer_insert_expr_type_tag(expr) or "any") for expr in projection_exprs]
    plain_probe_items = [expr.copy() for expr in projection_exprs if not _insert_expr_has_aggregate(expr)]
    return _insert_having_clause(
        projection_exprs,
        expr_types,
        dialect,
        allow_plain_probe=True,
        plain_probe_items=plain_probe_items,
    )


def _ensure_group_for_select_having(select: exp.Select, dialect: str) -> None:
    probe_exprs: list[exp.Expression] = []
    probe_exprs.extend(_select_projection_exprs(select))
    where = select.args.get("where")
    if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression):
        probe_exprs.append(where.this)
    having = select.args.get("having")
    if isinstance(having, exp.Having) and isinstance(having.this, exp.Expression):
        probe_exprs.append(having.this)

    has_aggregate_context = _insert_exprs_have_aggregate(probe_exprs)
    group_items = _collect_insert_group_items(probe_exprs, dialect)
    if has_aggregate_context and group_items and select.args.get("group") is None:
        select.group_by(*[item.copy() for item in group_items], copy=False)


def _inject_random_having_on_selects(root: exp.Expression, dialect: str, *, prob: float) -> None:
    for select in list(root.find_all(exp.Select)):
        if _select_is_in_scalar_subquery(select):
            continue
        if select.args.get("having") is not None:
            continue
        if random.random() >= prob:
            continue
        select.set("having", exp.Having(this=_build_select_having_expr(select, dialect)))
        _ensure_group_for_select_having(select, dialect)


def _expr_has_column_ref(expr: exp.Expression) -> bool:
    return any(True for _ in expr.find_all(exp.Column))


def _collect_insert_group_columns(node: exp.Expression | None) -> list[exp.Column]:
    if node is None:
        return []
    # Nested subqueries have independent scopes; do not leak their aliases into
    # the current SELECT's GROUP BY list.
    if isinstance(node, (exp.Subquery, exp.Select, exp.CTE)):
        return []
    if isinstance(node, exp.Window) and isinstance(node.this, exp.AggFunc):
        return []
    if isinstance(node, exp.AggFunc):
        return []
    if isinstance(node, exp.Column):
        return [node.copy()]
    cols: list[exp.Column] = []
    for value in node.args.values():
        if isinstance(value, exp.Expression):
            if isinstance(value, (exp.Subquery, exp.Select, exp.CTE)):
                continue
            cols.extend(_collect_insert_group_columns(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, exp.Expression):
                    if isinstance(item, (exp.Subquery, exp.Select, exp.CTE)):
                        continue
                    cols.extend(_collect_insert_group_columns(item))
    return cols


def _collect_insert_group_items(
    exprs: list[exp.Expression],
    dialect: str,
) -> list[exp.Expression]:
    seen: set[str] = set()
    group_items: list[exp.Expression] = []
    for expr in exprs:
        for col in _collect_insert_group_columns(expr):
            key = col.sql(dialect=_sqlglot_dialect(dialect))
            if key in seen:
                continue
            seen.add(key)
            group_items.append(col)
    return group_items


def _insert_expr_has_aggregate(node: exp.Expression | None) -> bool:
    if node is None:
        return False
    if isinstance(node, exp.Window) and isinstance(node.this, exp.AggFunc):
        return True
    if isinstance(node, exp.AggFunc):
        return True
    return any(_insert_expr_has_aggregate(child) for child in node.iter_expressions())


def _insert_exprs_have_aggregate(exprs: list[exp.Expression]) -> bool:
    return any(_insert_expr_has_aggregate(expr) for expr in exprs)


def _apply_insert_group_by_items(select_ast: exp.Select, group_items: list[exp.Expression]) -> exp.Select:
    if not group_items:
        return select_ast
    return select_ast.group_by(*[item.copy() for item in group_items], copy=False)


def _insert_ignore(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_insert_ignore", True) and random.random() < 0.25)


def _duckdb_conflict_assignment_sql(schema: dict, table: str, col: str) -> str:
    col_ref = col
    excluded_ref = f"EXCLUDED.{col}"
    type_tag = _type_tag_for_col_type(schema[table]["columns"].get(col))
    if type_tag == "numeric":
        options = [
            f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
            f"THEN COALESCE(TRY_CAST({excluded_ref} AS BIGINT), {excluded_ref}, {col_ref}) + 1 ELSE {col_ref} END",
            f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
            f"THEN ABS(COALESCE({excluded_ref}, {col_ref})) ELSE {col_ref} END",
        ]
        return random.choice(options)
    if type_tag == "string":
        return (
            f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
            f"THEN COALESCE(SUBSTRING(LOWER(CAST({excluded_ref} AS VARCHAR)), 1, 64), {col_ref}) ELSE {col_ref} END"
        )
    if type_tag == "date":
        return (
            f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
            f"THEN COALESCE(CAST({excluded_ref} AS DATE), {col_ref}) ELSE {col_ref} END"
        )
    if type_tag == "datetime":
        return (
            f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
            f"THEN COALESCE(CAST({excluded_ref} AS TIMESTAMP), {col_ref}) ELSE {col_ref} END"
        )
    return (
        f"{col} = CASE WHEN {excluded_ref} IS DISTINCT FROM {col_ref} "
        f"THEN COALESCE({excluded_ref}, {col_ref}) ELSE {col_ref} END"
    )


def _duckdb_insert_conflict(schema: dict, table: str) -> exp.OnConflict:
    pk = schema[table]["pk"]
    cols = [c for c in schema[table]["columns"].keys() if c != pk]
    if not cols or random.random() < DUCKDB_CONFLICT_DO_NOTHING_PROB:
        stmt = sqlglot.parse_one(
            f"INSERT INTO {table} ({pk}) VALUES (1) ON CONFLICT ({pk}) DO NOTHING",
            read="duckdb",
        )
        return stmt.args["conflict"].copy()

    chosen = random.sample(cols, k=random.randint(1, min(2, len(cols))))
    if random.random() < DUCKDB_CONFLICT_COMPLEX_EXPR_PROB:
        assignments = ", ".join(_duckdb_conflict_assignment_sql(schema, table, col) for col in chosen)
    else:
        assignments = ", ".join(f"{col} = EXCLUDED.{col}" for col in chosen)
    stmt = sqlglot.parse_one(
        f"INSERT INTO {table} ({pk}) VALUES (1) ON CONFLICT ({pk}) DO UPDATE SET {assignments}",
        read="duckdb",
    )
    return stmt.args["conflict"].copy()


def _build_insert_stmt(
    schema: dict,
    table: str,
    cols: list[str],
    expression: exp.Expression,
    dialect: str,
    *,
    ignore: bool = False,
    replace: bool = False,
    upsert: bool = False,
    by_name: bool = False,
    returning: exp.Returning | None = None,
) -> exp.Insert:
    kwargs: dict = {
        "this": exp.to_table(table) if by_name else _target_schema(table, cols),
        "expression": expression,
    }
    if dialect == "duckdb":
        if by_name:
            kwargs["by_name"] = True
        if replace:
            kwargs["alternative"] = "REPLACE"
        elif ignore:
            kwargs["alternative"] = "IGNORE"
        if upsert:
            kwargs["conflict"] = _duckdb_insert_conflict(schema, table)
        if returning is not None:
            kwargs["returning"] = returning
    else:
        kwargs["ignore"] = ignore
        if upsert:
            kwargs["conflict"] = _upsert_conflict(schema, table)
    return exp.Insert(**kwargs)


def _maybe_insert_returning(schema: dict, table: str, dialect: str) -> exp.Returning | None:
    if not dialect_option(dialect, "supports_dml_returning", False):
        return None
    if random.random() >= INSERT_RETURNING_PROB:
        return None
    cols = list(schema[table]["columns"].keys())
    chosen = [schema[table]["pk"]]
    extras = [col for col in cols if col != chosen[0]]
    if extras and random.random() < 0.35:
        chosen.append(random.choice(extras))
    return exp.Returning(expressions=[exp.column(col) for col in chosen])


def _alias_projection_expr(expr: exp.Expression, alias: str) -> exp.Expression:
    base = expr.this.copy() if isinstance(expr, exp.Alias) else expr.copy()
    return base.as_(alias, copy=False)


def _query_with_output_aliases(query: exp.Expression, cols: list[str]) -> exp.Select:
    if isinstance(query, exp.Select):
        wrapped = query.copy()
        wrapped.set(
            "expressions",
            [_alias_projection_expr(expr, col) for expr, col in zip(wrapped.expressions, cols, strict=False)],
        )
        return wrapped

    alias = f"ins_wrap_{rand_str(3, 5)}"
    return sqlglot.select(
        *[exp.column(col, table=alias).as_(col, copy=False) for col in cols]
    ).from_(query.copy().subquery(alias), copy=False)


def _duckdb_by_name_projection_cols(schema: dict, table: str, cols: list[str]) -> list[str]:
    pk = schema[table]["pk"]
    required = [col for col in cols if col == pk or not column_nullable(schema, table, col)]
    optional = [col for col in cols if col not in required]
    chosen = required.copy()
    if optional:
        if random.random() < DUCKDB_BY_NAME_SUBSET_PROB:
            keep = random.randint(0, len(optional))
            if keep:
                chosen.extend(random.sample(optional, k=keep))
        else:
            chosen.extend(optional)
    if len(chosen) > 1 and random.random() < 0.75:
        random.shuffle(chosen)
    return chosen


def _by_name_insert_expression(
    query: exp.Expression,
    cols: list[str],
    *,
    schema: dict | None = None,
    table: str | None = None,
    dialect: str | None = None,
) -> exp.Subquery:
    wrapped = _query_with_output_aliases(query, cols)
    alias = f"ins_by_name_{rand_str(3, 5)}"
    projection_cols = cols
    if dialect == "duckdb" and schema is not None and table is not None:
        projection_cols = _duckdb_by_name_projection_cols(schema, table, cols)
    return exp.Subquery(
        this=sqlglot.select(
            *[exp.column(col, table=alias).as_(col, copy=False) for col in projection_cols]
        ).from_(wrapped.subquery(alias), copy=False)
    )


def _wrap_insert_top_level_from_subquery(query: exp.Expression, cols: list[str]) -> exp.Select:
    alias = f"ins_top_{rand_str(3, 5)}"
    return sqlglot.select(
        *[exp.column(col, table=alias) for col in cols]
    ).from_(_query_with_output_aliases(query, cols).subquery(alias), copy=False)


def _is_intersect_safe_insert_query(query: exp.Select) -> bool:
    return query.args.get("distinct") is not None or query.args.get("group") is not None


def _available_insert_top_level_setops(dialect: str, left: exp.Select) -> list[str]:
    options = ["UNION_ALL", "UNION"]
    if dialect_option(dialect, "supports_except", True):
        options.append("EXCEPT")
    if dialect_option(dialect, "supports_intersect", True) and _is_intersect_safe_insert_query(left):
        options.append("INTERSECT")
    return options


def _candidate_insert_row(schema: dict, table: str, table_ids: dict, dialect: str) -> dict:
    pk_base = max(table_ids[table]) if table_ids.get(table) else 0
    pk = pk_base + random.randint(1, 3)
    return build_row(schema, table, pk, table_ids, dialect)


def _maybe_attach_insert_select_cte(
    select_ast: exp.Select,
    *,
    schema: dict,
    table: str,
    table_ids: dict,
    dialect: str,
    allow_with: bool,
) -> exp.Select:
    if not allow_with or not dialect_option(dialect, "supports_dml_with", True) or random.random() >= INSERT_SELECT_CTE_PROB:
        return select_ast
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
    cte_name = f"cte_ins_{rand_str(3, 5)}"
    cte_select = sqlglot.select(exp.column("k", table="cte_sub").as_("c1")).from_(_parse_query(cte_stmt, dialect).subquery("cte_sub"))
    select_ast = select_ast.with_(cte_name, as_=cte_select, copy=False)
    return select_ast.where(
        exp.Exists(this=sqlglot.select(exp.Literal.number(1)).from_(exp.to_table(cte_name))),
        copy=False,
    )


def _maybe_group_insert_select(
    select_ast: exp.Select,
    *,
    select_items: list[exp.Expression],
    expr_types: list[str],
    dialect: str,
) -> exp.Select:
    supports_having = bool(dialect_option(dialect, "supports_select_having", True))

    safe_having = any(type_tag in INSERT_SELECT_GROUP_SAFE_TYPES for type_tag in expr_types)
    want_having = bool(supports_having and safe_having and random.random() < INSERT_SELECT_HAVING_PROB)

    group_probe_exprs = [expr.copy() for expr in select_items]
    where = select_ast.args.get("where")
    if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression):
        group_probe_exprs.append(where.this.copy())
    plain_probe_items = [expr.copy() for expr in select_items if not _insert_expr_has_aggregate(expr)]
    having_expr: exp.Expression | None = None
    if want_having:
        having_expr = _insert_having_clause(
            select_items,
            expr_types,
            dialect,
            allow_plain_probe=True,
            plain_probe_items=plain_probe_items,
        )
        group_probe_exprs.append(having_expr.copy())
    has_aggregate_context = _insert_exprs_have_aggregate(group_probe_exprs)
    group_items = _collect_insert_group_items(group_probe_exprs, dialect)
    has_group = False
    if has_aggregate_context and group_items:
        select_ast = _apply_insert_group_by_items(select_ast, group_items)
        has_group = True

    if not want_having:
        return select_ast

    return select_ast.having(
        having_expr,
        dialect=_sqlglot_dialect(dialect),
        copy=False,
    )


def _independent_insert_setop_branch(
    schema: dict,
    table: str,
    cols: list[str],
    table_ids: dict,
    dialect: str,
) -> exp.Select:
    row = _candidate_insert_row(schema, table, table_ids, dialect)
    expr_types = [_type_tag_for_col_type(schema[table]["columns"].get(col)) for col in cols]
    source = _insert_select_source(schema, table, table_ids, dialect)
    source, stable_pk_expr = _wrap_source_with_stable_pk(source, dialect)
    pk_col = schema[table]["pk"]
    pk_offset = _setop_right_pk_offset(table_ids, table)
    select_items = []
    for col in cols:
        if col == pk_col:
            select_items.append(
                _fresh_insert_pk_expr_with_offset(
                    row[col],
                    stable_pk_expr,
                    pk_offset,
                    dialect,
                    target_row_count=len(table_ids.get(table, [])),
                )
            )
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
    return select_ast


def _wrap_insert_top_level_setop(
    query: exp.Expression,
    cols: list[str],
    dialect: str,
    *,
    schema: dict,
    target_table: str,
    table_ids: dict,
) -> exp.Expression:
    left = _query_with_output_aliases(query, cols)
    right = _query_with_output_aliases(
        _independent_insert_setop_branch(schema, target_table, cols, table_ids, dialect),
        cols,
    )
    setop_kind = random.choice(_available_insert_top_level_setops(dialect, left))
    if setop_kind == "UNION_ALL":
        return exp.Union(this=left, expression=right, distinct=False)
    if setop_kind == "UNION":
        return exp.Union(this=left, expression=right, distinct=True)
    if setop_kind == "EXCEPT":
        return exp.Except(this=left, expression=right)
    return exp.Intersect(this=left, expression=right)


def _wrap_insert_top_level_cte(query: exp.Expression, cols: list[str]) -> exp.Select:
    cte_name = f"cte_ins_top_{rand_str(3, 5)}"
    alias = f"ins_cte_{rand_str(3, 5)}"
    outer = sqlglot.select(
        *[exp.column(col, table=alias) for col in cols]
    ).from_(exp.to_table(cte_name).as_(alias, copy=False), copy=False)
    return outer.with_(cte_name, as_=_query_with_output_aliases(query, cols), copy=False)


def _query_has_with(query: exp.Expression) -> bool:
    return bool(isinstance(query, exp.Expression) and query.args.get("with_") is not None)


def _maybe_apply_insert_statement_with(
    query: exp.Expression,
    cols: list[str],
    dialect: str,
) -> exp.Expression:
    if not dialect_option(dialect, "supports_dml_with", True):
        return query
    if random.random() >= INSERT_TOPLEVEL_CTE_PROB:
        return query
    if _query_has_with(query):
        return query
    return _wrap_insert_top_level_cte(query, cols)


def _apply_insert_top_level_wrappers(
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


def _limit_insert_target_rows(query: exp.Expression, *, dialect: str) -> exp.Expression:
    del dialect
    # Keep generation-side SELECTs free of LIMIT unless LIMIT is explicitly
    # introduced inside scalar subqueries by scalarization logic.
    return query


def _next_insert_row(schema: dict, table_ids: dict, dialect: str) -> tuple[str, list[str], dict]:
    table = random.choice(list(schema.keys()))
    cols = list(schema[table]["columns"].keys())
    pk_base = max(table_ids[table]) if table_ids.get(table) else 0
    pk = pk_base + random.randint(1, 3)
    row = build_row(schema, table, pk, table_ids, dialect)
    table_ids[table].append(pk)
    return table, cols, row


def _existing_insert_row(schema: dict, table_ids: dict, dialect: str) -> tuple[str, list[str], dict]:
    candidates = [table for table in schema if table_ids.get(table)]
    table = random.choice(candidates or list(schema.keys()))
    cols = list(schema[table]["columns"].keys())
    pk = random.choice(table_ids[table])
    row = build_row(schema, table, pk, table_ids, dialect)
    return table, cols, row


def _duckdb_insert_row_count(max_rows: int, *, upsert: bool = False) -> int:
    batch_prob = DUCKDB_UPSERT_BATCH_PROB if upsert else DUCKDB_INSERT_MULTI_VALUES_PROB
    if random.random() >= batch_prob:
        return 1
    return random.randint(2, max_rows)


def _fresh_insert_rows(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    row_count: int,
) -> tuple[str, list[str], list[dict]]:
    table = random.choice(list(schema.keys()))
    cols = list(schema[table]["columns"].keys())
    pk_col = schema[table]["pk"]
    table_ids.setdefault(table, [])
    next_pk = max(table_ids[table]) if table_ids[table] else 0
    rows: list[dict] = []
    new_pks: list[int] = []
    for _ in range(row_count):
        next_pk += random.randint(1, 3)
        row = build_row(schema, table, next_pk, table_ids, dialect)
        rows.append(row)
        new_pks.append(int(row[pk_col]))
    table_ids[table].extend(new_pks)
    return table, cols, rows


def _upsert_insert_rows(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    row_count: int,
) -> tuple[str, list[str], list[dict]]:
    candidates = [table for table in schema if table_ids.get(table)]
    table = random.choice(candidates or list(schema.keys()))
    cols = list(schema[table]["columns"].keys())
    pk_col = schema[table]["pk"]
    existing_pks = list(table_ids.get(table, []))
    next_pk = max(existing_pks) if existing_pks else 0
    rows: list[dict] = []
    new_pks: list[int] = []
    if existing_pks:
        duplicate_pk = random.choice(existing_pks)
        rows.append(build_row(schema, table, duplicate_pk, table_ids, dialect))
        if row_count > 1:
            rows.append(build_row(schema, table, duplicate_pk, table_ids, dialect))
    while len(rows) < row_count:
        use_existing = bool(existing_pks) and random.random() < 0.6
        if use_existing:
            pk = random.choice(existing_pks)
        else:
            next_pk += random.randint(1, 3)
            pk = next_pk
            new_pks.append(pk)
        rows.append(build_row(schema, table, pk, table_ids, dialect))
    if new_pks:
        table_ids.setdefault(table, []).extend(new_pks)
    return table, cols, rows


def _insert_values_ast(cols: list[str], rows: list[dict], dialect: str) -> exp.Values:
    return exp.Values(
        expressions=[
            exp.Tuple(expressions=[_parse_expr(row[col], dialect) for col in cols])
            for row in rows
        ]
    )


def _upsert_conflict(schema: dict, table: str) -> exp.OnConflict:
    cols = [c for c in schema[table]["columns"].keys() if c != schema[table]["pk"]]
    if not cols:
        cols = [schema[table]["pk"]]
    chosen = random.sample(cols, k=random.randint(1, min(2, len(cols))))
    assignments = [
        exp.EQ(
            this=exp.column(col),
            expression=exp.Anonymous(this="VALUES", expressions=[exp.to_identifier(col)]),
        )
        for col in chosen
    ]
    return exp.OnConflict(
        duplicate=True,
        action=exp.Var(this="UPDATE"),
        expressions=assignments,
    )


def _insert_values(
    schema: dict,
    table_ids: dict,
    dialect: str,
    *,
    replace: bool = False,
) -> str:
    row_count = 1
    if dialect == "duckdb":
        row_count = _duckdb_insert_row_count(DUCKDB_INSERT_MULTI_VALUES_MAX_ROWS)
    table, cols, rows = _fresh_insert_rows(schema, table_ids, dialect, row_count=row_count)
    values = _insert_values_ast(cols, rows, dialect)
    stmt = _build_insert_stmt(
        schema,
        table,
        cols,
        values,
        dialect,
        ignore=(not replace and _insert_ignore(dialect)),
        replace=replace,
        returning=_maybe_insert_returning(schema, table, dialect),
    )
    _inject_random_having_on_selects(stmt, dialect, prob=INSERT_SELECT_HAVING_PROB)
    return _render_sql(stmt, dialect) + ";"


def _insert_on_duplicate_values(schema: dict, table_ids: dict, dialect: str) -> str:
    row_count = 1
    if dialect == "duckdb":
        row_count = _duckdb_insert_row_count(DUCKDB_UPSERT_BATCH_MAX_ROWS, upsert=True)
    table, cols, rows = _upsert_insert_rows(schema, table_ids, dialect, row_count=row_count)
    values = _insert_values_ast(cols, rows, dialect)
    stmt = _build_insert_stmt(
        schema,
        table,
        cols,
        values,
        dialect,
        upsert=True,
        returning=_maybe_insert_returning(schema, table, dialect),
    )
    _inject_random_having_on_selects(stmt, dialect, prob=INSERT_SELECT_HAVING_PROB)
    return _render_sql(stmt, dialect) + ";"


def _typed_source_ref_sql(
    schema: dict,
    target_table: str,
    target_col: str,
    source: QuerySource,
    source_col: str,
    dialect: str,
) -> str:
    if not source.source_alias or not source.source_table:
        raise ValueError("source alias/table is required")
    ref_sql = _render_sql(exp.column(source_col, table=source.source_alias), dialect)
    target_type = _type_tag_for_col_type(schema[target_table]["columns"].get(target_col))
    source_type = _type_tag_for_col_type(schema[source.source_table]["columns"].get(source_col))
    if target_type == source_type:
        return ref_sql
    if target_type == "date" and source_type == "datetime":
        return f"CAST({ref_sql} AS DATE)"
    if target_type == "datetime" and source_type == "date":
        return f"CAST({ref_sql} AS DATETIME)"
    return ref_sql


def _compatible_source_columns(schema: dict, target_table: str, target_col: str, source: QuerySource) -> list[str]:
    if not source.source_table or not source.source_alias:
        return []
    col_type = schema[target_table]["columns"].get(target_col)
    type_tag = _type_tag_for_col_type(col_type)
    if type_tag in {"json"} or col_type in {"bin", "geom"}:
        return []
    return [
        source_col
        for source_col, source_col_type in schema[source.source_table]["columns"].items()
        if source_col_type not in {"bin", "geom", "json"}
        and _type_tag_for_col_type(source_col_type) == type_tag
    ]


def _insert_source_value_expr(
    schema: dict,
    target_table: str,
    target_col: str,
    row_value: str,
    source: QuerySource,
    dialect: str,
) -> exp.Expression | None:
    compatible_cols = _compatible_source_columns(schema, target_table, target_col, source)
    if not compatible_cols:
        return None

    source_col = random.choice(compatible_cols)
    ref_sql = _typed_source_ref_sql(schema, target_table, target_col, source, source_col, dialect)
    alt_cols = [col for col in compatible_cols if col != source_col]
    alt_ref_sql = None
    if alt_cols:
        alt_ref_sql = _typed_source_ref_sql(
            schema,
            target_table,
            target_col,
            source,
            random.choice(alt_cols),
            dialect,
        )
    col_type = schema[target_table]["columns"].get(target_col)
    target_type = _type_tag_for_col_type(col_type)
    target_meta = schema[target_table].get("column_meta", {}).get(target_col, {})

    if target_type == "numeric":
        options = [ref_sql]
        if alt_ref_sql is not None:
            options.append(f"COALESCE({ref_sql}, {alt_ref_sql})")
        if col_type not in {"bool", "year"}:
            options.extend([
                f"ABS({ref_sql})",
                f"{ref_sql} + {random.randint(1, 9)}",
            ])
            if alt_ref_sql is not None:
                options.append(f"ABS(COALESCE({ref_sql}, {alt_ref_sql}))")
        return _parse_expr(random.choice(options), dialect)

    if target_type == "string":
        suffix_sql = sql_str(rand_str(1, 3))
        concat_sql = f"CONCAT({ref_sql}, {suffix_sql})"
        max_len = target_meta.get("max_len")
        if max_len:
            concat_sql = f"SUBSTRING({concat_sql}, 1, {int(max_len)})"
        options = [ref_sql, f"UPPER({ref_sql})", f"LOWER({ref_sql})", concat_sql]
        if alt_ref_sql is not None:
            options.append(f"COALESCE({ref_sql}, {alt_ref_sql})")
        return _parse_expr(random.choice(options), dialect)

    if target_type in {"date", "datetime"}:
        options = [ref_sql, f"{ref_sql} + INTERVAL {random.randint(1, 10)} DAY"]
        if alt_ref_sql is not None:
            options.append(f"COALESCE({ref_sql}, {alt_ref_sql})")
        return _parse_expr(random.choice(options), dialect)

    return None


def _insert_scalar_subquery_expr(schema: dict, table: str, col: str, row_value: str, table_ids: dict, dialect: str) -> exp.Expression | None:
    del row_value
    col_type = schema[table]["columns"].get(col)
    type_tag = _type_tag_for_col_type(col_type)
    if type_tag not in INSERT_SELECT_SUBQUERY_TYPES:
        return None
    if col_type in {"bin", "geom"} or type_tag == "json":
        return None

    candidates: list[tuple[str, str]] = []
    for source_table, info in schema.items():
        for source_col, source_col_type in info["columns"].items():
            if source_col_type in {"bin", "geom", "json"}:
                continue
            if _type_tag_for_col_type(source_col_type) == type_tag:
                candidates.append((source_table, source_col))
    if not candidates:
        return None

    source_table, source_col = random.choice(candidates)
    return _simple_scalar_subquery(
        schema=schema,
        table_ids=table_ids,
        table_name=source_table,
        column_name=source_col,
        type_tag=type_tag,
        dialect=dialect,
    )


def _insert_select_expr(
    schema: dict,
    table: str,
    col: str,
    row_value: str,
    table_ids: dict,
    dialect: str,
    source: QuerySource | None = None,
) -> exp.Expression:
    source_expr = None
    if source is not None:
        source_expr = _insert_source_value_expr(schema, table, col, row_value, source, dialect)
    scalar_expr = _insert_scalar_subquery_expr(schema, table, col, row_value, table_ids, dialect)

    if source_expr is not None and random.random() < 0.7:
        return source_expr
    if scalar_expr is not None and random.random() < 0.6:
        return scalar_expr
    if source_expr is not None:
        return source_expr
    if scalar_expr is not None:
        return scalar_expr
    return _parse_expr(row_value, dialect)


def _pk_literal_expr(row_value: str, dialect: str) -> exp.Expression:
    try:
        return exp.Literal.number(int(row_value))
    except Exception:
        return _parse_expr(row_value, dialect)


def _duckdb_unique_pk_expr(
    row_value: str,
    stable_pk_expr: exp.Expression | None,
    dialect: str,
    *,
    offset: int = 0,
    target_row_count: int | None = None,
) -> exp.Expression:
    base = _pk_literal_expr(row_value, dialect)
    if offset > 0:
        base = exp.Add(this=base, expression=exp.Literal.number(offset))
    if stable_pk_expr is None:
        return base

    stable_pk_bigint = exp.Cast(this=stable_pk_expr.copy(), to=exp.DataType.build("BIGINT"))
    stride_seed = max(64, (target_row_count or 0) + 1)
    stride_expr = exp.Literal.number(stride_seed)

    # Avoid ROW_NUMBER-based PK generation so that unrelated row additions
    # do not globally renumber generated keys.
    dup_ordinal_expr = exp.Window(
        this=exp.Count(this=exp.Star()),
        partition_by=[stable_pk_expr.copy()],
        order=exp.Order(expressions=[_ordered(stable_pk_expr.copy(), "ASC")]),
        spec=exp.WindowSpec(
            kind="ROWS",
            start="UNBOUNDED",
            start_side="PRECEDING",
            end="CURRENT ROW",
        ),
        over="OVER",
    )
    dup_offset_expr = exp.Sub(this=dup_ordinal_expr, expression=exp.Literal.number(1))
    stable_pk_offset_expr = exp.Mul(this=stable_pk_bigint, expression=stride_expr)
    non_null_unique_expr = exp.Add(
        this=exp.Add(this=base.copy(), expression=stable_pk_offset_expr),
        expression=dup_offset_expr.copy(),
    )
    null_unique_expr = exp.Add(this=base.copy(), expression=dup_offset_expr.copy())
    unique_expr = exp.Case(
        ifs=[
            exp.If(
                this=exp.Is(this=stable_pk_expr.copy(), expression=exp.Null()),
                true=null_unique_expr,
            )
        ],
        default=non_null_unique_expr,
    )
    return exp.Coalesce(this=unique_expr, expressions=[base.copy()])


def _fresh_insert_pk_expr(
    row_value: str,
    stable_pk_expr: exp.Expression | None,
    dialect: str,
    *,
    target_row_count: int | None = None,
) -> exp.Expression:
    if dialect == "duckdb":
        return _duckdb_unique_pk_expr(
            row_value,
            stable_pk_expr,
            dialect,
            target_row_count=target_row_count,
        )
    base = _pk_literal_expr(row_value, dialect)
    if stable_pk_expr is None:
        return base
    return _stable_insert_pk_expr(row_value, stable_pk_expr, dialect)


def _stable_insert_pk_expr(row_value: str, stable_pk_expr: exp.Expression | None, dialect: str) -> exp.Expression:
    base = _parse_expr(row_value, dialect)
    if stable_pk_expr is None:
        return base
    return exp.Sub(
        this=exp.Add(this=base, expression=stable_pk_expr.copy()),
        expression=exp.Literal.number(1),
    )


def _setop_right_pk_offset(table_ids: dict, table: str) -> int:
    current_max = max(table_ids[table]) if table_ids.get(table) else 0
    return current_max + 64


def _stable_insert_pk_expr_with_offset(
    row_value: str,
    stable_pk_expr: exp.Expression | None,
    offset: int,
    dialect: str,
) -> exp.Expression:
    expr = _stable_insert_pk_expr(row_value, stable_pk_expr, dialect)
    if offset <= 0:
        return expr
    return exp.Add(this=expr, expression=exp.Literal.number(offset))


def _fresh_insert_pk_expr_with_offset(
    row_value: str,
    stable_pk_expr: exp.Expression | None,
    offset: int,
    dialect: str,
    *,
    target_row_count: int | None = None,
) -> exp.Expression:
    if dialect == "duckdb":
        return _duckdb_unique_pk_expr(
            row_value,
            stable_pk_expr,
            dialect,
            offset=offset,
            target_row_count=target_row_count,
        )
    return _stable_insert_pk_expr_with_offset(row_value, stable_pk_expr, offset, dialect)


def _target_pk_scalar_subquery_sql(table: str, pk_col: str, *, desc: bool) -> str:
    agg = "MAX" if desc else "MIN"
    return f"(SELECT {agg}({pk_col}) FROM {table} WHERE {pk_col} IS NOT NULL)"


def _upsert_insert_pk_expr(
    schema: dict,
    table: str,
    row_value: str,
    table_ids: dict,
    dialect: str,
) -> exp.Expression:
    fallback_literal = _pk_literal_expr(row_value, dialect)
    existing_pks = sorted({int(pk) for pk in table_ids.get(table, [])})
    if not existing_pks:
        return fallback_literal

    existing_pk = random.choice(existing_pks)
    if random.random() >= DUCKDB_UPSERT_SELECT_EXISTING_PK_PROB:
        return fallback_literal

    pk_col = schema[table]["pk"]
    pk_subquery_sql = _target_pk_scalar_subquery_sql(table, pk_col, desc=(random.random() < 0.5))
    candidates = [str(existing_pk)]
    if random.random() < DUCKDB_PK_SUBQUERY_EXPR_PROB:
        candidates.append(f"COALESCE({pk_subquery_sql}, {existing_pk})")
    if dialect == "duckdb":
        if random.random() < DUCKDB_PK_SUBQUERY_EXPR_PROB:
            candidates.extend([
                f"COALESCE(TRY_CAST({pk_subquery_sql} AS BIGINT), {existing_pk})",
                f"CASE WHEN {pk_subquery_sql} IS NOT NULL THEN {pk_subquery_sql} ELSE {existing_pk} END",
            ])
        candidates.append(
            f"CASE WHEN EXISTS(SELECT 1 FROM {table} WHERE {pk_col} = {existing_pk}) THEN {existing_pk} ELSE {int(row_value)} END"
        )
    return _parse_expr(random.choice(candidates), dialect)


def _wrap_source_with_stable_pk(source: QuerySource, dialect: str) -> tuple[QuerySource, exp.Expression | None]:
    del dialect
    return source, source.stable_pk_expr.copy() if source.stable_pk_expr is not None else None


def _identity_filter_expr(schema: dict, table: str, alias: str, table_ids: dict, dialect: str) -> exp.Expression:
    candidates = [
        col
        for col, col_type in schema[table]["columns"].items()
        if col_type not in {"bin", "geom", "json"}
    ] or list(schema[table]["columns"].keys())
    probe_col = random.choice(candidates)
    probe_ref = exp.column(probe_col, table=alias)
    type_tag = _type_tag_for_col_type(schema[table]["columns"].get(probe_col))
    return _simple_typed_predicate(type_tag, probe_ref, schema=schema, dialect=dialect, table_ids=table_ids)


def _join_comparison_expr(
    left_ref: exp.Expression,
    right_ref: exp.Expression,
    type_tag: str,
) -> exp.Expression:
    op = _join_key_op(type_tag)
    cls = {
        "=": exp.EQ,
        "<>": exp.NEQ,
        ">": exp.GT,
        "<": exp.LT,
        ">=": exp.GTE,
        "<=": exp.LTE,
    }[op]
    return cls(this=left_ref.copy(), expression=right_ref.copy())


def _combine_predicates_with_and(predicates: list[exp.Expression]) -> exp.Expression | None:
    items = [pred.copy() for pred in predicates if isinstance(pred, exp.Expression)]
    if not items:
        return None
    current = items[0]
    for pred in items[1:]:
        current = exp.And(this=current, expression=pred)
    return current


def _compatible_join_column_pairs(
    schema: dict,
    left_table: str,
    right_table: str,
) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for left_col, left_col_type in schema[left_table]["columns"].items():
        if left_col_type in {"bin", "geom", "json"}:
            continue
        type_tag = _type_tag_for_col_type(left_col_type)
        if type_tag not in INSERT_SELECT_SOURCE_TYPES:
            continue
        for right_col, right_col_type in schema[right_table]["columns"].items():
            if right_col_type in {"bin", "geom", "json"}:
                continue
            if _type_tag_for_col_type(right_col_type) == type_tag:
                pairs.append((left_col, right_col, type_tag))
    return pairs


def _all_column_projection(schema: dict, table: str, alias: str) -> list[exp.Expression]:
    return [
        exp.column(col, table=alias).as_(col, copy=False)
        for col in schema[table]["columns"].keys()
    ]


def _lateral_from_node(schema: dict, rhs_table: str, inner_alias: str) -> exp.Expression:
    base_alias = f"{inner_alias}_base"
    base_table = exp.to_table(rhs_table).as_(base_alias, copy=False)
    if random.random() < 0.5:
        return exp.to_table(rhs_table).as_(inner_alias, copy=False)

    base_query = sqlglot.select(
        *_all_column_projection(schema, rhs_table, base_alias)
    ).from_(base_table, copy=False)
    if random.random() < 0.3:
        base_query.set("distinct", exp.Distinct())
    return base_query.subquery(inner_alias)


def _lateral_select_items(schema: dict, rhs_table: str, inner_alias: str) -> list[exp.Expression]:
    if random.random() < 0.5:
        return [exp.Star()]
    return _all_column_projection(schema, rhs_table, inner_alias)


def _lateral_order_exprs(schema: dict, rhs_table: str, inner_alias: str) -> list[exp.Expression]:
    candidates = [
        col
        for col, col_type in schema[rhs_table]["columns"].items()
        if col_type not in {"bin", "geom", "json"}
    ] or list(schema[rhs_table]["columns"].keys())
    random.shuffle(candidates)
    selected = candidates[: random.randint(1, min(2, len(candidates)))]
    return [
        _ordered(exp.column(col, table=inner_alias), random.choice(["ASC", "DESC"]))
        for col in selected
    ]


def _build_lateral_query(
    schema: dict,
    rhs_table: str,
    inner_alias: str,
    correlated_predicate: exp.Expression,
    table_ids: dict,
    dialect: str,
) -> exp.Select:
    from_node = _lateral_from_node(schema, rhs_table, inner_alias)
    effective_alias = from_node.alias_or_name if isinstance(from_node, exp.Expression) else inner_alias
    effective_alias = effective_alias or inner_alias
    select_items = _lateral_select_items(schema, rhs_table, effective_alias)
    lateral_query = sqlglot.select(*select_items).from_(from_node, copy=False)

    predicates = [correlated_predicate]
    if random.random() < 0.45:
        predicates.append(_identity_filter_expr(schema, rhs_table, effective_alias, table_ids, dialect))
    if random.random() < 0.3:
        predicates.append(
            _parse_expr(f"{effective_alias}.{schema[rhs_table]['pk']} IS NOT NULL", dialect)
        )
    where_expr = _combine_predicates_with_and(predicates)
    if where_expr is not None:
        lateral_query = lateral_query.where(where_expr, copy=False)

    if random.random() < 0.25:
        lateral_query.set("distinct", exp.Distinct())
    if random.random() < 0.35:
        lateral_query.set("order", exp.Order(expressions=_lateral_order_exprs(schema, rhs_table, effective_alias)))
    return lateral_query


def _anchor_source_clause(schema: dict, table_ids: dict, dialect: str) -> QuerySource:
    source_table = random.choice(list(schema.keys()))
    alias = f"ins_src_{random.randint(100, 999)}"
    source = QuerySource(
        from_expression=exp.From(this=exp.to_table(source_table).as_(alias, copy=False)),
        joins=[],
        filters=[],
        limit=None,
        row_order_exprs=None,
        stable_pk_expr=exp.column(schema[source_table]["pk"], table=alias),
        source_table=source_table,
        source_alias=alias,
    )
    if random.random() < 0.45:
        source.filters.append(_identity_filter_expr(schema, source_table, alias, table_ids, dialect))
    return source


def _single_source_clause(schema: dict, table: str, table_ids: dict, dialect: str) -> QuerySource:
    del table
    return _anchor_source_clause(schema, table_ids, dialect)


def _window_source_clause(schema: dict, table: str, table_ids: dict, dialect: str) -> QuerySource:
    del table
    return _anchor_source_clause(schema, table_ids, dialect)


def _join_source_clause(schema: dict, table: str, table_ids: dict, dialect: str) -> QuerySource:
    source = _anchor_source_clause(schema, table_ids, dialect)
    if not source.source_table or not source.source_alias:
        return source

    rhs_candidates = list(schema.keys())
    random.shuffle(rhs_candidates)
    rhs_table = None
    join_pairs: list[tuple[str, str, str]] = []
    for candidate in rhs_candidates:
        pairs = _compatible_join_column_pairs(schema, source.source_table, candidate)
        if pairs:
            rhs_table = candidate
            join_pairs = pairs
            break
    if rhs_table is None:
        return source

    outer_col, inner_col, type_tag = random.choice(join_pairs)
    outer_ref = exp.column(outer_col, table=source.source_alias)

    if dialect == "duckdb":
        join_alias = f"ins_join_{random.randint(100, 999)}"
        inner_alias = f"{join_alias}_src"
        lateral_query = _build_lateral_query(
            schema,
            rhs_table,
            inner_alias,
            _join_comparison_expr(
                exp.column(inner_col, table=inner_alias),
                outer_ref,
                type_tag,
            ),
            table_ids,
            dialect,
        )
        join = exp.Join(
            this=exp.Lateral(
                this=exp.Subquery(this=lateral_query),
                alias=exp.TableAlias(this=exp.to_identifier(join_alias)),
            ),
            side="LEFT",
            on=_true_expr(dialect),
        )
        source.joins.append(join)
        source.filters.append(
            _parse_expr(f"{join_alias}.{schema[rhs_table]['pk']} IS NOT NULL", dialect)
        )
        _bind_query_source_side(
            source,
            table=rhs_table,
            alias=join_alias,
            schema=schema,
        )
        return source

    join_alias = f"ins_join_{random.randint(100, 999)}"
    right_ref = exp.column(inner_col, table=join_alias)
    join = exp.Join(
        this=exp.to_table(rhs_table).as_(join_alias, copy=False),
        on=_join_comparison_expr(outer_ref, right_ref, type_tag),
    )
    join_type = _join_type_with_on(dialect).upper()
    if join_type.startswith("LEFT"):
        join.set("side", "LEFT")
    elif join_type.startswith("RIGHT"):
        join.set("side", "RIGHT")
    elif join_type.startswith("INNER"):
        join.set("kind", "INNER")
    source.joins.append(join)
    # Keep payload columns and synthesized target PK on the same side.
    # Otherwise join widening can preserve the same target PK while pulling
    # different payload rows from the opposite side, which makes insert
    # approximation unsound under INSERT IGNORE / duplicate-key handling.
    if join_type.startswith("RIGHT"):
        _bind_query_source_side(
            source,
            table=rhs_table,
            alias=join_alias,
            schema=schema,
        )
    return source


def _exists_source_clause(schema: dict, table: str, table_ids: dict, dialect: str) -> QuerySource:
    source = _anchor_source_clause(schema, table_ids, dialect)
    type_tag = random.choice(INSERT_SELECT_SOURCE_TYPES)
    stmt = _select_stmt_for_setop(
        schema,
        table_ids,
        dialect,
        type_tag,
        depth=2,
        target_table=table,
        protected_tables={table},
        column_count=random.randint(1, 3),
    )
    alias = f"ins_filter_{random.randint(100, 999)}"
    query = _parse_query(stmt, dialect)
    exists_query = sqlglot.select(exp.Literal.number(1)).from_(query.subquery(alias))
    source.filters.append(exp.Exists(this=exists_query))
    return source


def _insert_select_source(schema: dict, table: str, table_ids: dict, dialect: str) -> QuerySource:
    if random.random() >= INSERT_SELECT_COMPLEX_PROB:
        return _single_source_clause(schema, table, table_ids, dialect)

    r = random.random()
    if dialect_option(dialect, "supports_window_functions", False) and r < INSERT_SELECT_WINDOW_PROB:
        return _window_source_clause(schema, table, table_ids, dialect)
    if r < 0.4:
        return _join_source_clause(schema, table, table_ids, dialect)
    if r < 0.75:
        return _single_source_clause(schema, table, table_ids, dialect)
    return _exists_source_clause(schema, table, table_ids, dialect)


def _insert_select(
    schema: dict,
    table_ids: dict,
    dialect: str,
    upsert: bool = False,
    *,
    replace: bool = False,
) -> str:
    if upsert:
        table, cols, row = _existing_insert_row(schema, table_ids, dialect)
        ignore = False
    else:
        table, cols, row = _next_insert_row(schema, table_ids, dialect)
        ignore = (not replace) and _insert_ignore(dialect)
    expr_types = [_type_tag_for_col_type(schema[table]["columns"].get(col)) for col in cols]
    if upsert:
        source = QuerySource(from_expression=exp.From(this=_seed_subquery("ins_seed")), joins=[], filters=[], limit=None)
    else:
        source = _insert_select_source(schema, table, table_ids, dialect)
        source, stable_pk_expr = _wrap_source_with_stable_pk(source, dialect)
    if upsert:
        stable_pk_expr = None
    pk_col = schema[table]["pk"]
    select_items = []
    for col in cols:
        if col == pk_col:
            if upsert:
                select_items.append(_upsert_insert_pk_expr(schema, table, row[col], table_ids, dialect))
            else:
                select_items.append(
                    _fresh_insert_pk_expr(
                        row[col],
                        stable_pk_expr,
                        dialect,
                        target_row_count=len(table_ids.get(table, [])),
                    )
                )
            continue
        select_items.append(_insert_select_expr(schema, table, col, row[col], table_ids, dialect, source))

    select_ast = sqlglot.select(*select_items)
    if source.from_expression is not None:
        select_ast.set("from_", source.from_expression)
    if source.joins:
        select_ast.set("joins", source.joins)
    for filt in source.filters:
        select_ast = select_ast.where(filt, copy=False)

    select_ast = _maybe_attach_insert_select_cte(
        select_ast,
        schema=schema,
        table=table,
        table_ids=table_ids,
        dialect=dialect,
        allow_with=True,
    )
    select_ast = _maybe_group_insert_select(
        select_ast,
        select_items=select_items,
        expr_types=expr_types,
        dialect=dialect,
    )
    if dialect_option(dialect, "supports_select_distinct", True) and random.random() < 0.25:
        select_ast.set("distinct", exp.Distinct())

    select_ast = _apply_insert_top_level_wrappers(
        select_ast,
        cols,
        dialect,
        schema=schema,
        target_table=table,
        table_ids=table_ids,
    )
    select_ast = _maybe_apply_insert_statement_with(select_ast, cols, dialect)
    if not upsert:
        select_ast = _limit_insert_target_rows(select_ast, dialect=dialect)

    by_name = bool(
        dialect_option(dialect, "supports_insert_by_name", False)
        and random.random() < INSERT_BY_NAME_PROB
    )
    insert_expression: exp.Expression = (
        _by_name_insert_expression(
            select_ast,
            cols,
            schema=schema,
            table=table,
            dialect=dialect,
        )
        if by_name
        else select_ast
    )
    stmt = _build_insert_stmt(
        schema,
        table,
        cols,
        insert_expression,
        dialect,
        ignore=ignore,
        replace=replace,
        upsert=upsert,
        by_name=by_name,
        returning=_maybe_insert_returning(schema, table, dialect),
    )
    _inject_random_having_on_selects(stmt, dialect, prob=INSERT_SELECT_HAVING_PROB)
    sql = _render_sql(stmt, dialect)
    with_style = dialect_option(dialect, "insert_select_with_style", "before_select")
    if with_style == "statement_prefix":
        return sql + ";"
    return sql + ";"
