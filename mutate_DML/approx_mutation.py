import random
import re
from dataclasses import dataclass
from typing import Iterable

try:
    import sqlglot
    from sqlglot import exp
except Exception as exc:  # pragma: no cover
    raise ImportError("sqlglot is required for approximate mutation") from exc

from init_env.core.dialect import dialect_option, get_sqlglot_dialect
from mutate_DML.sql_stream import split_sql_stream
from init_env.utils.sqlglot_compat import normalize_sqlglot_input


@dataclass
class MutationResult:
    original: str
    mutated: str
    mode: str  # over / under


class ApproximateMutator:
    _ANONYMOUS_AGGREGATES = {
        "STD",
        "STDDEV",
        "STDDEV_POP",
        "STDDEV_SAMP",
        "VARIANCE",
        "VAR_POP",
        "VAR_SAMP",
        "GROUP_CONCAT",
    }

    def __init__(
        self,
        dialect: str = "mysql",
        seed: int | None = None,
        *,
        preprocess_join_types: bool = False,
    ) -> None:
        self.dialect = dialect
        self.sqlglot_dialect = get_sqlglot_dialect(dialect)
        self.rand = random.Random(seed)
        self.preprocess_join_types = preprocess_join_types

    def mutate_statement(
        self,
        sql: str,
        *,
        preprocess_join_types: bool | None = None,
    ) -> list[MutationResult]:
        orig_sql = self.preprocess_statement(
            sql,
            preprocess_join_types=preprocess_join_types,
        ).strip().rstrip(";")
        return self.mutate_preprocessed_statement(orig_sql)

    def mutate_preprocessed_statement(self, sql: str) -> list[MutationResult]:
        orig_sql = (sql or "").strip().rstrip(";")
        if not orig_sql:
            return []
        has_var_samp = bool(re.search(r"\bVAR_SAMP\b", orig_sql, flags=re.IGNORECASE))
        has_variance = bool(re.search(r"\bVARIANCE\b", orig_sql, flags=re.IGNORECASE))
        has_div = bool(re.search(r"\bDIV\b", orig_sql, flags=re.IGNORECASE))
        parse_sql, kind, has_low_priority, has_quick, has_ignore, replace_has_into = (
            self._strip_dml_modifiers_with_meta(orig_sql)
        )
        try:
            stmt = self._parse_sql_best_effort(parse_sql)
        except Exception:
            return []

        if not isinstance(stmt, (exp.Update, exp.Delete, exp.Insert)):
            return []

        is_delete = isinstance(stmt, exp.Delete)
        self._strip_limit(stmt)
        if isinstance(stmt, exp.Insert) and self._insert_select_has_limit(stmt):
            return []

        results: list[MutationResult] = []
        if isinstance(stmt, exp.Insert):
            results.extend(
                self._mutate_insert_variants(
                    stmt,
                    orig_sql,
                    mode="over",
                    has_ignore=has_ignore,
                    has_low_priority=has_low_priority,
                    has_quick=has_quick,
                    is_replace=(kind == "replace"),
                    replace_has_into=replace_has_into,
                    has_var_samp=has_var_samp,
                    has_variance=has_variance,
                    has_div=has_div,
                )
            )
            results.extend(
                self._mutate_insert_variants(
                    stmt,
                    orig_sql,
                    mode="under",
                    has_ignore=has_ignore,
                    has_low_priority=has_low_priority,
                    has_quick=has_quick,
                    is_replace=(kind == "replace"),
                    replace_has_into=replace_has_into,
                    has_var_samp=has_var_samp,
                    has_variance=has_variance,
                    has_div=has_div,
                )
            )
            return results

        results.extend(
            self._mutate_variants(
                stmt,
                orig_sql,
                mode="over",
                has_ignore=has_ignore,
                is_delete=is_delete,
                has_low_priority=has_low_priority,
                has_quick=has_quick,
                has_var_samp=has_var_samp,
                has_variance=has_variance,
                has_div=has_div,
            )
        )
        results.extend(
            self._mutate_variants(
                stmt,
                orig_sql,
                mode="under",
                has_ignore=has_ignore,
                is_delete=is_delete,
                has_low_priority=has_low_priority,
                has_quick=has_quick,
                has_var_samp=has_var_samp,
                has_variance=has_variance,
                has_div=has_div,
            )
        )
        return results

    def _insert_select_has_limit(self, stmt: exp.Insert) -> bool:
        select = stmt.args.get("expression")
        return isinstance(select, exp.Select) and select.args.get("limit") is not None

    def _strip_limit(self, stmt: exp.Expression) -> None:
        if isinstance(stmt, (exp.Update, exp.Delete)):
            if stmt.args.get("limit") is not None:
                stmt.set("limit", None)

    def _strip_dml_modifiers_with_meta(
        self,
        sql: str,
    ) -> tuple[str, str | None, bool, bool, bool, bool]:
        if re.search(r"\bUPDATE\b", sql, flags=re.IGNORECASE):
            has_low = bool(re.search(r"\bUPDATE\s+LOW_PRIORITY\b", sql, flags=re.IGNORECASE))
            has_ignore = bool(re.search(r"\bUPDATE\s+(?:LOW_PRIORITY\s+)?IGNORE\b", sql, flags=re.IGNORECASE))
            stripped = re.sub(
                r"\bUPDATE\b\s+(?:LOW_PRIORITY\s+)?(?:IGNORE\s+)?",
                "UPDATE ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "update", has_low, False, has_ignore, False
        if re.search(r"\bDELETE\b", sql, flags=re.IGNORECASE):
            has_low = bool(re.search(r"\bDELETE\s+LOW_PRIORITY\b", sql, flags=re.IGNORECASE))
            has_quick = bool(re.search(r"\bDELETE\s+(?:LOW_PRIORITY\s+)?QUICK\b", sql, flags=re.IGNORECASE))
            has_ignore = bool(re.search(r"\bDELETE\s+(?:LOW_PRIORITY\s+)?(?:QUICK\s+)?IGNORE\b", sql, flags=re.IGNORECASE))
            stripped = re.sub(
                r"\bDELETE\b\s+(?:LOW_PRIORITY\s+)?(?:QUICK\s+)?(?:IGNORE\s+)?",
                "DELETE ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "delete", has_low, has_quick, has_ignore, False
        if re.search(r"\bINSERT\s+OR\s+REPLACE\b", sql, flags=re.IGNORECASE):
            has_into = bool(re.search(r"\bINSERT\s+OR\s+REPLACE\s+INTO\b", sql, flags=re.IGNORECASE))
            stripped = re.sub(
                r"\bINSERT\s+OR\s+REPLACE\b\s+(?:INTO\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "replace", False, False, False, has_into
        if re.search(r"\bINSERT\s+OR\s+IGNORE\b", sql, flags=re.IGNORECASE):
            stripped = re.sub(
                r"\bINSERT\s+OR\s+IGNORE\b\s+(?:INTO\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "insert", False, False, True, False
        if re.search(r"\bINSERT\b", sql, flags=re.IGNORECASE):
            has_ignore = bool(re.search(r"\bINSERT\s+IGNORE\b", sql, flags=re.IGNORECASE))
            stripped = re.sub(
                r"\bINSERT\b\s+(?:IGNORE\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "insert", False, False, has_ignore, False
        if re.search(r"\bREPLACE\b", sql, flags=re.IGNORECASE):
            has_low = bool(re.search(r"\bREPLACE\s+LOW_PRIORITY\b", sql, flags=re.IGNORECASE))
            has_delayed = bool(re.search(r"\bREPLACE\s+DELAYED\b", sql, flags=re.IGNORECASE))
            has_into = bool(re.search(r"\bREPLACE\b(?:\s+LOW_PRIORITY|\s+DELAYED)?\s+INTO\b", sql, flags=re.IGNORECASE))
            stripped = re.sub(
                r"\bREPLACE\b\s+(?:LOW_PRIORITY\s+)?(?:DELAYED\s+)?(?:INTO\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
            return stripped, "replace", has_low, has_delayed, False, has_into
        return sql, None, False, False, False, False

    def _strip_dml_modifiers(self, sql: str) -> str:
        if re.search(r"\bUPDATE\b", sql, flags=re.IGNORECASE):
            return re.sub(
                r"\bUPDATE\b\s+(?:LOW_PRIORITY\s+)?(?:IGNORE\s+)?",
                "UPDATE ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        if re.search(r"\bDELETE\b", sql, flags=re.IGNORECASE):
            return re.sub(
                r"\bDELETE\b\s+(?:LOW_PRIORITY\s+)?(?:QUICK\s+)?(?:IGNORE\s+)?",
                "DELETE ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        if re.search(r"\bINSERT\s+OR\s+REPLACE\b", sql, flags=re.IGNORECASE):
            return re.sub(
                r"\bINSERT\s+OR\s+REPLACE\b\s+(?:INTO\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        if re.search(r"\bINSERT\b", sql, flags=re.IGNORECASE):
            return re.sub(
                r"\bINSERT\b\s+(?:IGNORE\s+)?",
                "INSERT ",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        return sql

    def _normalize_dialect_render(self, sql: str) -> str:
        if self.dialect != "monetdb":
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

    def _split_top_level_csv(self, text: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        in_single = False
        in_double = False
        in_backtick = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_double and not in_backtick:
                if in_single and i + 1 < len(text) and text[i + 1] == "'":
                    buf.append(ch)
                    buf.append(text[i + 1])
                    i += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single and not in_backtick:
                in_double = not in_double
            elif ch == "`" and not in_single and not in_double:
                in_backtick = not in_backtick
            elif not in_single and not in_double and not in_backtick:
                if ch == "(":
                    depth += 1
                elif ch == ")" and depth > 0:
                    depth -= 1
                elif ch == "," and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    i += 1
                    continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    def _split_top_level_assignment(self, text: str) -> tuple[str, str] | None:
        depth = 0
        in_single = False
        in_double = False
        in_backtick = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_double and not in_backtick:
                if in_single and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single and not in_backtick:
                in_double = not in_double
            elif ch == "`" and not in_single and not in_double:
                in_backtick = not in_backtick
            elif not in_single and not in_double and not in_backtick:
                if ch == "(":
                    depth += 1
                elif ch == ")" and depth > 0:
                    depth -= 1
                elif ch == "=" and depth == 0:
                    left = text[:i].strip()
                    right = text[i + 1:].strip()
                    if left and right:
                        return left, right
                    return None
            i += 1
        return None

    def _rewrite_duckdb_replace_set(self, sql: str) -> str:
        match = re.match(
            r"(?is)^\s*(INSERT\s+OR\s+REPLACE\s+INTO\s+.+?)\s+SET\s+(.*)\s*$",
            sql,
        )
        if not match:
            return sql
        target_sql = match.group(1).strip()
        assign_sql = match.group(2).strip()
        assignments = self._split_top_level_csv(assign_sql)
        cols: list[str] = []
        vals: list[str] = []
        for assignment in assignments:
            pair = self._split_top_level_assignment(assignment)
            if pair is None:
                return sql
            col, val = pair
            cols.append(col)
            vals.append(val)
        return f"{target_sql} ({', '.join(cols)}) VALUES ({', '.join(vals)})"

    def _normalize_duckdb_source_sql(self, sql: str) -> str:
        if self.dialect != "duckdb":
            return sql
        sql = re.sub(r"\bX'([0-9A-Fa-f]+)'", r"FROM_HEX('\1')", sql)
        sql = re.sub(
            r"(?i)\bST_GeomFromText\s*\(\s*('(?:''|[^'])*')\s*\)",
            r"\1",
            sql,
        )
        sql = re.sub(
            r"(?i)\bDAYNAME\s*\(\s*('(?:''|[^'])*')\s*\)",
            r"DAYNAME(TRY_CAST(\1 AS DATE))",
            sql,
        )
        sql = re.sub(
            r"(?is)\bREPLACE\b\s+(?:LOW_PRIORITY\s+)?(?:DELAYED\s+)?(?:INTO\s+)?",
            "INSERT OR REPLACE INTO ",
            sql,
            count=1,
        )
        sql = self._rewrite_duckdb_replace_set(sql)
        return sql

    def _parse_sql_best_effort(self, sql: str) -> exp.Expression:
        source_sql = normalize_sqlglot_input(self._normalize_duckdb_source_sql(sql), self.dialect)
        if self.dialect == "duckdb":
            last_exc: Exception | None = None
            for read in ("mysql", "duckdb"):
                try:
                    return sqlglot.parse_one(source_sql, read=read)
                except Exception as exc:
                    last_exc = exc
            assert last_exc is not None
            raise last_exc
        return sqlglot.parse_one(source_sql, read=self.sqlglot_dialect)

    def _aggregate_argument(self, agg: exp.Expression) -> exp.Expression:
        arg = agg.args.get("this")
        if isinstance(arg, exp.Distinct):
            exprs = list(arg.expressions or [])
            if exprs:
                return exprs[0].copy()
            return exp.Literal.number(1)
        if isinstance(arg, exp.Star):
            return exp.Literal.number(1)
        if isinstance(arg, exp.Expression):
            return arg.copy()
        exprs = list(agg.args.get("expressions") or [])
        if exprs and isinstance(exprs[0], exp.Expression):
            return exprs[0].copy()
        return exp.Literal.number(1)

    def _is_aggregate_like(self, node: exp.Expression | None) -> bool:
        if node is None:
            return False
        if isinstance(node, exp.AggFunc):
            return True
        if isinstance(node, exp.Anonymous):
            return node.name.upper() in self._ANONYMOUS_AGGREGATES
        return False

    def _nearest_select(self, node: exp.Expression) -> exp.Select | None:
        current: exp.Expression | None = node
        while current is not None:
            if isinstance(current, exp.Select):
                return current
            current = current.parent
        return None

    def _is_select_in_scalar_context(self, select: exp.Select) -> bool:
        current: exp.Expression | None = select
        while current is not None:
            parent = current.parent
            if isinstance(parent, exp.Subquery) and self._is_scalar_subquery(parent):
                return True
            current = parent
        return False

    def _is_select_under_set_operation(self, select: exp.Select) -> bool:
        current: exp.Expression | None = select
        while current is not None:
            parent = current.parent
            if isinstance(parent, exp.SetOperation):
                return True
            if isinstance(parent, exp.Subquery) and self._is_scalar_subquery(parent):
                return False
            current = parent
        return False

    def _ensure_scalar_query_determinism(self, select: exp.Select) -> None:
        if select.args.get("order") is None:
            first_expr: exp.Expression | None = None
            expressions = list(select.expressions or [])
            if expressions:
                candidate = expressions[0]
                if isinstance(candidate, exp.Alias):
                    first_expr = candidate.this.copy() if isinstance(candidate.this, exp.Expression) else None
                elif isinstance(candidate, exp.Expression):
                    first_expr = candidate.copy()
            if isinstance(first_expr, exp.Literal):
                for col in select.find_all(exp.Column):
                    first_expr = col.copy()
                    break
            if first_expr is None:
                first_expr = exp.Literal.number(1)
            select.set(
                "order",
                exp.Order(expressions=[exp.Ordered(this=first_expr, desc=False)]),
            )
        order = select.args.get("order")
        if isinstance(order, exp.Order):
            for ordered in order.expressions or []:
                if not isinstance(ordered, exp.Ordered):
                    continue
                if ordered.args.get("nulls_first") is not None:
                    continue
                desc = bool(ordered.args.get("desc"))
                ordered.set("nulls_first", not desc)
        if select.args.get("limit") is None:
            select.limit(1, copy=False)

    def _preprocess_statement_ast(self, stmt: exp.Expression) -> exp.Expression:
        scalar_select_ids: set[int] = set()

        for node in stmt.walk():
            if not isinstance(node, exp.Expression):
                continue
            owner: exp.Select | None = None
            if isinstance(node, exp.Window) and self._is_aggregate_like(node.this):
                owner = self._nearest_select(node)
            elif self._is_aggregate_like(node):
                owner = self._nearest_select(node)
            if (
                owner is not None
                and self._is_select_in_scalar_context(owner)
                and not self._is_select_under_set_operation(owner)
            ):
                scalar_select_ids.add(id(owner))

        def transform(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Window) and self._is_aggregate_like(node.this):
                return self._aggregate_argument(node.this)
            if self._is_aggregate_like(node):
                return self._aggregate_argument(node)
            return node

        stmt = stmt.transform(transform, copy=False)
        for select in stmt.find_all(exp.Select):
            if select.args.get("group") is not None:
                select.set("group", None)
            having = select.args.get("having")
            if select.args.get("group") is None and isinstance(having, exp.Having) and isinstance(having.this, exp.Expression):
                where = select.args.get("where")
                if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression):
                    select.set("where", exp.Where(this=exp.And(this=where.this.copy(), expression=having.this.copy())))
                else:
                    select.set("where", exp.Where(this=having.this.copy()))
                select.set("having", None)
            if id(select) in scalar_select_ids:
                self._ensure_scalar_query_determinism(select)
        return stmt

    def _normalize_all_joins_to_inner(self, stmt: exp.Expression) -> None:
        for join in stmt.find_all(exp.Join):
            join.set("side", None)
            join.set("kind", "INNER")

    def preprocess_statement(
        self,
        sql: str,
        *,
        preprocess_join_types: bool | None = None,
    ) -> str:
        sql = (sql or "").strip().rstrip(";")
        if not sql:
            return sql
        sql = self._normalize_duckdb_source_sql(sql)
        normalize_join_types = (
            self.preprocess_join_types
            if preprocess_join_types is None
            else preprocess_join_types
        )

        has_var_samp = bool(re.search(r"\bVAR_SAMP\b", sql, flags=re.IGNORECASE))
        has_variance = bool(re.search(r"\bVARIANCE\b", sql, flags=re.IGNORECASE))
        has_div = bool(re.search(r"\bDIV\b", sql, flags=re.IGNORECASE))
        parse_sql, kind, has_low, has_quick, has_ignore, replace_has_into = self._strip_dml_modifiers_with_meta(sql)
        try:
            stmt = self._parse_sql_best_effort(parse_sql)
        except Exception:
            return sql

        self._strip_limit(stmt)
        stmt = self._preprocess_statement_ast(stmt)
        if normalize_join_types:
            self._normalize_all_joins_to_inner(stmt)
        stmt = self._rewrite_duckdb_compat(stmt)
        self._restore_mysql_ifnull(stmt)
        rendered = self._normalize_dialect_render(stmt.sql(dialect=self.sqlglot_dialect).strip())
        if self.dialect in {"mysql", "mariadb", "percona", "oceanbase", "tidb", "polardb"}:
            rendered = self._restore_mysql_regexp_like(rendered)
        rendered = self._restore_dml_modifiers(
            rendered,
            is_delete=(kind == "delete"),
            has_low_priority=has_low,
            has_quick=has_quick,
            has_ignore=has_ignore,
            is_insert=(kind == "insert"),
            is_replace=(kind == "replace"),
            replace_has_into=replace_has_into,
        )
        if has_var_samp and not has_variance:
            rendered = re.sub(r"\bVARIANCE\b", "VAR_SAMP", rendered, flags=re.IGNORECASE)
        if has_div:
            rendered = self._restore_mysql_div(rendered)
        return rendered

    def _rewrite_duckdb_update_join(self, stmt: exp.Update) -> exp.Expression:
        joins = list(stmt.args.get("joins") or [])
        if not joins and isinstance(stmt.this, exp.Table):
            joins = list(stmt.this.args.get("joins") or [])
        if not joins:
            return stmt

        target = stmt.this.copy()
        source_from = stmt.this.copy()
        if isinstance(target, exp.Table):
            target.set("joins", None)
        for join in source_from.find_all(exp.Join):
            kind = str(join.args.get("kind") or "").upper()
            if kind == "STRAIGHT_JOIN":
                join.set("kind", None)
        target_name = target.alias_or_name
        assignments = list(stmt.args.get("expressions") or [])
        if not target_name or not assignments:
            return stmt

        rewritten_assignments: list[tuple[str, str]] = []
        for idx, assign in enumerate(assignments, 1):
            if not isinstance(assign, exp.EQ) or not isinstance(assign.this, exp.Column):
                return stmt
            lhs = assign.this
            lhs_table = lhs.table
            if lhs_table and lhs_table != target_name:
                return stmt
            target_col = lhs.name
            if not target_col or not isinstance(assign.expression, exp.Expression):
                return stmt
            rewritten_assignments.append(
                (target_col, assign.expression.sql(dialect=self.sqlglot_dialect), idx)
            )

        select_items = [f"{target_name}.c1 AS __duckdb_pk"]
        select_items.extend(
            f"{rhs_sql} AS __duckdb_set_{idx}" for _, rhs_sql, idx in rewritten_assignments
        )
        target_sql = target.sql(dialect=self.sqlglot_dialect)
        source_sql = f"SELECT {', '.join(select_items)} FROM {source_from.sql(dialect=self.sqlglot_dialect)}"
        where = stmt.args.get("where")
        if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression):
            source_sql += f" WHERE {where.this.sql(dialect=self.sqlglot_dialect)}"

        set_sql = ", ".join(
            f"{col} = __duck_src.__duckdb_set_{idx}" for col, _, idx in rewritten_assignments
        )
        rewritten_sql = (
            f"UPDATE {target_sql} SET {set_sql} "
            f"FROM __duck_src "
            f"WHERE {target_name}.c1 = __duck_src.__duckdb_pk"
        )
        try:
            source_query = sqlglot.parse_one(source_sql, read=self.sqlglot_dialect)
            source_query = self._hoist_duckdb_non_inner_join_subqueries(source_query)
            source_query = self._rewrite_duckdb_non_inner_join_on_subqueries(source_query)
            source_extra_ctes: list[exp.CTE] = []
            source_with = source_query.args.get("with_")
            if isinstance(source_with, exp.With):
                source_extra_ctes = [cte.copy() for cte in (source_with.expressions or [])]
                source_query.set("with_", None)

            rewritten = sqlglot.parse_one(rewritten_sql, read=self.sqlglot_dialect)
            source_cte = exp.CTE(
                this=source_query,
                alias=exp.TableAlias(this=exp.to_identifier("__duck_src")),
            )
            with_clause = stmt.args.get("with_")
            if isinstance(with_clause, exp.With):
                rewritten.set(
                    "with_",
                    exp.With(
                        expressions=[expr.copy() for expr in (with_clause.expressions or [])]
                        + source_extra_ctes
                        + [source_cte],
                        recursive=with_clause.args.get("recursive"),
                    ),
                )
            else:
                rewritten.set("with_", exp.With(expressions=source_extra_ctes + [source_cte]))
            return self._normalize_duckdb_cte_order(rewritten)
        except Exception:
            return stmt

    def _rewrite_duckdb_delete_join(self, stmt: exp.Delete) -> exp.Expression:
        target = stmt.this.copy()
        if isinstance(target, exp.Table):
            target.set("joins", None)
        target_name = target.alias_or_name
        if not target_name:
            return stmt

        source_sql: str | None = None
        if stmt.args.get("tables"):
            source_from = stmt.this.copy()
            for join in source_from.find_all(exp.Join):
                kind = str(join.args.get("kind") or "").upper()
                if kind == "STRAIGHT_JOIN":
                    join.set("kind", None)
            source_sql = f"SELECT {target_name}.c1 AS __duckdb_pk FROM {source_from.sql(dialect=self.sqlglot_dialect)}"
        else:
            using_items = list(stmt.args.get("using") or [])
            if using_items:
                using_sql = ", ".join(item.copy().sql(dialect=self.sqlglot_dialect) for item in using_items)
                target_from_sql = target.sql(dialect=self.sqlglot_dialect)
                source_sql = f"SELECT {target_name}.c1 AS __duckdb_pk FROM {target_from_sql}, {using_sql}"
        if source_sql is None:
            return stmt

        where = stmt.args.get("where")
        if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression):
            source_sql += f" WHERE {where.this.sql(dialect=self.sqlglot_dialect)}"

        target_sql = target.sql(dialect=self.sqlglot_dialect)
        rewritten_sql = (
            f"DELETE FROM {target_sql} "
            f"WHERE {target_name}.c1 IN (SELECT __duck_del.__duckdb_pk FROM __duck_del)"
        )
        try:
            source_query = sqlglot.parse_one(source_sql, read=self.sqlglot_dialect)
            source_query = self._hoist_duckdb_non_inner_join_subqueries(source_query)
            source_query = self._rewrite_duckdb_non_inner_join_on_subqueries(source_query)
            source_extra_ctes: list[exp.CTE] = []
            source_with = source_query.args.get("with_")
            if isinstance(source_with, exp.With):
                source_extra_ctes = [cte.copy() for cte in (source_with.expressions or [])]
                source_query.set("with_", None)

            rewritten = sqlglot.parse_one(rewritten_sql, read=self.sqlglot_dialect)
            source_cte = exp.CTE(
                this=source_query,
                alias=exp.TableAlias(this=exp.to_identifier("__duck_del")),
            )
            with_clause = stmt.args.get("with_")
            if isinstance(with_clause, exp.With):
                rewritten.set(
                    "with_",
                    exp.With(
                        expressions=[expr.copy() for expr in (with_clause.expressions or [])]
                        + source_extra_ctes
                        + [source_cte],
                        recursive=with_clause.args.get("recursive"),
                    ),
                )
            else:
                rewritten.set("with_", exp.With(expressions=source_extra_ctes + [source_cte]))
            return self._normalize_duckdb_cte_order(rewritten)
        except Exception:
            return stmt

    def _rewrite_duckdb_nodes(self, expr: exp.Expression) -> exp.Expression:
        def _text_cast(value: exp.Expression) -> exp.Expression:
            return exp.Cast(this=value.copy(), to=exp.DataType.build("TEXT"))

        def transform(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Time) and isinstance(node.this, exp.Expression):
                # sqlglot currently renders TIME(expr) to malformed DuckDB SQL
                # like `CAST(CAST(expr AS TIMESTAMPTZ) AT TIME ZONE  AS TIME)`.
                # Normalize it to a plain cast before serialization.
                return exp.Cast(this=node.this.copy(), to=exp.DataType.build("TIME"))
            if isinstance(node, exp.Anonymous):
                name = node.name.upper()
                args = list(node.expressions or [])
                if name == "DATE_FORMAT" and len(args) == 2:
                    return exp.TimeToStr(this=args[0].copy(), format=args[1].copy())
                if name == "STR_TO_DATE" and len(args) == 2:
                    return exp.StrToTime(this=args[0].copy(), format=args[1].copy())
                if name == "TIMEDIFF" and len(args) == 2:
                    return exp.Cast(
                        this=exp.Sub(
                            this=exp.Cast(this=args[0].copy(), to=exp.DataType.build("TIMESTAMP")),
                            expression=exp.Cast(this=args[1].copy(), to=exp.DataType.build("TIMESTAMP")),
                        ),
                        to=exp.DataType.build("TEXT"),
                    )
                if name == "STD" and len(args) == 1:
                    return exp.Anonymous(this="STDDEV_SAMP", expressions=[args[0].copy()])
                if name == "REGEXP_MATCHES" and len(args) >= 2:
                    rewritten_args = [_text_cast(args[0]), _text_cast(args[1])]
                    rewritten_args.extend(arg.copy() for arg in args[2:])
                    return exp.Anonymous(this="REGEXP_MATCHES", expressions=rewritten_args)
                if name == "DAYNAME" and len(args) == 1:
                    return exp.Anonymous(
                        this="DAYNAME",
                        expressions=[
                            exp.TryCast(this=args[0].copy(), to=exp.DataType.build("DATE"))
                        ],
                    )
                if name == "DATE_DIFF" and len(args) == 3:
                    unit = args[0].copy()
                    if isinstance(unit, exp.Cast) and isinstance(unit.this, exp.Literal) and unit.this.is_string:
                        unit = exp.Literal.string(unit.this.this)
                    elif isinstance(unit, exp.Literal) and not unit.is_string:
                        unit = exp.Literal.string(unit.this)
                    elif isinstance(unit, exp.Var):
                        unit = exp.Literal.string(unit.this)
                    return exp.DateDiff(this=args[1].copy(), expression=args[2].copy(), unit=unit)
                if name == "DATEDIFF" and len(args) == 3:
                    unit = args[2].copy()
                    if isinstance(unit, exp.Cast) and isinstance(unit.this, exp.Literal) and unit.this.is_string:
                        unit = exp.Literal.string(unit.this.this)
                    elif isinstance(unit, exp.Literal) and not unit.is_string:
                        unit = exp.Literal.string(unit.this)
                    elif isinstance(unit, exp.Var):
                        unit = exp.Literal.string(unit.this)
                    return exp.DateDiff(this=args[0].copy(), expression=args[1].copy(), unit=unit)
            if isinstance(node, exp.RegexpLike):
                left = node.this.copy() if isinstance(node.this, exp.Expression) else exp.Literal.string("")
                right = node.expression.copy() if isinstance(node.expression, exp.Expression) else exp.Literal.string("")
                return exp.RegexpLike(this=_text_cast(left), expression=_text_cast(right))
            if isinstance(node, exp.SubstringIndex):
                count_expr = node.args.get("count")
                if count_expr is None:
                    count_expr = exp.Literal.number(1)
                return exp.Anonymous(
                    this="SPLIT_PART",
                    expressions=[
                        node.this.copy(),
                        node.args["delimiter"].copy(),
                        count_expr.copy(),
                    ],
                )
            if isinstance(node, exp.DateDiff):
                date_this = node.args.get("this")
                date_expr = node.args.get("expression")
                unit = node.args.get("unit")
                if (
                    isinstance(date_this, (exp.Literal, exp.Var))
                    and (
                        isinstance(date_this, exp.Var)
                        or (isinstance(date_this, exp.Literal) and date_this.is_string)
                    )
                    and isinstance(unit, exp.Expression)
                    and not isinstance(unit, (exp.Literal, exp.Var))
                    and isinstance(date_expr, exp.Expression)
                ):
                    node.set("this", date_expr.copy())
                    node.set("expression", unit.copy())
                    unit = date_this.copy()
                normalized_unit: exp.Expression
                if unit is None:
                    normalized_unit = exp.Literal.string("DAY")
                elif isinstance(unit, exp.Cast) and isinstance(unit.this, exp.Literal) and unit.this.is_string:
                    normalized_unit = exp.Literal.string(unit.this.this)
                elif isinstance(unit, exp.Literal):
                    normalized_unit = exp.Literal.string(unit.this)
                elif isinstance(unit, exp.Var):
                    normalized_unit = exp.Literal.string(unit.this)
                else:
                    normalized_unit = unit.copy()
                node.set("unit", normalized_unit)
                return node
            if isinstance(node, (exp.TimeToStr, exp.StrToTime)):
                fmt = node.args.get("format")
                if isinstance(fmt, exp.Literal) and fmt.is_string:
                    normalized = (
                        fmt.this.replace("%i", "%M").replace("%s", "%S")
                    )
                    node.set("format", exp.Literal.string(normalized))
            if isinstance(node, (exp.Left, exp.Right)) and isinstance(node.this, exp.Expression):
                node.set("this", exp.Cast(this=node.this.copy(), to=exp.DataType.build("TEXT")))
                if isinstance(node.expression, exp.Expression):
                    node.set("expression", exp.Cast(this=node.expression.copy(), to=exp.DataType.build("BIGINT")))
            if isinstance(node, (exp.Hour, exp.Minute, exp.Second)) and isinstance(node.this, exp.Literal) and node.this.is_string:
                node.set("this", exp.Cast(this=node.this.copy(), to=exp.DataType.build("TIMESTAMP")))
            return node

        return expr.transform(transform, copy=False)

    def _attach_duckdb_ctes(self, expr: exp.Expression, ctes: list[exp.CTE]) -> exp.Expression:
        if not ctes:
            return expr
        with_clause = expr.args.get("with_")
        if isinstance(with_clause, exp.With):
            with_clause.set("expressions", list(with_clause.expressions or []) + ctes)
            return self._normalize_duckdb_cte_order(expr)
        expr.set("with_", exp.With(expressions=ctes))
        return self._normalize_duckdb_cte_order(expr)

    def _duckdb_cte_name(self, cte: exp.CTE) -> str:
        alias = cte.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.this is not None:
            return alias.this.name
        return ""

    def _duckdb_cte_priority(self, cte: exp.CTE) -> int:
        name = self._duckdb_cte_name(cte)
        if not name.startswith("__duck_"):
            return 0
        if name.startswith("__duck_join_"):
            return 1
        return 2

    def _normalize_duckdb_cte_order(self, expr: exp.Expression) -> exp.Expression:
        if self.dialect != "duckdb":
            return expr
        for node in expr.walk():
            if not isinstance(node, exp.With):
                continue
            expressions = list(node.expressions or [])
            if len(expressions) <= 1:
                continue
            node.set(
                "expressions",
                sorted(
                    expressions,
                    key=lambda item: self._duckdb_cte_priority(item)
                    if isinstance(item, exp.CTE)
                    else 99,
                ),
            )
        return expr

    def _duckdb_true_expr(self) -> exp.Expression:
        return exp.Boolean(this=True)

    def _join_condition_has_subquery(self, cond: exp.Expression | None) -> bool:
        if not isinstance(cond, exp.Expression):
            return False
        return any(
            isinstance(node, (exp.Subquery, exp.Select, exp.SetOperation))
            for node in cond.walk()
            if node is not cond
        )

    def _duckdb_source_alias(self, source: exp.Expression | None) -> str | None:
        if isinstance(source, exp.Table):
            return source.alias_or_name or source.name or None
        if isinstance(source, exp.Subquery):
            alias = source.args.get("alias")
            if isinstance(alias, exp.TableAlias):
                return alias.this.name
        if isinstance(source, exp.Lateral):
            alias = source.args.get("alias")
            if isinstance(alias, exp.TableAlias):
                return alias.this.name
        return getattr(source, "alias_or_name", None) or None

    def _duckdb_lateralize_join_source(
        self,
        source: exp.Expression,
        cond: exp.Expression,
        outer_alias: str,
    ) -> exp.Lateral | None:
        try:
            source_sql = source.sql(dialect=self.sqlglot_dialect)
            cond_sql = cond.sql(dialect=self.sqlglot_dialect)
            query = sqlglot.parse_one(
                f"SELECT * FROM {source_sql} WHERE {cond_sql}",
                read=self.sqlglot_dialect,
            )
            return exp.Lateral(
                this=exp.Subquery(this=query),
                alias=exp.TableAlias(this=exp.to_identifier(outer_alias)),
            )
        except Exception:
            return None

    def _rewrite_duckdb_non_inner_join_on_subqueries(
        self,
        expr: exp.Expression,
    ) -> exp.Expression:
        for select in expr.find_all(exp.Select):
            joins = list(select.args.get("joins") or [])
            if len(joins) != 1:
                continue

            join = joins[0]
            side = str(join.args.get("side") or "").upper()
            if side not in {"LEFT", "RIGHT"}:
                continue

            cond = join.args.get("on")
            if not self._join_condition_has_subquery(cond):
                continue

            from_clause = select.args.get("from_")
            if not isinstance(from_clause, exp.From) or not isinstance(cond, exp.Expression):
                continue

            if side == "LEFT":
                outer_alias = self._duckdb_source_alias(join.this)
                if not outer_alias:
                    continue
                lateral = self._duckdb_lateralize_join_source(join.this.copy(), cond.copy(), outer_alias)
                if lateral is None:
                    continue
                join.set("this", lateral)
                join.set("on", self._duckdb_true_expr())
                continue

            left_source = from_clause.this
            if not isinstance(left_source, exp.Expression):
                continue
            left_alias = self._duckdb_source_alias(left_source)
            if not left_alias:
                continue
            lateral = self._duckdb_lateralize_join_source(left_source.copy(), cond.copy(), left_alias)
            if lateral is None:
                continue
            from_clause.set("this", join.this.copy())
            join.set("side", "LEFT")
            join.set("this", lateral)
            join.set("on", self._duckdb_true_expr())

        return expr

    def _hoist_duckdb_non_inner_join_subqueries(self, expr: exp.Expression) -> exp.Expression:
        ctes: list[exp.CTE] = []
        counter = 0

        def transform(node: exp.Expression) -> exp.Expression:
            nonlocal counter
            if not isinstance(node, exp.Join):
                return node
            side = str(node.args.get("side") or "").upper()
            if side not in {"LEFT", "RIGHT", "FULL"}:
                return node
            join_target = node.this
            if not isinstance(join_target, exp.Subquery) or not isinstance(join_target.this, exp.Query):
                return node

            counter += 1
            cte_name = f"__duck_join_{counter}"
            ctes.append(
                exp.CTE(
                    this=join_target.this.copy(),
                    alias=exp.TableAlias(this=exp.to_identifier(cte_name)),
                )
            )

            replacement = exp.to_table(cte_name)
            alias = join_target.args.get("alias")
            if isinstance(alias, exp.TableAlias):
                replacement.set("alias", alias.copy())
            node.set("this", replacement)
            return node

        expr = expr.transform(transform, copy=False)
        return self._attach_duckdb_ctes(expr, ctes)

    def _rewrite_duckdb_compat(self, expr: exp.Expression) -> exp.Expression:
        if self.dialect != "duckdb":
            return expr

        expr = self._rewrite_duckdb_nodes(expr)
        expr = self._hoist_duckdb_non_inner_join_subqueries(expr)
        expr = self._rewrite_duckdb_non_inner_join_on_subqueries(expr)

        if isinstance(expr, exp.Update) and (
            expr.args.get("joins") or (isinstance(expr.this, exp.Table) and expr.this.args.get("joins"))
        ):
            expr = self._rewrite_duckdb_update_join(expr)
        elif isinstance(expr, exp.Delete) and (expr.args.get("tables") or expr.args.get("using")):
            expr = self._rewrite_duckdb_delete_join(expr)

        expr = self._hoist_duckdb_non_inner_join_subqueries(expr)
        expr = self._normalize_duckdb_cte_order(expr)

        for join in expr.find_all(exp.Join):
            kind = str(join.args.get("kind") or "").upper()
            if kind == "STRAIGHT_JOIN":
                join.set("kind", None)

        return expr

    def _restore_dml_modifiers(
        self,
        sql: str,
        *,
        is_delete: bool,
        has_low_priority: bool,
        has_quick: bool,
        has_ignore: bool,
        is_insert: bool = False,
        is_replace: bool = False,
        replace_has_into: bool = False,
    ) -> str:
        if is_replace:
            if self.dialect == "duckdb":
                return re.sub(
                    r"\bINSERT\b(?:\s+INTO\b)?",
                    "INSERT OR REPLACE INTO",
                    sql,
                    flags=re.IGNORECASE,
                    count=1,
                )
            opts = []
            if has_low_priority and dialect_option(self.dialect, "supports_replace_low_priority", True):
                opts.append("LOW_PRIORITY")
            if has_quick and dialect_option(self.dialect, "supports_replace_delayed", True):
                opts.append("DELAYED")
            prefix = "REPLACE"
            if opts:
                prefix += " " + " ".join(opts)
            if replace_has_into:
                prefix += " INTO"
            return re.sub(
                r"\bINSERT\b(?:\s+INTO\b)?",
                prefix,
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        if is_insert:
            if has_ignore and dialect_option(self.dialect, "supports_insert_ignore", True):
                replacement = "INSERT OR IGNORE" if self.dialect == "duckdb" else "INSERT IGNORE"
                return re.sub(
                    r"\bINSERT\b",
                    replacement,
                    sql,
                    flags=re.IGNORECASE,
                    count=1,
                )
            return sql
        if is_delete:
            opts = []
            if has_low_priority and dialect_option(self.dialect, "supports_delete_low_priority", True):
                opts.append("LOW_PRIORITY")
            if has_quick and dialect_option(self.dialect, "supports_delete_quick", True):
                opts.append("QUICK")
            if has_ignore and dialect_option(self.dialect, "supports_delete_ignore", True):
                opts.append("IGNORE")
            if opts:
                return re.sub(
                    r"\bDELETE\b",
                    "DELETE " + " ".join(opts),
                    sql,
                    flags=re.IGNORECASE,
                    count=1,
                )
            return sql
        opts = []
        if has_low_priority and dialect_option(self.dialect, "supports_update_low_priority", True):
            opts.append("LOW_PRIORITY")
        if has_ignore and dialect_option(self.dialect, "supports_update_ignore", True):
            opts.append("IGNORE")
        if opts:
            return re.sub(
                r"\bUPDATE\b",
                "UPDATE " + " ".join(opts),
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        return sql

    def _restore_mysql_div(self, sql: str) -> str:
        pattern = r"\bCAST\(\s*([A-Za-z0-9_\.]+|\d+(?:\.\d+)?)\s*/\s*([A-Za-z0-9_\.]+|\d+(?:\.\d+)?)\s+AS\s+(SIGNED|UNSIGNED|INT|INTEGER)\s*\)"
        return re.sub(pattern, r"\1 DIV \2", sql, flags=re.IGNORECASE)

    def _mutate_insert_variants(
        self,
        stmt: exp.Insert,
        raw: str,
        mode: str,
        has_ignore: bool,
        has_low_priority: bool,
        has_quick: bool,
        is_replace: bool,
        replace_has_into: bool,
        has_var_samp: bool,
        has_variance: bool,
        has_div: bool,
    ) -> list[MutationResult]:
        select = stmt.args.get("expression")
        if not isinstance(select, exp.Select):
            return []

        seen: set[str] = set()
        results: list[MutationResult] = []

        def emit(mutated: exp.Expression, result_mode: str | None = None) -> None:
            sql = self._to_sql(mutated)
            sql = self._restore_dml_modifiers(
                sql,
                is_delete=False,
                has_low_priority=has_low_priority,
                has_quick=has_quick,
                has_ignore=has_ignore,
                is_insert=not is_replace,
                is_replace=is_replace,
                replace_has_into=replace_has_into,
            )
            if has_var_samp and not has_variance:
                sql = re.sub(r"\bVARIANCE\b", "VAR_SAMP", sql, flags=re.IGNORECASE)
            if has_div:
                sql = self._restore_mysql_div(sql)
            if sql in seen:
                return
            seen.add(sql)
            results.append(MutationResult(raw, sql, result_mode or mode))

        where = select.args.get("where")
        if where is None:
            if mode == "under":
                mutated = stmt.copy()
                mselect = mutated.args.get("expression")
                if isinstance(mselect, exp.Select):
                    mselect.set("where", exp.Where(this=self._false_expr()))
                    emit(mutated)
        else:
            cond = where.this
            for new_cond in self._where_mutations(cond, mode):
                mutated = stmt.copy()
                mselect = mutated.args.get("expression")
                if not isinstance(mselect, exp.Select):
                    continue
                mselect.set("where", exp.Where(this=new_cond))
                emit(mutated)

        # Keep INSERT mutations on the top-level source only. Do not recurse into
        # target projection expressions, otherwise scalar subqueries/COALESCE can
        # change inserted values and create false positives.
        joins = list(select.args.get("joins") or [])
        if joins:
            for idx, join in enumerate(joins):
                if self._is_node_in_case_when_condition(join):
                    continue
                on = join.args.get("on")
                if on is not None:
                    result_mode = self._join_on_result_mode(join, select, mode)
                    if result_mode is None:
                        continue
                    for new_on in self._join_on_mutations(on, mode):
                        mutated = stmt.copy()
                        mselect = mutated.args.get("expression")
                        if not isinstance(mselect, exp.Select):
                            continue
                        mjoins = list(mselect.args.get("joins") or [])
                        if idx >= len(mjoins):
                            continue
                        mjoins[idx].set("on", new_on)
                        emit(mutated, result_mode)

                for new_type in self._join_type_mutations(join, mode):
                    mutated = stmt.copy()
                    mselect = mutated.args.get("expression")
                    if not isinstance(mselect, exp.Select):
                        continue
                    mjoins = list(mselect.args.get("joins") or [])
                    if idx >= len(mjoins):
                        continue
                    self._set_join_type(mjoins[idx], new_type)
                    emit(mutated)

        source_query_roots = [
            (path, query_expr)
            for path, query_expr in self._source_subquery_query_roots(select)
            if query_expr is not select
            and not self._is_node_in_scalar_subquery(query_expr)
            and not self._is_node_in_having_scalar_subquery(query_expr)
            and not self._is_node_in_case_when_condition(query_expr)
        ]
        for source_path, source_query in source_query_roots:
            effective_mode = self._effective_mode(mode, self._polarity_for_node(source_query))
            for mutated_query in self._query_structure_mutations(
                source_query,
                effective_mode,
                allow_filter_mutation=True,
            ):
                mutated = stmt.copy()
                mselect = mutated.args.get("expression")
                if not isinstance(mselect, exp.Select):
                    continue
                target_query = self._node_at_path(mselect, source_path)
                if not isinstance(target_query, exp.Expression):
                    continue
                target_query.replace(mutated_query)
                emit(mutated)

        source_set_ops = [
            (self._expression_path(set_op, select), set_op)
            for set_op in select.find_all(exp.SetOperation)
            if not self._is_node_in_select_projection(set_op, select)
            and not self._is_node_in_scalar_subquery(set_op)
            and not self._is_node_in_having_scalar_subquery(set_op)
            and not self._is_node_in_case_when_condition(set_op)
        ]
        if source_set_ops:
            for source_path, set_op in source_set_ops:
                for new_setop in self._setop_mutations(set_op, mode):
                    mutated = stmt.copy()
                    mselect = mutated.args.get("expression")
                    if not isinstance(mselect, exp.Select):
                        continue
                    target_setop = self._node_at_path(mselect, source_path)
                    if not isinstance(target_setop, exp.SetOperation):
                        continue
                    target_setop.replace(new_setop)
                    emit(mutated)

        for new_select in self._select_distinct_mutations(select, mode):
            mutated = stmt.copy()
            mselect = mutated.args.get("expression")
            if not isinstance(mselect, exp.Select):
                continue
            mselect.set("distinct", new_select.args.get("distinct"))
            emit(mutated)

        for new_select in self._select_having_mutations(select, mode):
            mutated = stmt.copy()
            mselect = mutated.args.get("expression")
            if not isinstance(mselect, exp.Select):
                continue
            mselect.set("having", new_select.args.get("having"))
            emit(mutated)

        return results

    def _mutate_variants(
        self,
        stmt: exp.Expression,
        raw: str,
        mode: str,
        has_ignore: bool,
        is_delete: bool,
        has_low_priority: bool,
        has_quick: bool,
        has_var_samp: bool,
        has_variance: bool,
        has_div: bool,
    ) -> list[MutationResult]:
        where = stmt.args.get("where")
        seen: set[str] = set()
        results: list[MutationResult] = []

        def emit(mutated: exp.Expression, result_mode: str | None = None) -> None:
            sql = self._to_sql(mutated)
            sql = self._restore_dml_modifiers(
                sql,
                is_delete=is_delete,
                has_low_priority=has_low_priority,
                has_quick=has_quick,
                has_ignore=has_ignore,
            )
            if has_var_samp and not has_variance:
                sql = re.sub(r"\bVARIANCE\b", "VAR_SAMP", sql, flags=re.IGNORECASE)
            if has_div:
                sql = self._restore_mysql_div(sql)
            if sql in seen:
                return
            seen.add(sql)
            results.append(MutationResult(raw, sql, result_mode or mode))

        if where is None:
            if mode == "under":
                mutated = stmt.copy()
                mutated.set("where", exp.Where(this=self._false_expr()))
                emit(mutated)
        else:
            cond = where.this
            for new_cond in self._where_mutations(cond, mode):
                mutated = stmt.copy()
                mutated.set("where", exp.Where(this=new_cond))
                emit(mutated)

        joins = list(stmt.find_all(exp.Join))
        if joins:
            for idx, join in enumerate(joins):
                if self._is_node_in_case_when_condition(join):
                    continue
                on = join.args.get("on")
                if on is not None:
                    result_mode = self._join_on_result_mode(join, stmt, mode)
                    if result_mode is None:
                        continue
                    for new_on in self._join_on_mutations(on, mode):
                        mutated = stmt.copy()
                        mjoins = list(mutated.find_all(exp.Join))
                        if idx >= len(mjoins):
                            continue
                        mjoins[idx].set("on", new_on)
                        emit(mutated, result_mode)

                for new_type in self._join_type_mutations(join, mode):
                    mutated = stmt.copy()
                    mjoins = list(mutated.find_all(exp.Join))
                    if idx >= len(mjoins):
                        continue
                    self._set_join_type(mjoins[idx], new_type)
                    emit(mutated)

        set_ops = list(stmt.find_all(exp.SetOperation))
        if set_ops:
            for idx, set_op in enumerate(set_ops):
                if (
                    self._is_node_in_scalar_subquery(set_op)
                    or self._is_node_in_having_scalar_subquery(set_op)
                    or self._is_node_in_case_when_condition(set_op)
                ):
                    continue
                for new_setop in self._setop_mutations(set_op, mode):
                    mutated = stmt.copy()
                    mset_ops = list(mutated.find_all(exp.SetOperation))
                    if idx >= len(mset_ops):
                        continue
                    mset_ops[idx].replace(new_setop)
                    emit(mutated)

        source_selects = [
            (idx, select)
            for idx, select in enumerate(stmt.find_all(exp.Select))
            if not self._is_select_in_update_assignment(select, stmt)
            and not self._is_node_in_scalar_subquery(select)
            and not self._is_node_in_having_scalar_subquery(select)
            and not self._is_node_in_case_when_condition(select)
        ]
        if source_selects:
            for source_idx, source_select in source_selects:
                effective_mode = self._effective_mode(mode, self._polarity_for_node(source_select))

                for new_select in self._select_distinct_mutations(source_select, effective_mode):
                    mutated = stmt.copy()
                    mselects = list(mutated.find_all(exp.Select))
                    if source_idx >= len(mselects):
                        continue
                    mselects[source_idx].set("distinct", new_select.args.get("distinct"))
                    emit(mutated)

                for new_select in self._select_having_mutations(source_select, effective_mode):
                    mutated = stmt.copy()
                    mselects = list(mutated.find_all(exp.Select))
                    if source_idx >= len(mselects):
                        continue
                    mselects[source_idx].set("having", new_select.args.get("having"))
                    emit(mutated)

        return results

    def _is_select_in_update_assignment(self, select: exp.Select, stmt: exp.Expression) -> bool:
        if not isinstance(stmt, exp.Update):
            return False

        current = select
        while current is not None and current is not stmt:
            if current.parent is stmt and current.arg_key == "expressions":
                return True
            current = current.parent
        return False

    def _is_node_in_select_projection(self, node: exp.Expression, select: exp.Select) -> bool:
        current: exp.Expression | None = node
        while current is not None and current is not select:
            if current.parent is select and current.arg_key == "expressions":
                return True
            current = current.parent
        return False

    def _has_non_scalar_subquery_ancestor(
        self,
        node: exp.Expression,
        ancestor: exp.Expression,
    ) -> bool:
        current: exp.Expression | None = node.parent
        while current is not None and current is not ancestor:
            if isinstance(current, exp.Subquery) and not self._is_scalar_subquery(current):
                return True
            current = current.parent
        return False

    def _source_subquery_query_roots(
        self,
        ancestor: exp.Expression,
    ) -> list[tuple[list[tuple[str, int | None]], exp.Expression]]:
        roots: list[tuple[list[tuple[str, int | None]], exp.Expression]] = []

        for query in ancestor.find_all(exp.Select, exp.SetOperation):
            if query is ancestor:
                continue
            if not self._has_non_scalar_subquery_ancestor(query, ancestor):
                continue
            if isinstance(ancestor, exp.Select) and self._is_node_in_select_projection(query, ancestor):
                continue
            if self._is_node_in_case_when_condition(query):
                continue
            try:
                roots.append((self._expression_path(query, ancestor), query))
            except ValueError:
                continue

        return roots

    def _is_scalar_subquery(self, subquery: exp.Subquery) -> bool:
        parent = subquery.parent
        if isinstance(parent, exp.In) and subquery.arg_key == "query":
            return False
        if isinstance(parent, exp.Exists) and subquery.arg_key == "this":
            return False
        if isinstance(parent, (exp.From, exp.Join, exp.CTE, exp.Lateral)):
            return False
        return True

    def _is_node_in_having_scalar_subquery(self, node: exp.Expression) -> bool:
        current: exp.Expression | None = node
        seen_scalar_subquery = False

        while current is not None:
            if isinstance(current, exp.Subquery) and self._is_scalar_subquery(current):
                seen_scalar_subquery = True
            elif seen_scalar_subquery and isinstance(current, exp.Having):
                return True
            current = current.parent

        return False

    def _is_node_in_scalar_subquery(self, node: exp.Expression) -> bool:
        current: exp.Expression | None = node

        while current is not None:
            if isinstance(current, exp.Subquery) and self._is_scalar_subquery(current):
                return True
            current = current.parent

        return False

    def _is_node_in_case_when_condition(self, node: exp.Expression) -> bool:
        current: exp.Expression | None = node

        while current is not None and current.parent is not None:
            parent = current.parent
            if isinstance(parent, exp.If) and current.arg_key == "this":
                return True
            current = parent

        return False

    def _expression_path(
        self,
        node: exp.Expression,
        ancestor: exp.Expression,
    ) -> list[tuple[str, int | None]]:
        path: list[tuple[str, int | None]] = []
        current: exp.Expression | None = node

        while current is not None and current is not ancestor:
            parent = current.parent
            if parent is None or current.arg_key is None:
                raise ValueError("Node is not a descendant of the requested ancestor.")
            parent_arg = parent.args.get(current.arg_key)
            if isinstance(parent_arg, list):
                index = next((i for i, item in enumerate(parent_arg) if item is current), None)
                if index is None:
                    raise ValueError("Could not locate node inside parent list argument.")
                path.append((current.arg_key, index))
            else:
                path.append((current.arg_key, None))
            current = parent

        if current is not ancestor:
            raise ValueError("Node is not a descendant of the requested ancestor.")

        path.reverse()
        return path

    def _node_at_path(
        self,
        ancestor: exp.Expression,
        path: list[tuple[str, int | None]],
    ) -> exp.Expression | None:
        current: exp.Expression | None = ancestor
        for arg_key, index in path:
            if current is None:
                return None
            child = current.args.get(arg_key)
            if index is None:
                if not isinstance(child, exp.Expression):
                    return None
                current = child
            else:
                if not isinstance(child, list) or index >= len(child):
                    return None
                item = child[index]
                if not isinstance(item, exp.Expression):
                    return None
                current = item
        return current

    def _to_sql(self, expr: exp.Expression) -> str:
        expr = self._rewrite_duckdb_compat(expr)
        self._restore_mysql_ifnull(expr)
        sql = self._normalize_dialect_render(expr.sql(dialect=self.sqlglot_dialect).strip())
        if self.dialect in {"mysql", "mariadb", "percona", "oceanbase", "tidb", "polardb"}:
            sql = self._restore_mysql_regexp_like(sql)
        return sql + ";"

    def _restore_mysql_regexp_like(self, sql: str) -> str:
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

    def _restore_mysql_ifnull(self, expr: exp.Expression) -> None:
        if self.dialect not in {"mysql", "mariadb", "percona", "oceanbase", "tidb", "polardb"}:
            return
        for node in list(expr.walk()):
            if isinstance(node, exp.Coalesce):
                args = list(node.expressions)
                if len(args) == 2:
                    node.replace(exp.Anonymous(this="IFNULL", expressions=[a.copy() for a in args]))

    def _literal_string_value(self, node: exp.Expression | None) -> str | None:
        if isinstance(node, exp.Literal) and node.is_string:
            return node.this
        return None

    def _like_over_patterns(self, pattern: str) -> list[str]:
        base = pattern.replace("%", "").replace("_", "")
        if not base:
            return []
        starts = pattern.startswith("%")
        ends = pattern.endswith("%")

        variants: list[str] = []
        if not starts and not ends:
            variants.extend([f"{base}%", f"%{base}", f"%{base}%"])
        elif starts and not ends:
            variants.append(f"%{base}%")
        elif not starts and ends:
            variants.append(f"%{base}%")
        return [v for i, v in enumerate(variants) if v != pattern and v not in variants[:i]]

    def _like_under_patterns(self, pattern: str) -> list[str]:
        base = pattern.replace("%", "").replace("_", "")
        if not base:
            return []
        variants = []
        if pattern.startswith("%") and pattern.endswith("%"):
            variants.extend([f"{base}%", f"%{base}", base])
        elif pattern.startswith("%"):
            variants.append(base)
        elif "%" in pattern or "_" in pattern:
            variants.append(base)
        return [v for i, v in enumerate(variants) if v != pattern and v not in variants[:i]]

    def _regexp_over_patterns(self, pattern: str) -> list[str]:
        variants = [f".*{pattern}.*", f"{pattern}.*"]
        return [v for i, v in enumerate(variants) if v != pattern and v not in variants[:i]]

    def _regexp_under_patterns(self, pattern: str) -> list[str]:
        variants: list[str] = []
        stripped = pattern.removeprefix(".*").removesuffix(".*")
        if stripped and stripped != pattern:
            variants.append(stripped)
        literal = re.sub(r"[\.\*\+\?\^\$\|\(\)\[\]\{\}\\]", "", pattern)
        if literal and literal != pattern:
            variants.append(literal)
        return [v for i, v in enumerate(variants) if v not in variants[:i]]

    def _true_expr(self) -> exp.Expression:
        return exp.EQ(
            this=exp.Literal.number(1),
            expression=exp.Literal.number(1),
        )

    def _false_expr(self) -> exp.Expression:
        return exp.EQ(
            this=exp.Literal.number(1),
            expression=exp.Literal.number(0),
        )

    def _unwrap_parens(self, node: exp.Expression | None) -> exp.Expression | None:
        current = node
        while isinstance(current, exp.Paren) and current.this is not None:
            current = current.this
        return current

    def _column_matches(
        self,
        node: exp.Expression | None,
        *,
        table: str | None,
        name: str,
    ) -> bool:
        col = self._unwrap_parens(node)
        if not isinstance(col, exp.Column):
            return False
        if (col.name or "").lower() != name.lower():
            return False
        if table is None:
            return True
        return (col.table or "").lower() == table.lower()

    def _relation_aliases(self, node: exp.Expression | None) -> set[str]:
        expr = self._unwrap_parens(node)
        if expr is None:
            return set()

        aliases: set[str] = set()
        alias = getattr(expr, "alias_or_name", None)
        if alias:
            aliases.add(str(alias))

        if isinstance(expr, exp.Table):
            if expr.name:
                aliases.add(expr.alias_or_name or expr.name)
            return {a for a in aliases if a}

        if isinstance(expr, exp.Subquery):
            return {a for a in aliases if a}

        for table in expr.find_all(exp.Table):
            name = table.alias_or_name or table.name
            if name:
                aliases.add(name)

        return {a for a in aliases if a}

    def _left_input_aliases_for_join(self, join: exp.Join) -> set[str]:
        parent = join.parent
        if not isinstance(parent, exp.Expression) or join.arg_key != "joins":
            return set()

        aliases: set[str] = set()
        joins = list(parent.args.get("joins") or [])

        if isinstance(parent, exp.Select):
            from_clause = parent.args.get("from_")
            if isinstance(from_clause, exp.From) and isinstance(from_clause.this, exp.Expression):
                aliases |= self._relation_aliases(from_clause.this)
        elif isinstance(parent, exp.From):
            if isinstance(parent.this, exp.Expression):
                aliases |= self._relation_aliases(parent.this)
        else:
            aliases |= self._relation_aliases(parent)

        for sibling in joins:
            if sibling is join:
                break
            aliases |= self._relation_aliases(sibling.this)

        return aliases

    def _null_extended_aliases_for_join(self, join: exp.Join) -> set[str]:
        join_type = self._join_type(join)
        if join_type == "LEFT":
            return self._relation_aliases(join.this)
        if join_type == "RIGHT":
            return self._left_input_aliases_for_join(join)
        return set()

    def _root_filter_for_join(self, root: exp.Expression | None) -> exp.Expression | None:
        if not isinstance(root, (exp.Select, exp.Update, exp.Delete)):
            return None
        where = root.args.get("where")
        return where.this if isinstance(where, exp.Where) and isinstance(where.this, exp.Expression) else None

    def _expr_references_aliases(self, node: exp.Expression | None, aliases: set[str]) -> bool:
        if not isinstance(node, exp.Expression) or not aliases:
            return False
        alias_lc = {a.lower() for a in aliases}
        return any((col.table or "").lower() in alias_lc for col in node.find_all(exp.Column))

    def _join_on_result_mode(
        self,
        join: exp.Join,
        root: exp.Expression | None,
        mode: str,
    ) -> str | None:
        null_side_aliases = self._null_extended_aliases_for_join(join)
        if not null_side_aliases:
            return mode

        where_cond = self._root_filter_for_join(root)
        if not isinstance(where_cond, exp.Expression):
            return mode

        saw_is_null = False
        saw_is_not_null = False

        for node in where_cond.walk():
            if self._is_null_predicate(node):
                target = node.this if isinstance(node, exp.Is) else None
                if self._expr_references_aliases(target, null_side_aliases):
                    saw_is_null = True
            elif isinstance(node, exp.Expression) and self._is_is_not_null_predicate(node):
                inner = node.this.this if isinstance(node.this, exp.Is) else None
                if self._expr_references_aliases(inner, null_side_aliases):
                    saw_is_not_null = True

        if saw_is_null and saw_is_not_null:
            return None
        if saw_is_null:
            return self._flip_mode(mode)
        return mode

    def _is_duckdb_update_glue_predicate(self, cond: exp.Expression | None) -> bool:
        node = self._unwrap_parens(cond)
        if not isinstance(node, exp.EQ):
            return False
        return (
            self._column_matches(node.this, table="__duck_src", name="__duckdb_pk")
            and self._column_matches(node.expression, table=None, name="c1")
        ) or (
            self._column_matches(node.this, table=None, name="c1")
            and self._column_matches(node.expression, table="__duck_src", name="__duckdb_pk")
        )

    def _is_duckdb_delete_glue_predicate(self, cond: exp.Expression | None) -> bool:
        node = self._unwrap_parens(cond)
        if not isinstance(node, exp.In) or node.args.get("query") is None:
            return False
        if not self._column_matches(node.this, table=None, name="c1"):
            return False

        query = node.args.get("query")
        if not isinstance(query, exp.Subquery) or not isinstance(query.this, exp.Select):
            return False

        select = query.this
        projections = list(select.expressions or [])
        if len(projections) != 1:
            return False
        if not self._column_matches(projections[0], table="__duck_del", name="__duckdb_pk"):
            return False

        from_clause = select.args.get("from_")
        if not isinstance(from_clause, exp.From):
            return False
        source = self._unwrap_parens(from_clause.this)
        return isinstance(source, exp.Table) and (source.name or "").lower() == "__duck_del"

    def _is_duckdb_rewrite_glue_predicate(self, cond: exp.Expression | None) -> bool:
        if self.dialect != "duckdb":
            return False
        return self._is_duckdb_update_glue_predicate(cond) or self._is_duckdb_delete_glue_predicate(cond)

    def _contains_duckdb_rewrite_glue_predicate(self, cond: exp.Expression | None) -> bool:
        if not isinstance(cond, exp.Expression) or self.dialect != "duckdb":
            return False
        return any(
            self._is_duckdb_rewrite_glue_predicate(node)
            for node in cond.walk()
            if isinstance(node, exp.Expression)
        )

    def _join_type(self, join: exp.Join) -> str:
        side = join.args.get("side")
        kind = join.args.get("kind")
        if side:
            return str(side).upper()
        if kind:
            return str(kind).upper()
        return "INNER"

    def _set_join_type(self, join: exp.Join, join_type: str) -> None:
        jt = join_type.upper()
        if jt in {"LEFT", "RIGHT"}:
            join.set("side", jt)
            join.set("kind", None)
        elif jt in {"INNER", "STRAIGHT_JOIN", "CROSS"}:
            join.set("side", None)
            join.set("kind", jt)
        else:
            join.set("side", None)
            join.set("kind", None)

    def _join_type_mutations(self, join: exp.Join, mode: str) -> list[str]:
        jt = self._join_type(join)
        if mode == "over":
            if jt in {"INNER", "STRAIGHT_JOIN"}:
                variants = ["LEFT"]
                if dialect_option(self.dialect, "supports_right_join", True):
                    variants.append("RIGHT")
                return variants
            return []
        if jt in {"LEFT", "RIGHT"}:
            return ["INNER"]
        return []

    def _flip_mode(self, mode: str) -> str:
        return "under" if mode == "over" else "over"

    def _effective_mode(self, mode: str, positive_polarity: bool) -> str:
        return mode if positive_polarity else self._flip_mode(mode)

    def _polarity_for_node(self, node: exp.Expression) -> bool:
        positive = True
        current = node

        while current.parent is not None:
            parent = current.parent
            if isinstance(parent, exp.Not) and current.arg_key == "this":
                positive = not positive
            elif isinstance(parent, exp.Except) and current.arg_key == "expression":
                positive = not positive
            current = parent

        return positive

    def _setop_mutations(
        self,
        set_op: exp.SetOperation,
        mode: str,
        *,
        apply_node_polarity: bool = True,
    ) -> list[exp.Expression]:
        muts: list[exp.Expression] = []
        effective_mode = (
            self._effective_mode(mode, self._polarity_for_node(set_op))
            if apply_node_polarity
            else mode
        )

        if effective_mode == "over":
            if isinstance(set_op, exp.Union) and bool(set_op.args.get("distinct", True)):
                u = set_op.copy()
                u.set("distinct", False)
                muts.append(u)
            if isinstance(set_op, exp.Except):
                # Replace the operator instead of dropping the right branch so
                # the result column type derivation still sees both inputs.
                muts.append(
                    exp.Union(
                        this=set_op.this.copy(),
                        expression=set_op.expression.copy(),
                        distinct=True,
                    )
                )
            if isinstance(set_op, exp.Intersect):
                muts.append(
                    exp.Union(
                        this=set_op.this.copy(),
                        expression=set_op.expression.copy(),
                        distinct=True,
                    )
                )
            return muts

        if isinstance(set_op, exp.Union):
            if not bool(set_op.args.get("distinct", True)):
                u = set_op.copy()
                u.set("distinct", True)
                muts.append(u)
            if dialect_option(self.dialect, "supports_intersect", True):
                muts.append(exp.Intersect(this=set_op.this.copy(), expression=set_op.expression.copy()))

        return muts

    def _select_distinct_mutations(self, select: exp.Select, mode: str) -> list[exp.Select]:
        muts: list[exp.Select] = []
        if not self._is_distinct_approx_safe(select):
            return muts
        has_distinct = select.args.get("distinct") is not None
        if mode == "over":
            if has_distinct:
                s = select.copy()
                s.set("distinct", None)
                muts.append(s)
        else:
            if not has_distinct and dialect_option(self.dialect, "supports_select_distinct", True):
                s = select.copy()
                s.set("distinct", exp.Distinct())
                muts.append(s)
        return muts

    def _is_distinct_approx_safe(self, select: exp.Select) -> bool:
        # DISTINCT is approximation-safe only when duplicate multiplicity is not
        # observed inside the query and the query result is consumed in a
        # duplicate-insensitive way such as IN / EXISTS.
        if select.args.get("group") is not None or select.args.get("having") is not None:
            return False
        if select.args.get("limit") is not None:
            return False
        if any(True for _ in select.find_all(exp.AggFunc)):
            return False
        if any(True for _ in select.find_all(exp.Window)):
            return False

        current: exp.Expression | None = select
        while current is not None:
            parent = current.parent
            if isinstance(parent, exp.Subquery):
                grand = parent.parent
                if isinstance(grand, (exp.In, exp.Exists)):
                    return True
                if isinstance(grand, (exp.From, exp.Join, exp.CTE)):
                    return False
            if isinstance(parent, exp.Insert):
                return False
            current = parent

        return False

    def _select_having_mutations(self, select: exp.Select, mode: str) -> list[exp.Select]:
        muts: list[exp.Select] = []
        if not dialect_option(self.dialect, "supports_select_having", True):
            return muts

        group = select.args.get("group")
        having = select.args.get("having")

        if isinstance(having, exp.Having) and having.this is not None:
            for replacement in self._whole_predicate_mutations(mode):
                s = select.copy()
                s.set("having", exp.Having(this=replacement))
                muts.append(s)
            # HAVING often contains scalar subqueries folded into comparisons or
            # aggregates. Mutating set-ops inside those scalar subqueries is not
            # approximation-safe once LIMIT / scalarization is involved.
            for mutated_cond in self._collect_mutations(
                having.this,
                mode,
                allow_scalar_setop_mutation=False,
                allow_scalar_subquery_mutation=False,
            ):
                s = select.copy()
                s.set("having", exp.Having(this=mutated_cond))
                muts.append(s)
            for mutated_cond in self._collect_having_expr_mutations(having.this, mode):
                s = select.copy()
                s.set("having", exp.Having(this=mutated_cond))
                muts.append(s)
            if mode == "over":
                s = select.copy()
                s.set("having", None)
                muts.append(s)
        elif mode == "under" and group is not None:
            s = select.copy()
            s.set("having", exp.Having(this=self._false_expr()))
            muts.append(s)

        return muts
    def _comparison_child_effect(
        self,
        parent: exp.Expression,
        arg_key: str,
        mode: str,
    ) -> str | None:
        if isinstance(parent, (exp.GT, exp.GTE)):
            if arg_key == "this":
                return "increase" if mode == "over" else "decrease"
            if arg_key == "expression":
                return "decrease" if mode == "over" else "increase"
        if isinstance(parent, (exp.LT, exp.LTE)):
            if arg_key == "this":
                return "decrease" if mode == "over" else "increase"
            if arg_key == "expression":
                return "increase" if mode == "over" else "decrease"
        if isinstance(parent, exp.Between):
            if arg_key == "low":
                return "decrease" if mode == "over" else "increase"
            if arg_key == "high":
                return "increase" if mode == "over" else "decrease"
        return None

    def _collect_having_expr_mutations(
        self,
        expr: exp.Expression,
        mode: str,
        desired_effect: str | None = None,
        positive_polarity: bool = True,
    ) -> list[exp.Expression]:
        if self._is_node_in_case_when_condition(expr):
            return []

        variants: list[exp.Expression] = []
        effective_mode = self._effective_mode(mode, positive_polarity)
        variants.extend(self._having_expr_mutations_at_node(expr, desired_effect))

        for arg_key, child in expr.args.items():
            child_effect = self._comparison_child_effect(expr, arg_key, effective_mode)
            if child_effect is None:
                child_effect = desired_effect
            child_positive_polarity = positive_polarity
            if isinstance(expr, exp.Not) and arg_key == "this":
                child_positive_polarity = not child_positive_polarity
            elif isinstance(expr, exp.Except) and arg_key == "expression":
                child_positive_polarity = not child_positive_polarity
            if isinstance(child, exp.Expression):
                if isinstance(child, exp.Subquery) and self._is_scalar_subquery(child):
                    continue
                for mutated_child in self._collect_having_expr_mutations(
                    child,
                    mode,
                    child_effect,
                    child_positive_polarity,
                ):
                    mutated_expr = expr.copy()
                    mutated_expr.set(arg_key, mutated_child)
                    variants.append(mutated_expr)
            elif isinstance(child, list):
                for idx, item in enumerate(child):
                    if not isinstance(item, exp.Expression):
                        continue
                    if isinstance(item, exp.Subquery) and self._is_scalar_subquery(item):
                        continue
                    for mutated_child in self._collect_having_expr_mutations(
                        item,
                        mode,
                        child_effect,
                        child_positive_polarity,
                    ):
                        mutated_expr = expr.copy()
                        new_items = list(mutated_expr.args.get(arg_key) or [])
                        if idx >= len(new_items):
                            continue
                        new_items[idx] = mutated_child
                        mutated_expr.set(arg_key, new_items)
                        variants.append(mutated_expr)

        return variants

    def _having_expr_mutations_at_node(
        self,
        expr: exp.Expression,
        desired_effect: str | None,
    ) -> list[exp.Expression]:
        muts: list[exp.Expression] = []

        def offset_numeric(base: exp.Expression, delta: int) -> exp.Expression:
            literal = exp.Literal.number(abs(delta))
            if delta >= 0:
                return exp.Add(this=base.copy(), expression=literal)
            return exp.Sub(this=base.copy(), expression=literal)

        if desired_effect == "increase":
            if isinstance(expr, (exp.Count, exp.Sum, exp.Avg)):
                muts.append(offset_numeric(expr, 1))
            if isinstance(expr, exp.Sum) and expr.this is not None:
                muts.append(
                    exp.Sum(
                        this=exp.Mul(
                            this=expr.this.copy(),
                            expression=exp.Literal.number(2),
                        )
                    )
                )
            if isinstance(expr, exp.Count) and isinstance(expr.this, exp.Distinct):
                distinct_exprs = list(expr.this.expressions or [])
                if len(distinct_exprs) == 1:
                    mutated = expr.copy()
                    mutated.set("this", distinct_exprs[0].copy())
                    muts.append(mutated)
            if isinstance(expr, exp.Count) and expr.this is not None and not isinstance(expr.this, (exp.Distinct, exp.Star)):
                mutated = expr.copy()
                mutated.set("this", exp.Star())
                muts.append(mutated)
        elif desired_effect == "decrease":
            if isinstance(expr, (exp.Count, exp.Sum, exp.Avg)):
                muts.append(offset_numeric(expr, -1))
            if isinstance(expr, exp.Count) and expr.this is not None and not isinstance(expr.this, exp.Distinct):
                if not isinstance(expr.this, exp.Star):
                    mutated = expr.copy()
                    mutated.set("this", exp.Distinct(expressions=[expr.this.copy()]))
                    muts.append(mutated)

        return muts

    def _query_structure_mutations(
        self,
        query_expr: exp.Expression,
        mode: str,
        *,
        allow_setop: bool = True,
        allow_having: bool = True,
        allow_filter_mutation: bool = False,
    ) -> list[exp.Expression]:
        muts: list[exp.Expression] = []
        if isinstance(query_expr, exp.Select):
            muts.extend(self._select_distinct_mutations(query_expr, mode))
            if allow_having:
                muts.extend(self._select_having_mutations(query_expr, mode))
            if allow_filter_mutation:
                where = query_expr.args.get("where")
                if isinstance(where, exp.Where) and where.this is not None:
                    for mutated_cond in self._where_mutations(where.this, mode):
                        s = query_expr.copy()
                        s.set("where", exp.Where(this=mutated_cond))
                        muts.append(s)
                    if mode == "over" and self.dialect != "duckdb":
                        s = query_expr.copy()
                        s.set("where", None)
                        muts.append(s)
                elif mode == "under":
                    s = query_expr.copy()
                    s.set("where", exp.Where(this=self._false_expr()))
                    muts.append(s)

                joins = list(query_expr.find_all(exp.Join))
                for idx, join in enumerate(joins):
                    on = join.args.get("on")
                    if on is not None:
                        for new_on in self._join_on_mutations(on, mode):
                            s = query_expr.copy()
                            mjoins = list(s.find_all(exp.Join))
                            if idx >= len(mjoins):
                                continue
                            mjoins[idx].set("on", new_on)
                            muts.append(s)
                    for new_type in self._join_type_mutations(join, mode):
                        s = query_expr.copy()
                        mjoins = list(s.find_all(exp.Join))
                        if idx >= len(mjoins):
                            continue
                        self._set_join_type(mjoins[idx], new_type)
                        muts.append(s)
        if allow_setop and isinstance(query_expr, exp.SetOperation):
            # The caller has already propagated predicate polarity into `mode`
            # for membership/existence subqueries. Re-applying ancestor polarity
            # here would flip twice under constructs like NOT IN (... INTERSECT ...).
            muts.extend(
                self._setop_mutations(
                    query_expr,
                    mode,
                    apply_node_polarity=False,
                )
            )
        return muts

    def _is_null_predicate(self, cond: exp.Expression | None) -> bool:
        return isinstance(cond, exp.Is) and isinstance(cond.expression, exp.Null)

    def _is_is_not_null_predicate(self, cond: exp.Expression) -> bool:
        return isinstance(cond, exp.Not) and self._is_null_predicate(cond.this)

    def _scoped_not(self, cond: exp.Expression) -> exp.Not:
        wrapped = exp.Paren(this=cond) if isinstance(cond, (exp.And, exp.Or)) else cond
        return exp.Not(this=wrapped)

    def _scoped_or(self, left: exp.Expression, right: exp.Expression) -> exp.Paren:
        return exp.Paren(this=exp.Or(this=left, expression=right))

    def _whole_predicate_mutations(self, mode: str) -> list[exp.Expression]:
        if mode == "over":
            return [self._true_expr()]
        return [self._false_expr()]

    def _flatten_and_terms(self, cond: exp.Expression | None) -> list[exp.Expression]:
        if cond is None:
            return []
        node = self._unwrap_parens(cond)
        if isinstance(node, exp.And):
            return self._flatten_and_terms(node.this) + self._flatten_and_terms(node.expression)
        return [cond.copy()]

    def _build_and_terms(self, terms: list[exp.Expression]) -> exp.Expression:
        if not terms:
            return self._true_expr()
        current = terms[0].copy()
        for term in terms[1:]:
            current = exp.And(this=current, expression=term.copy())
        return current

    def _is_immutable_join_anchor_term(self, cond: exp.Expression | None) -> bool:
        node = self._unwrap_parens(cond)
        if not isinstance(node, exp.EQ):
            return False
        left = self._unwrap_parens(node.this)
        right = self._unwrap_parens(node.expression)
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return False
        left_table = (left.table or "").strip()
        right_table = (right.table or "").strip()
        return bool(left_table and right_table and left_table.lower() != right_table.lower())

    def _join_on_mutations(self, on: exp.Expression, mode: str) -> list[exp.Expression]:
        if self._contains_duckdb_rewrite_glue_predicate(on):
            return []

        terms = self._flatten_and_terms(on)
        immutable_indexes = {
            idx for idx, term in enumerate(terms) if self._is_immutable_join_anchor_term(term)
        }
        if not immutable_indexes:
            return self._collect_mutations(
                on,
                mode,
                allow_scalar_setop_mutation=False,
                allow_scalar_subquery_mutation=False,
                allow_or_expansion=False,
            )

        variants: list[exp.Expression] = []
        seen: set[str] = set()
        for idx, term in enumerate(terms):
            if idx in immutable_indexes:
                continue
            for mutated_term in self._collect_mutations(
                term,
                mode,
                allow_scalar_setop_mutation=False,
                allow_scalar_subquery_mutation=False,
                allow_or_expansion=False,
            ):
                rebuilt_terms = [
                    mutated_term if term_idx == idx else original_term
                    for term_idx, original_term in enumerate(terms)
                ]
                candidate = self._build_and_terms(rebuilt_terms)
                key = candidate.sql(dialect=self.sqlglot_dialect)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(candidate)
        return variants

    def _where_mutations(self, cond: exp.Expression, mode: str) -> list[exp.Expression]:
        if self._contains_duckdb_rewrite_glue_predicate(cond):
            return []
        if self.dialect == "duckdb":
            return self._join_on_mutations(cond, mode)

        variants: list[exp.Expression] = []
        for replacement in self._whole_predicate_mutations(mode):
            variants.append(replacement)
        variants.extend(
            self._collect_mutations(
                cond,
                mode,
                allow_scalar_setop_mutation=False,
                allow_scalar_subquery_mutation=False,
            )
        )
        return variants

    def _collect_mutations(
        self,
        cond: exp.Expression,
        mode: str,
        positive_polarity: bool = True,
        allow_scalar_setop_mutation: bool = True,
        allow_scalar_subquery_mutation: bool = True,
        allow_or_expansion: bool = True,
    ) -> list[exp.Expression]:
        if self._is_node_in_case_when_condition(cond) or self._is_duckdb_rewrite_glue_predicate(cond):
            return []

        variants: list[exp.Expression] = []
        effective_mode = self._effective_mode(mode, positive_polarity)
        variants.extend(self._mutations_at_node(cond, effective_mode, allow_or_expansion=allow_or_expansion))

        if isinstance(cond, exp.Paren) and cond.this is not None:
            for mutated in self._collect_mutations(
                cond.this,
                mode,
                positive_polarity,
                allow_scalar_setop_mutation=allow_scalar_setop_mutation,
                allow_scalar_subquery_mutation=allow_scalar_subquery_mutation,
                allow_or_expansion=allow_or_expansion,
            ):
                variants.append(exp.Paren(this=mutated))

        if isinstance(cond, (exp.And, exp.Or)):
            if cond.this is not None:
                for mutated in self._collect_mutations(
                    cond.this,
                    mode,
                    positive_polarity,
                    allow_scalar_setop_mutation=allow_scalar_setop_mutation,
                    allow_scalar_subquery_mutation=allow_scalar_subquery_mutation,
                    allow_or_expansion=allow_or_expansion,
                ):
                    variants.append(cond.__class__(this=mutated, expression=cond.expression.copy()))
            if cond.expression is not None:
                for mutated in self._collect_mutations(
                    cond.expression,
                    mode,
                    positive_polarity,
                    allow_scalar_setop_mutation=allow_scalar_setop_mutation,
                    allow_scalar_subquery_mutation=allow_scalar_subquery_mutation,
                    allow_or_expansion=allow_or_expansion,
                ):
                    variants.append(cond.__class__(this=cond.this.copy(), expression=mutated))

        if isinstance(cond, exp.Not) and cond.this is not None and not self._is_is_not_null_predicate(cond):
            for mutated in self._collect_mutations(
                cond.this,
                mode,
                not positive_polarity,
                allow_scalar_setop_mutation=allow_scalar_setop_mutation,
                allow_scalar_subquery_mutation=allow_scalar_subquery_mutation,
                allow_or_expansion=allow_or_expansion,
            ):
                variants.append(self._scoped_not(mutated))

        if isinstance(cond, exp.In):
            query = cond.args.get("query")
            if isinstance(query, exp.Subquery) and query.this is not None:
                for mutated_query in self._query_structure_mutations(
                    query.this,
                    effective_mode,
                    allow_filter_mutation=True,
                ):
                    variants.append(exp.In(this=cond.this.copy(), query=exp.Subquery(this=mutated_query)))

        if isinstance(cond, exp.Exists) and cond.this is not None:
            for mutated_query in self._query_structure_mutations(
                cond.this,
                effective_mode,
                allow_filter_mutation=True,
            ):
                variants.append(exp.Exists(this=mutated_query))

        if isinstance(cond, exp.Subquery) and cond.this is not None:
            if self._is_scalar_subquery(cond) and not allow_scalar_subquery_mutation:
                return variants
            for mutated_query in self._query_structure_mutations(
                cond.this,
                effective_mode,
                allow_setop=allow_scalar_setop_mutation,
                allow_having=allow_scalar_subquery_mutation,
            ):
                variants.append(exp.Subquery(this=mutated_query))

        return variants

    def _mutations_at_node(
        self,
        cond: exp.Expression,
        mode: str,
        *,
        allow_or_expansion: bool = True,
    ) -> list[exp.Expression]:
        muts: list[exp.Expression] = []

        if mode == "over":
            if isinstance(cond, exp.GT):
                muts.append(exp.GTE(this=cond.this.copy(), expression=cond.expression.copy()))
            if isinstance(cond, exp.LT):
                muts.append(exp.LTE(this=cond.this.copy(), expression=cond.expression.copy()))
            if isinstance(cond, exp.EQ):
                muts.append(exp.GTE(this=cond.this.copy(), expression=cond.expression.copy()))
                muts.append(exp.LTE(this=cond.this.copy(), expression=cond.expression.copy()))
            if isinstance(cond, exp.Between):
                muts.append(exp.GTE(this=cond.this.copy(), expression=cond.args["low"].copy()))
                muts.append(exp.LTE(this=cond.this.copy(), expression=cond.args["high"].copy()))
            if isinstance(cond, exp.Like):
                pattern = self._literal_string_value(cond.expression)
                if pattern is not None:
                    for variant in self._like_over_patterns(pattern):
                        muts.append(exp.Like(this=cond.this.copy(), expression=exp.Literal.string(variant)))
            if isinstance(cond, exp.RegexpLike):
                pattern = self._literal_string_value(cond.expression)
                if pattern is not None:
                    for variant in self._regexp_over_patterns(pattern):
                        muts.append(exp.RegexpLike(this=cond.this.copy(), expression=exp.Literal.string(variant)))
            if allow_or_expansion and isinstance(cond, exp.And):
                muts.append(self._scoped_or(cond.this.copy(), cond.expression.copy()))
            if allow_or_expansion:
                muts.append(self._scoped_or(cond.copy(), self._true_expr()))
        else:
            if isinstance(cond, exp.GTE):
                muts.append(exp.GT(this=cond.this.copy(), expression=cond.expression.copy()))
            if isinstance(cond, exp.LTE):
                muts.append(exp.LT(this=cond.this.copy(), expression=cond.expression.copy()))
            if isinstance(cond, exp.Between):
                muts.append(
                    exp.And(
                        this=exp.GT(this=cond.this.copy(), expression=cond.args["low"].copy()),
                        expression=exp.LT(this=cond.this.copy(), expression=cond.args["high"].copy()),
                    )
                )
            if isinstance(cond, exp.In) and not cond.args.get("query"):
                exprs = list(cond.expressions or [])
                if len(exprs) > 1:
                    for idx in range(len(exprs)):
                        muts.append(
                            exp.In(
                                this=cond.this.copy(),
                                expressions=[expr.copy() for j, expr in enumerate(exprs) if j != idx],
                            )
                        )
            if isinstance(cond, exp.Like):
                pattern = self._literal_string_value(cond.expression)
                if pattern is not None:
                    for variant in self._like_under_patterns(pattern):
                        muts.append(exp.Like(this=cond.this.copy(), expression=exp.Literal.string(variant)))
            if isinstance(cond, exp.RegexpLike):
                pattern = self._literal_string_value(cond.expression)
                if pattern is not None:
                    for variant in self._regexp_under_patterns(pattern):
                        muts.append(exp.RegexpLike(this=cond.this.copy(), expression=exp.Literal.string(variant)))
            if isinstance(cond, exp.Or):
                muts.append(exp.And(this=cond.this.copy(), expression=cond.expression.copy()))
            muts.append(exp.And(this=cond.copy(), expression=self._false_expr()))

        return muts


def mutate_file(
    input_path: str,
    output_path: str,
    dialect: str = "mysql",
    seed: int | None = None,
    *,
    preprocess_join_types: bool = False,
) -> int:
    mutator = ApproximateMutator(
        dialect=dialect,
        seed=seed,
        preprocess_join_types=preprocess_join_types,
    )
    count = 0
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for stmt in split_sql_stream(src):
            results = mutator.mutate_statement(stmt)
            for item in results:
                dst.write(item.mutated + "\n")
                count += 1
    return count
