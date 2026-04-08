import random
import re
import shutil
import multiprocessing as mp
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

try:
    import sqlglot
    from sqlglot import exp
except Exception as exc:  # pragma: no cover
    raise ImportError("sqlglot is required for result comparison") from exc

from init_env.core.dialect import dialect_option, get_database_name, get_dialect_config, get_sqlglot_dialect, normalize_dialect
from init_env.core.schema import get_schema
from init_env.utils.db_connect import connect_db, connection_driver, resolve_duckdb_database_path, resolve_round_database
from init_env.utils.sqlglot_compat import normalize_sqlglot_input
from mutate_DML.sql_stream import split_sql_stream


@dataclass
class CompareResult:
    ok: bool
    detail: str = ""


@dataclass
class _ParsedDml:
    kind: str  # update/delete/insert
    table: str
    table_sql: str
    pk: str
    where_sql: str
    order_sql: str
    limit_sql: str
    set_sql: str | None
    raw_sql: str
    targets: list[str]
    top_level_update_join_sqls: list[str]
    match_sql: str | None


@dataclass
class _BugArtifacts:
    bundle_dir: str
    snapshot_paths: list[str]
    original_plan_path: str | None
    mutated_plan_path: str | None


@dataclass
class _TimedCompareResult:
    result: CompareResult
    crashed: bool = False
    timed_out: bool = False


from mutate_DML.approx_mutation import ApproximateMutator


_CRASH_PATTERNS = (
    "lost connection to mysql server during query",
    "server has gone away",
    "server closed connection",
    "already closed",
    "can't connect to mysql server",
    "connection refused",
    "connection reset",
    "broken pipe",
    "interfaceerror(0",
)

_COMPARE_HEARTBEAT_SECONDS = 30.0
_COMPARE_PAIR_TIMEOUT_SECONDS = 30.0


def _safe_close(obj) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _print_compare_heartbeat(
    *,
    round_label: str,
    total: int,
    failed: int,
    skipped: int,
    crashed: int,
    started_at: float,
    stmt_idx: int | None = None,
    mut_idx: int | None = None,
) -> None:
    elapsed = time.monotonic() - started_at
    parts = [
        "[compare]",
        f"round={round_label}",
        f"total={total}",
        f"failed={failed}",
        f"skipped={skipped}",
        f"crashed={crashed}",
        f"elapsed_s={elapsed:.1f}",
    ]
    if stmt_idx is not None:
        parts.append(f"stmt_idx={stmt_idx}")
    if mut_idx is not None:
        parts.append(f"mut_idx={mut_idx}")
    print(" ".join(parts), flush=True)


@dataclass
class _CompareHeartbeatState:
    total: int = 0
    failed: int = 0
    skipped: int = 0
    crashed: int = 0
    stmt_idx: int | None = None
    mut_idx: int | None = None


def _start_compare_heartbeat(
    *,
    round_label: str,
    started_at: float,
    state: _CompareHeartbeatState,
    stop_event: threading.Event,
    lock: threading.Lock,
) -> threading.Thread:
    def _runner() -> None:
        while not stop_event.wait(_COMPARE_HEARTBEAT_SECONDS):
            with lock:
                total = state.total
                failed = state.failed
                skipped = state.skipped
                crashed = state.crashed
                stmt_idx = state.stmt_idx
                mut_idx = state.mut_idx
            _print_compare_heartbeat(
                round_label=round_label,
                total=total,
                failed=failed,
                skipped=skipped,
                crashed=crashed,
                started_at=started_at,
                stmt_idx=stmt_idx,
                mut_idx=mut_idx,
            )

    thread = threading.Thread(
        target=_runner,
        name=f"compare-heartbeat-{round_label}",
        daemon=True,
    )
    thread.start()
    return thread


def _compare_pair_once(
    *,
    dialect: str,
    db_name: str,
    original_sql: str,
    mutated_sql: str,
    mode: str | None,
    compare_basis: str,
    conn_holder: dict[str, object] | None = None,
) -> _TimedCompareResult:
    conn = None
    cur = None
    try:
        conn = connect_db(dialect, database=db_name)
        if conn_holder is not None:
            conn_holder["conn"] = conn
        cur = conn.cursor()
        result = _compare_pair_with_cursor(
            conn,
            cur,
            original_sql,
            mutated_sql,
            mode,
            dialect,
            compare_basis=compare_basis,
        )
        crashed = _is_exec_error(result.detail) and (
            _is_crash_error(result.detail) or not _connection_alive(conn)
        )
        return _TimedCompareResult(result=result, crashed=crashed)
    except Exception as exc:
        crashed = bool(conn is not None and not _connection_alive(conn))
        return _TimedCompareResult(
            result=CompareResult(False, f"EXEC_ERROR: {exc}"),
            crashed=crashed,
        )
    finally:
        if conn_holder is not None:
            conn_holder.pop("conn", None)
        _safe_close(cur)
        _safe_close(conn)


def _compare_pair_worker(
    send_conn,
    *,
    dialect: str,
    db_name: str,
    original_sql: str,
    mutated_sql: str,
    mode: str | None,
    compare_basis: str,
) -> None:
    try:
        timed_result = _compare_pair_once(
            dialect=dialect,
            db_name=db_name,
            original_sql=original_sql,
            mutated_sql=mutated_sql,
            mode=mode,
            compare_basis=compare_basis,
        )
        send_conn.send(
            {
                "ok": timed_result.result.ok,
                "detail": timed_result.result.detail,
                "crashed": timed_result.crashed,
                "timed_out": timed_result.timed_out,
            }
        )
    except Exception as exc:
        send_conn.send(
            {
                "ok": False,
                "detail": f"EXEC_ERROR: compare worker exception: {exc}",
                "crashed": False,
                "timed_out": False,
            }
        )
    finally:
        send_conn.close()


def _compare_pair_with_timeout(
    *,
    dialect: str,
    db_name: str,
    original_sql: str,
    mutated_sql: str,
    mode: str | None,
    compare_basis: str,
    timeout_s: float = _COMPARE_PAIR_TIMEOUT_SECONDS,
) -> _TimedCompareResult:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_compare_pair_worker,
        kwargs={
            "send_conn": send_conn,
            "dialect": dialect,
            "db_name": db_name,
            "original_sql": original_sql,
            "mutated_sql": mutated_sql,
            "mode": mode,
            "compare_basis": compare_basis,
        },
        daemon=True,
    )
    process.start()
    send_conn.close()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        recv_conn.close()
        return _TimedCompareResult(
            result=CompareResult(False, f"EXEC_ERROR: compare timeout after {timeout_s:.1f}s"),
            timed_out=True,
        )

    try:
        if recv_conn.poll(1.0):
            payload = recv_conn.recv()
            return _TimedCompareResult(
                result=CompareResult(bool(payload.get("ok")), str(payload.get("detail", ""))),
                crashed=bool(payload.get("crashed", False)),
                timed_out=bool(payload.get("timed_out", False)),
            )
    except (EOFError, OSError):
        pass
    finally:
        recv_conn.close()

    if process.exitcode not in (0, None):
        return _TimedCompareResult(
            result=CompareResult(False, f"EXEC_ERROR: compare worker exited with code {process.exitcode}")
        )
    return _TimedCompareResult(
        result=CompareResult(False, "EXEC_ERROR: compare worker exited without a result")
    )

def _strip_dml_modifiers(sql: str) -> tuple[str, str | None, bool, bool, bool]:
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
        return stripped, "update", has_low, False, has_ignore
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
        return stripped, "delete", has_low, has_quick, has_ignore
    if re.search(r"\bINSERT\s+OR\s+REPLACE\b", sql, flags=re.IGNORECASE):
        stripped = re.sub(
            r"\bINSERT\s+OR\s+REPLACE\b\s+(?:INTO\s+)?",
            "INSERT ",
            sql,
            flags=re.IGNORECASE,
            count=1,
        )
        return stripped, "insert", False, False, False
    if re.search(r"\bINSERT\s+OR\s+IGNORE\b", sql, flags=re.IGNORECASE):
        stripped = re.sub(
            r"\bINSERT\s+OR\s+IGNORE\b\s+(?:INTO\s+)?",
            "INSERT ",
            sql,
            flags=re.IGNORECASE,
            count=1,
        )
        return stripped, "insert", False, False, True
    if re.search(r"\bINSERT\b", sql, flags=re.IGNORECASE):
        has_ignore = bool(re.search(r"\bINSERT\s+IGNORE\b", sql, flags=re.IGNORECASE))
        stripped = re.sub(
            r"\bINSERT\b\s+(?:IGNORE\s+)?",
            "INSERT ",
            sql,
            flags=re.IGNORECASE,
            count=1,
        )
        return stripped, "insert", False, False, has_ignore
    if re.search(r"\bREPLACE\b", sql, flags=re.IGNORECASE):
        has_ignore = False
        stripped = re.sub(
            r"\bREPLACE\b\s+(?:LOW_PRIORITY\s+)?(?:DELAYED\s+)?(?:INTO\s+)?",
            "INSERT ",
            sql,
            flags=re.IGNORECASE,
            count=1,
        )
        return stripped, "insert", False, False, has_ignore
    return sql, None, False, False, False


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


def _restore_dml_modifiers(
    sql: str,
    kind: str | None,
    has_low_priority: bool,
    has_quick: bool,
    has_ignore: bool,
) -> str:
    if kind == "delete":
        opts = []
        if has_low_priority:
            opts.append("LOW_PRIORITY")
        if has_quick:
            opts.append("QUICK")
        if has_ignore:
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
    if kind == "update":
        opts = []
        if has_low_priority:
            opts.append("LOW_PRIORITY")
        if has_ignore:
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
    if kind == "insert":
        if has_ignore:
            return re.sub(
                r"\bINSERT\b",
                "INSERT IGNORE",
                sql,
                flags=re.IGNORECASE,
                count=1,
            )
        return sql
    return sql


def _strip_limit_sql(
    sql: str,
    dialect: str,
    *,
    preprocess_join_types: bool = False,
) -> str:
    return ApproximateMutator(
        dialect=dialect,
        preprocess_join_types=preprocess_join_types,
    ).preprocess_statement(sql)


def compare_files_and_log(
    ori_path: str,
    *,
    dialect: str = "mysql",
    mode: str | None = None,
    compare_basis: str = "matched_rows",
    preprocess_join_types: bool = False,
    database: str | None = None,
    round_num: int | None = None,
    seed: int | None = None,
    bug_root: str = "bug",
) -> CompareResult:
    d = normalize_dialect(dialect)
    cfg = get_dialect_config(d)

    try:
        db_name = resolve_round_database(
            d,
            round_num=round_num,
            out_dir=str(Path(ori_path).resolve().parent),
            database=database or None,
        )
    except ValueError:
        db_name = database or cfg.database
        if not db_name and round_num is not None:
            db_name = get_database_name(round_num)
        if not db_name:
            return CompareResult(False, "Database name is empty; provide database or round_num.")

    round_label = _round_label(db_name, round_num)
    bug_dir = Path(bug_root) / round_label
    bug_dir.mkdir(parents=True, exist_ok=True)
    bug_path = bug_dir / "bugs.txt"
    crash_path = bug_dir / "crashes.txt"
    skipped_path = bug_dir / "skipped.txt"
    total_bug_path = Path(bug_root) / "total_bug.log"
    total_crash_path = Path(bug_root) / "total_crash.log"

    mutator = ApproximateMutator(
        dialect=d,
        seed=seed,
        preprocess_join_types=preprocess_join_types,
    )
    started_at = time.monotonic()
    heartbeat_state = _CompareHeartbeatState()
    heartbeat_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    heartbeat_thread = _start_compare_heartbeat(
        round_label=round_label,
        started_at=started_at,
        state=heartbeat_state,
        stop_event=heartbeat_stop,
        lock=heartbeat_lock,
    )

    total = 0
    failed = 0
    skipped = 0
    crashed = 0
    try:
        try:
            src = open(ori_path, "r", encoding="utf-8")
        except Exception:
            raise
        try:
            for stmt_idx, stmt in enumerate(split_sql_stream(src), 1):
                stripped = mutator.preprocess_statement(stmt)
                results = mutator.mutate_preprocessed_statement(stripped)
                for mut_idx, item in enumerate(results, 1):
                    total += 1
                    with heartbeat_lock:
                        heartbeat_state.total = total
                        heartbeat_state.failed = failed
                        heartbeat_state.skipped = skipped
                        heartbeat_state.crashed = crashed
                        heartbeat_state.stmt_idx = stmt_idx
                        heartbeat_state.mut_idx = mut_idx
                    timed_res = _compare_pair_with_timeout(
                        dialect=d,
                        db_name=db_name,
                        original_sql=stripped,
                        mutated_sql=item.mutated,
                        mode=mode or item.mode,
                        compare_basis=compare_basis,
                    )
                    res = timed_res.result
                    if timed_res.timed_out:
                        skipped += 1
                        with heartbeat_lock:
                            heartbeat_state.failed = failed
                            heartbeat_state.skipped = skipped
                            heartbeat_state.crashed = crashed
                        _write_skipped(
                            skipped_path,
                            db_name,
                            round_label,
                            item.mode,
                            stripped,
                            item.mutated,
                            res.detail,
                        )
                        continue
                    if _is_exec_error(res.detail):
                        if timed_res.crashed:
                            crashed += 1
                            with heartbeat_lock:
                                heartbeat_state.failed = failed
                                heartbeat_state.skipped = skipped
                                heartbeat_state.crashed = crashed
                            _write_crash(
                                crash_path,
                                total_crash_path,
                                db_name,
                                round_label,
                                item.mode,
                                stripped,
                                item.mutated,
                                res.detail,
                                stmt_idx=stmt_idx,
                                mut_idx=mut_idx,
                            )
                            _print_compare_heartbeat(
                                round_label=round_label,
                                total=total,
                                failed=failed,
                                skipped=skipped,
                                crashed=crashed,
                                started_at=started_at,
                                stmt_idx=stmt_idx,
                                mut_idx=mut_idx,
                            )
                            return CompareResult(
                                False,
                                (
                                    f"total={total} failed={failed} skipped={skipped} crashed={crashed} "
                                    f"crash_path={crash_path} bug_path={bug_path} skipped_path={skipped_path}"
                                ),
                            )
                        skipped += 1
                        with heartbeat_lock:
                            heartbeat_state.failed = failed
                            heartbeat_state.skipped = skipped
                            heartbeat_state.crashed = crashed
                        _write_skipped(skipped_path, db_name, round_label, item.mode, stripped, item.mutated, res.detail)
                        continue
                    if not res.ok:
                        failed += 1
                        with heartbeat_lock:
                            heartbeat_state.failed = failed
                            heartbeat_state.skipped = skipped
                            heartbeat_state.crashed = crashed
                        artifacts = _build_bug_artifacts_for_pair(
                            dialect=d,
                            db_name=db_name,
                            bug_dir=bug_dir,
                            round_label=round_label,
                            mode=item.mode,
                            original=stripped,
                            mutated=item.mutated,
                            detail=res.detail,
                            stmt_idx=stmt_idx,
                            mut_idx=mut_idx,
                        )
                        _write_bug(
                            bug_path,
                            total_bug_path,
                            db_name,
                            round_label,
                            item.mode,
                            stripped,
                            item.mutated,
                            res.detail,
                            stmt_idx=stmt_idx,
                            mut_idx=mut_idx,
                            artifacts=artifacts,
                        )
        finally:
            src.close()
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

    ok = failed == 0
    _print_compare_heartbeat(
        round_label=round_label,
        total=total,
        failed=failed,
        skipped=skipped,
        crashed=crashed,
        started_at=started_at,
    )
    detail = (
        f"total={total} failed={failed} skipped={skipped} crashed={crashed} "
        f"bug_path={bug_path} crash_path={crash_path} skipped_path={skipped_path}"
    )
    return CompareResult(ok, detail)


def compare_original_vs_mutated(
    original_sql: str,
    mutated_sql: str,
    *,
    dialect: str = "mysql",
    mode: str | None = None,
    compare_basis: str = "matched_rows",
    database: str | None = None,
    round_num: int | None = None,
    out_dir: str | None = None,
) -> CompareResult:
    """
    Execute original and mutated DML in an isolated transaction, then compare row sets.

    mode:
      - "over": mutated set should be a superset of original
      - "under": mutated set should be a subset of original
      - None: sets must be equal

    compare_basis:
      - "matched_rows": compare matched PKs from the original/mutated statement shape
      - "changed_rows": compare rows whose persisted contents changed after execution
    """
    d = normalize_dialect(dialect)
    cfg = get_dialect_config(d)

    try:
        db_name = resolve_round_database(
            d,
            round_num=round_num,
            out_dir=out_dir,
            database=database or None,
        )
    except ValueError:
        db_name = database or cfg.database
        if not db_name and round_num is not None:
            db_name = get_database_name(round_num)
        if not db_name:
            return CompareResult(False, "Database name is empty; provide database or round_num.")

    orig = _parse_dml(original_sql, d)
    if orig is None:
        return CompareResult(False, "Unsupported or invalid original DML.")
    mut = _parse_dml(mutated_sql, d)
    if mut is None:
        return CompareResult(False, "Unsupported or invalid mutated DML.")

    if orig.kind != mut.kind or orig.table != mut.table:
        return CompareResult(False, "Original and mutated statements are not comparable (kind/table mismatch).")

    conn = connect_db(d, database=db_name)

    try:
        cur = conn.cursor()
        try:
            res = _compare_pair_with_cursor(
                conn,
                cur,
                original_sql,
                mutated_sql,
                mode,
                d,
                compare_basis=compare_basis,
            )
        finally:
            _safe_close(cur)

    finally:
        _safe_close(conn)

    return res


def _parse_dml(sql: str, dialect: str) -> _ParsedDml | None:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return None
    parse_sql, _, _, _, _ = _strip_dml_modifiers(sql)
    parse_sql = normalize_sqlglot_input(parse_sql, dialect)
    parse_dialect = get_sqlglot_dialect(dialect)
    try:
        stmt = sqlglot.parse_one(parse_sql, read=parse_dialect)
    except Exception:
        return None

    schema = get_schema()

    if isinstance(stmt, exp.Update):
        table = stmt.this
        table_name = table.name
        if table_name not in schema:
            return None
        pk = schema[table_name]["pk"]
        where_sql = _where_sql(stmt, dialect)
        order_sql = _order_sql(stmt, dialect)
        limit_sql = _limit_sql(stmt, dialect)
        set_sql = _set_sql(stmt, dialect)
        if not set_sql:
            return None
        targets = _update_targets(stmt, table_name)
        return _ParsedDml(
            kind="update",
            table=table_name,
            table_sql=table.sql(dialect=parse_dialect),
            pk=pk,
            where_sql=where_sql,
            order_sql=order_sql,
            limit_sql=limit_sql,
            set_sql=set_sql,
            raw_sql=sql,
            targets=targets,
            top_level_update_join_sqls=_top_level_update_join_sqls(stmt, dialect),
            match_sql=_build_update_match_sql(stmt, table_name, pk, dialect),
        )

    if isinstance(stmt, exp.Insert):
        target = stmt.this
        if isinstance(target, exp.Schema):
            table_expr = target.this
        else:
            table_expr = target
        table_name = table_expr.name if isinstance(table_expr, exp.Table) else str(table_expr)
        if table_name not in schema:
            return None
        pk = schema[table_name]["pk"]
        return _ParsedDml(
            kind="insert",
            table=table_name,
            table_sql=table_expr.sql(dialect=parse_dialect) if isinstance(table_expr, exp.Table) else table_name,
            pk=pk,
            where_sql="",
            order_sql="",
            limit_sql="",
            set_sql=None,
            raw_sql=sql,
            targets=[table_name],
            top_level_update_join_sqls=[],
            match_sql=None,
        )

    if isinstance(stmt, exp.Delete):
        table = stmt.this
        table_name = table.name if isinstance(table, exp.Table) else str(table)
        if table_name not in schema:
            return None
        pk = schema[table_name]["pk"]
        where_sql = _where_sql(stmt, dialect)
        order_sql = _order_sql(stmt, dialect)
        limit_sql = _limit_sql(stmt, dialect)
        targets = _delete_targets(stmt, table_name)
        return _ParsedDml(
            kind="delete",
            table=table_name,
            table_sql=table.sql(dialect=parse_dialect) if isinstance(table, exp.Table) else table_name,
            pk=pk,
            where_sql=where_sql,
            order_sql=order_sql,
            limit_sql=limit_sql,
            set_sql=None,
            raw_sql=sql,
            targets=targets,
            top_level_update_join_sqls=[],
            match_sql=None,
        )

    return None


def _where_sql(stmt: exp.Expression, dialect: str) -> str:
    where = stmt.args.get("where")
    if where is None:
        return "TRUE"
    return _normalize_rendered_sql(where.this.sql(dialect=get_sqlglot_dialect(dialect)), dialect)


def _order_sql(stmt: exp.Expression, dialect: str) -> str:
    order = stmt.args.get("order")
    if order is None:
        return ""
    return " " + _normalize_rendered_sql(order.sql(dialect=get_sqlglot_dialect(dialect)), dialect)


def _limit_sql(stmt: exp.Expression, dialect: str) -> str:
    limit = stmt.args.get("limit")
    if limit is None:
        return ""
    return " " + _normalize_rendered_sql(limit.sql(dialect=get_sqlglot_dialect(dialect)), dialect)


def _with_sql(stmt: exp.Expression, dialect: str) -> str:
    with_ = stmt.args.get("with_")
    if with_ is None:
        return ""
    return _normalize_rendered_sql(with_.sql(dialect=get_sqlglot_dialect(dialect)), dialect) + " "


def _set_sql(stmt: exp.Update, dialect: str) -> str:
    exprs = stmt.expressions or []
    if not exprs:
        return ""
    render_dialect = get_sqlglot_dialect(dialect)
    return ", ".join(_normalize_rendered_sql(e.sql(dialect=render_dialect), dialect) for e in exprs)


def _normalize_rendered_sql(sql: str, dialect: str) -> str:
    if dialect in {"mysql", "mariadb", "percona", "oceanbase", "tidb", "polardb"}:
        return _restore_mysql_regexp_like(sql)
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




def _alias_map(stmt: exp.Expression) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table in stmt.find_all(exp.Table):
        name = table.name
        alias = table.alias_or_name
        if alias:
            mapping[alias] = name
        mapping[name] = name
    return mapping


def _top_level_update_join_sqls(stmt: exp.Update, dialect: str) -> list[str]:
    joins = list(stmt.args.get("joins") or [])
    if not joins and isinstance(stmt.this, exp.Table):
        joins = list(stmt.this.args.get("joins") or [])
    render_dialect = get_sqlglot_dialect(dialect)
    return [_normalize_rendered_sql(join.sql(dialect=render_dialect), dialect) for join in joins]


def _build_update_match_sql(stmt: exp.Update, table_name: str, pk: str, dialect: str) -> str | None:
    table_expr = stmt.this
    if not isinstance(table_expr, exp.Table):
        return None
    render_dialect = get_sqlglot_dialect(dialect)
    with_sql = _with_sql(stmt, dialect)
    target_ref = table_expr.alias_or_name or table_name
    # sqlglot stores MySQL/MariaDB UPDATE JOIN trees on the target Table node.
    # Render the base target table here and append joins exactly once below.
    base_table_expr = table_expr.copy()
    base_table_expr.set("joins", None)
    target_sql = _normalize_rendered_sql(base_table_expr.sql(dialect=render_dialect), dialect)
    joins = list(stmt.args.get("joins") or [])
    if not joins and isinstance(table_expr, exp.Table):
        joins = list(table_expr.args.get("joins") or [])
    joins_sql = "".join(f" {_normalize_rendered_sql(join.sql(dialect=render_dialect), dialect)}" for join in joins)
    from_clause = stmt.args.get("from_")
    extra_from_sql = ""
    if isinstance(from_clause, exp.From):
        sources: list[str] = []
        if isinstance(from_clause.this, exp.Expression):
                sources.append(_normalize_rendered_sql(from_clause.this.sql(dialect=render_dialect), dialect))
        for expr in from_clause.expressions or []:
            if isinstance(expr, exp.Expression):
                sources.append(_normalize_rendered_sql(expr.sql(dialect=render_dialect), dialect))
        if sources:
            extra_from_sql = ", " + ", ".join(sources)
    where_sql = _where_sql(stmt, dialect)
    target_pk_sql = exp.column(pk, table=target_ref).sql(dialect=render_dialect)
    where_sql = f"({where_sql}) AND ({target_pk_sql} IS NOT NULL)"
    order_sql = _order_sql(stmt, dialect)
    limit_sql = _limit_sql(stmt, dialect)
    sql = (
        f"{with_sql}SELECT DISTINCT {target_ref}.{pk} AS pk "
        f"FROM {target_sql}{joins_sql}{extra_from_sql} "
        f"WHERE {where_sql}{order_sql}{limit_sql}"
    )
    return _normalize_rendered_sql(sql, dialect)


def _update_targets(stmt: exp.Update, default_table: str) -> list[str]:
    schema = get_schema()
    mapping = _alias_map(stmt)
    targets: list[str] = []
    for expr in stmt.expressions or []:
        col = expr.this
        if isinstance(col, exp.Column):
            table_ref = col.table or default_table
            table_name = mapping.get(table_ref, table_ref)
            if table_name in schema and table_name not in targets:
                targets.append(table_name)
    if not targets and default_table in schema:
        targets.append(default_table)
    return targets


def _delete_targets(stmt: exp.Delete, default_table: str) -> list[str]:
    schema = get_schema()
    targets: list[str] = []
    tables = stmt.args.get("tables") or []
    if tables:
        for table in tables:
            if isinstance(table, exp.Table):
                name = table.name
                if name in schema and name not in targets:
                    targets.append(name)
    if not targets:
        if isinstance(stmt.this, exp.Table):
            name = stmt.this.name
            if name in schema:
                targets.append(name)
        elif default_table in schema:
            targets.append(default_table)
    return targets


def _effective_col_type(schema: dict, table: str, col: str, dialect: str) -> str | None:
    col_type = schema[table]["columns"].get(col)
    if col_type == "geom" and not dialect_option(dialect, "supports_geometry", True):
        return "str"
    return col_type


def _hash_expr(table: str, schema: dict, dialect: str, alias: str | None = None) -> str:
    cols = list(schema[table]["columns"].keys())
    prefix = f"{alias}." if alias else ""
    parts = []
    driver = connection_driver(dialect)
    for col in cols:
        col_type = _effective_col_type(schema, table, col, dialect)
        if col_type == "geom":
            # MariaDB/MySQL reject CAST(geometry AS CHAR); stringify geometry explicitly.
            if driver == "duckdb":
                parts.append(f"COALESCE(CAST({prefix}{col} AS VARCHAR), '<NULL>')")
            elif driver == "monetdb":
                parts.append(f"IFNULL(CAST({prefix}{col} AS TEXT), '<NULL>')")
            else:
                parts.append(f"IFNULL(ST_AsText({prefix}{col}), '<NULL>')")
        elif col_type == "bin":
            # Binary columns may contain invalid utf8mb4 bytes; hash their hex form instead.
            if driver == "duckdb":
                parts.append(f"COALESCE(HEX({prefix}{col}), '<NULL>')")
            elif driver == "monetdb":
                parts.append(f"IFNULL(CAST({prefix}{col} AS TEXT), '<NULL>')")
            else:
                parts.append(f"IFNULL(HEX({prefix}{col}), '<NULL>')")
        elif col_type == "json" and driver == "monetdb":
            parts.append(f"IFNULL(CAST({prefix}{col} AS TEXT), '<NULL>')")
        else:
            if driver == "duckdb":
                parts.append(f"COALESCE(CAST({prefix}{col} AS VARCHAR), '<NULL>')")
            elif driver == "monetdb":
                parts.append(f"IFNULL(CAST({prefix}{col} AS TEXT), '<NULL>')")
            else:
                parts.append(f"IFNULL(CAST({prefix}{col} AS CHAR), '<NULL>')")
    if driver == "monetdb":
        joined = " || '#' || ".join(parts)
        return f"MD5({joined})"
    return f"MD5(CONCAT_WS('#', {', '.join(parts)}))"


def _fetch_table_snapshot(cur, table: str, schema: dict, kind: str, dialect: str) -> dict:
    pk = schema[table]["pk"]
    del kind  # All DML kinds now compare full-row snapshots.
    hash_expr = _hash_expr(table, schema, dialect, alias="t")
    cur.execute(f"SELECT t.{pk} AS pk, {hash_expr} AS h FROM {table} t")
    return {row[0]: row[1] for row in cur.fetchall()}


def _row_effect_token(
    table: str,
    pk,
    before_hash: str | None,
    after_hash: str | None,
    *,
    pk_only: bool = False,
) -> str | None:
    if before_hash == after_hash:
        return None
    if pk_only:
        return f"{table}:{pk}"
    if after_hash is None:
        return f"{table}:{pk}:DELETE:{before_hash}"
    return f"{table}:{pk}:ROW:{after_hash}"


def _token_pk(token: str) -> str:
    parts = token.split(":", 3)
    if len(parts) < 2:
        return token
    return f"{parts[0]}:{parts[1]}"


def _execute_and_collect(
    cur,
    parsed: _ParsedDml,
    temp_table: str,
    parsed_dialect: str,
    *,
    pk_only: bool = False,
) -> set:
    del temp_table  # Kept for call-site compatibility.

    schema = get_schema()
    affected: set[str] = set()
    before = {
        table: _fetch_table_snapshot(cur, table, schema, parsed.kind, parsed_dialect)
        for table in parsed.targets
    }

    cur.execute(parsed.raw_sql)
    if getattr(cur, "description", None) is not None:
        fetchall = getattr(cur, "fetchall", None)
        if callable(fetchall):
            try:
                fetchall()
            except Exception:
                pass

    for table in parsed.targets:
        after = _fetch_table_snapshot(cur, table, schema, parsed.kind, parsed_dialect)
        changed_pks = set(before[table]) | set(after)
        for pk in changed_pks:
            token = _row_effect_token(
                table,
                pk,
                before[table].get(pk),
                after.get(pk),
                pk_only=pk_only,
            )
            if token is not None:
                affected.add(token)

    return affected


def _execute_and_collect_matched_rows(cur, parsed: _ParsedDml) -> set[str]:
    if not parsed.match_sql:
        return set()
    cur.execute(parsed.match_sql)
    return {
        f"{parsed.table}:{row[0]}"
        for row in cur.fetchall()
        if row and row[0] is not None
    }


def _execute_and_collect_matched_rows_isolated(
    conn,
    parsed: _ParsedDml,
    dialect: str,
) -> set[str]:
    cur = conn.cursor()
    try:
        _begin_transaction(cur, dialect)
        try:
            return _execute_and_collect_matched_rows(cur, parsed)
        finally:
            _rollback_transaction(conn, cur)
    finally:
        _safe_close(cur)


def _compare_sets(orig_set: set, mut_set: set, mode: str | None) -> tuple[bool, str]:
    if mode == "over":
        ok = orig_set.issubset(mut_set)
    elif mode == "under":
        ok = mut_set.issubset(orig_set)
    else:
        ok = orig_set == mut_set

    only_orig = sorted(orig_set - mut_set)
    only_mut = sorted(mut_set - orig_set)
    only_orig_pk = sorted({_token_pk(token) for token in only_orig})
    only_mut_pk = sorted({_token_pk(token) for token in only_mut})
    detail = (
        f"orig={len(orig_set)} mut={len(mut_set)} "
        f"only_orig={len(only_orig)} only_mut={len(only_mut)} "
        f"only_orig_pk={only_orig_pk} only_mut_pk={only_mut_pk} "
        f"only_orig_row={only_orig} only_mut_row={only_mut}"
    )
    return ok, detail

def _effective_compare_basis(requested_basis: str, orig: _ParsedDml, mut: _ParsedDml) -> str:
    if requested_basis == "matched_rows":
        if orig.match_sql and mut.match_sql:
            return "matched_rows"
        return "changed_rows"
    return "changed_rows"


def _parse_sqlglot_stmt(sql: str, dialect: str) -> exp.Expression | None:
    stmt_sql = (sql or "").strip().rstrip(";")
    if not stmt_sql:
        return None
    parse_sql, _, _, _, _ = _strip_dml_modifiers(stmt_sql)
    parse_sql = normalize_sqlglot_input(parse_sql, dialect)
    try:
        return sqlglot.parse_one(parse_sql, read=get_sqlglot_dialect(dialect))
    except Exception:
        return None


def _setop_kind(node: exp.SetOperation) -> str:
    if isinstance(node, exp.Union):
        return "union_all" if not bool(node.args.get("distinct", True)) else "union"
    if isinstance(node, exp.Except):
        return "except"
    if isinstance(node, exp.Intersect):
        return "intersect"
    return node.key


def _uses_pk_only_for_union_like_setop_mutation(
    original_sql: str,
    mutated_sql: str,
    dialect: str,
) -> bool:
    original_stmt = _parse_sqlglot_stmt(original_sql, dialect)
    mutated_stmt = _parse_sqlglot_stmt(mutated_sql, dialect)
    if original_stmt is None or mutated_stmt is None:
        return False

    original_setops = [_setop_kind(node) for node in original_stmt.find_all(exp.SetOperation)]
    mutated_setops = [_setop_kind(node) for node in mutated_stmt.find_all(exp.SetOperation)]
    if len(original_setops) != len(mutated_setops):
        return False

    union_like = {"union", "union_all"}
    return any(
        orig_kind not in union_like and mut_kind in union_like
        for orig_kind, mut_kind in zip(original_setops, mutated_setops)
    )

def _safe_execute(cur, sql: str) -> bool:
    try:
        cur.execute(sql)
        return True
    except Exception:
        return False


def _format_result_rows(description, rows) -> str:
    lines: list[str] = []
    cols = [str(col[0]) for col in (description or []) if col]
    if cols:
        lines.append("\t".join(cols))
    for row in rows or []:
        if not isinstance(row, (list, tuple)):
            row = (row,)
        lines.append("\t".join("" if value is None else str(value) for value in row))
    return "\n".join(lines).strip()


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _explain_sql(cur, sql: str) -> str:
    stmt = (sql or "").strip().rstrip(";")
    if not stmt:
        return ""
    try:
        cur.execute(f"EXPLAIN {stmt}")
        rows = cur.fetchall()
        rendered = _format_result_rows(getattr(cur, "description", None), rows)
        return rendered or "<empty plan>"
    except Exception as exc:
        return f"EXPLAIN_ERROR: {type(exc).__name__}: {exc}"


def _duckdb_catalog_name(cur, db_name: str) -> str:
    target_path = str(Path(resolve_duckdb_database_path(db_name))).replace("\\\\?\\", "").lower()
    try:
        cur.execute("PRAGMA database_list")
        rows = cur.fetchall()
    except Exception:
        return "main"

    fallback = "main"
    for row in rows or []:
        if len(row) >= 2:
            fallback = str(row[1])
        if len(row) >= 3:
            candidate = str(row[2]).replace("\\\\?\\", "").lower()
            if candidate == target_path:
                return str(row[1])
    return fallback


def _copy_database_snapshot(cur, db_name: str, dialect: str, snapshot_dir: Path) -> list[str]:
    snapshot_paths: list[str] = []
    if connection_driver(dialect) != "duckdb":
        return snapshot_paths

    try:
        db_path = Path(resolve_duckdb_database_path(db_name))
    except Exception as exc:
        return [f"<snapshot_error: {type(exc).__name__}: {exc}>"]

    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return [f"<snapshot_error: {type(exc).__name__}: {exc}>"]

    snapshot_db = snapshot_dir / db_path.name
    snapshot_alias = "__bugsnap"
    source_catalog = _duckdb_catalog_name(cur, db_name)
    attached = False
    try:
        cur.execute(f"ATTACH {_quote_sql_string(snapshot_db.as_posix())} AS {_quote_identifier(snapshot_alias)}")
        attached = True
        cur.execute(
            f"COPY FROM DATABASE {_quote_identifier(source_catalog)} TO {_quote_identifier(snapshot_alias)}"
        )
        snapshot_paths.append(str(snapshot_db))
        return snapshot_paths
    except Exception as exc:
        snapshot_paths.append(f"<snapshot_error: {type(exc).__name__}: {exc}>")
    finally:
        if attached:
            try:
                cur.execute(f"DETACH {_quote_identifier(snapshot_alias)}")
            except Exception:
                pass

    candidates = [db_path, Path(str(db_path) + ".wal")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        target = snapshot_dir / candidate.name
        try:
            shutil.copy2(candidate, target)
            snapshot_paths.append(str(target))
        except Exception as exc:
            snapshot_paths.append(f"<snapshot_error: {candidate}: {type(exc).__name__}: {exc}>")
    return snapshot_paths


def _build_bug_artifacts(
    *,
    cur,
    bug_dir: Path,
    db_name: str,
    round_label: str,
    mode: str,
    original: str,
    mutated: str,
    detail: str,
    dialect: str,
    stmt_idx: int,
    mut_idx: int,
) -> _BugArtifacts:
    case_dir = bug_dir / "cases" / f"stmt-{stmt_idx:05d}-mut-{mut_idx:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    original_sql_path = case_dir / "original.sql"
    mutated_sql_path = case_dir / "mutated.sql"
    original_plan_path = case_dir / "original.explain.txt"
    mutated_plan_path = case_dir / "mutated.explain.txt"
    snapshot_dir = case_dir / "db"
    meta_path = case_dir / "meta.txt"

    original_sql_path.write_text(original.strip().rstrip(";") + ";\n", encoding="utf-8")
    mutated_sql_path.write_text(mutated.strip().rstrip(";") + ";\n", encoding="utf-8")

    original_plan = _explain_sql(cur, original)
    mutated_plan = _explain_sql(cur, mutated)
    original_plan_path.write_text(original_plan + "\n", encoding="utf-8")
    mutated_plan_path.write_text(mutated_plan + "\n", encoding="utf-8")

    snapshot_paths = _copy_database_snapshot(cur, db_name, dialect, snapshot_dir)

    meta_lines = [
        f"database: {db_name}",
        f"round: {round_label}",
        f"mode: {mode}",
        f"stmt_idx: {stmt_idx}",
        f"mutation_idx: {mut_idx}",
        f"detail: {detail}",
        f"original_sql: {original_sql_path}",
        f"mutated_sql: {mutated_sql_path}",
        f"original_plan: {original_plan_path}",
        f"mutated_plan: {mutated_plan_path}",
    ]
    if snapshot_paths:
        meta_lines.append("snapshots:")
        meta_lines.extend(snapshot_paths)
    else:
        meta_lines.append("snapshots: <unavailable>")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    return _BugArtifacts(
        bundle_dir=str(case_dir),
        snapshot_paths=snapshot_paths,
        original_plan_path=str(original_plan_path),
        mutated_plan_path=str(mutated_plan_path),
    )


def _build_bug_artifacts_for_pair(
    *,
    dialect: str,
    db_name: str,
    bug_dir: Path,
    round_label: str,
    mode: str,
    original: str,
    mutated: str,
    detail: str,
    stmt_idx: int,
    mut_idx: int,
) -> _BugArtifacts | None:
    conn = None
    cur = None
    try:
        conn = connect_db(dialect, database=db_name)
        cur = conn.cursor()
        return _build_bug_artifacts(
            cur=cur,
            bug_dir=bug_dir,
            db_name=db_name,
            round_label=round_label,
            mode=mode,
            original=original,
            mutated=mutated,
            detail=detail,
            dialect=dialect,
            stmt_idx=stmt_idx,
            mut_idx=mut_idx,
        )
    except Exception:
        return None
    finally:
        _safe_close(cur)
        _safe_close(conn)


def _connection_alive(conn) -> bool:
    try:
        ping = getattr(conn, "ping", None)
        if callable(ping):
            ping(reconnect=False)
        else:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _begin_transaction(cur, dialect: str) -> None:
    if connection_driver(dialect) == "duckdb":
        cur.execute("BEGIN TRANSACTION")
        return
    if connection_driver(dialect) == "monetdb":
        # pymonetdb with autocommit=False already runs inside a transaction.
        # Emitting BEGIN TRANSACTION here causes "cannot start a transaction
        # within a transaction" and leads to mass skips.
        return
    cur.execute("START TRANSACTION")


def _rollback_transaction(conn, cur) -> None:
    if _safe_execute(cur, "ROLLBACK"):
        return
    rollback = getattr(conn, "rollback", None)
    if callable(rollback):
        try:
            rollback()
        except Exception:
            return


def _compare_pair_with_cursor(
    conn,
    cur,
    original_sql: str,
    mutated_sql: str,
    mode: str | None,
    dialect: str,
    *,
    compare_basis: str = "matched_rows",
) -> CompareResult:
    orig = _parse_dml(original_sql, dialect)
    if orig is None:
        return CompareResult(False, "Unsupported or invalid original DML.")
    mut = _parse_dml(mutated_sql, dialect)
    if mut is None:
        return CompareResult(False, "Unsupported or invalid mutated DML.")

    if orig.kind != mut.kind or orig.table != mut.table:
        return CompareResult(False, "Original and mutated statements are not comparable (kind/table mismatch).")

    effective_basis = _effective_compare_basis(compare_basis, orig, mut)
    pk_only_compare = (
        effective_basis == "changed_rows"
        and _uses_pk_only_for_union_like_setop_mutation(original_sql, mutated_sql, dialect)
    )
    try:
        if effective_basis == "matched_rows":
            # Run each match_sql SELECT in its own short transaction window so
            # any engine-internal intermediates are scoped to a single query.
            orig_set = _execute_and_collect_matched_rows_isolated(conn, orig, dialect)
            mut_set = _execute_and_collect_matched_rows_isolated(conn, mut, dialect)
        else:
            _begin_transaction(cur, dialect)
            try:
                orig_set = _execute_and_collect(
                    cur,
                    orig,
                    "affected_pk_orig",
                    dialect,
                    pk_only=pk_only_compare,
                )
            finally:
                _rollback_transaction(conn, cur)

            _begin_transaction(cur, dialect)
            try:
                mut_set = _execute_and_collect(
                    cur,
                    mut,
                    "affected_pk_mut",
                    dialect,
                    pk_only=pk_only_compare,
                )
            finally:
                _rollback_transaction(conn, cur)
    except Exception as exc:
        _rollback_transaction(conn, cur)
        return CompareResult(False, f"EXEC_ERROR: {exc}")

    ok, detail = _compare_sets(orig_set, mut_set, mode)
    detail = f"compare_basis={effective_basis} " + detail
    if effective_basis == "changed_rows":
        compare_mode = "pk_only" if pk_only_compare else "row_hash_set"
        detail = f"compare_mode={compare_mode} " + detail
    return CompareResult(ok, detail)


def _round_label(db_name: str, round_num: int | None) -> str:
    if round_num is not None:
        return f"round-{round_num:03d}"
    if db_name.startswith("databse"):
        suffix = db_name.replace("databse", "")
        if suffix.isdigit():
            return f"round-{int(suffix):03d}"
    return "round-unknown"


def _write_bug(
    path: Path,
    total_path: Path | None,
    db_name: str,
    round_label: str,
    mode: str,
    original: str,
    mutated: str,
    detail: str,
    *,
    stmt_idx: int,
    mut_idx: int,
    artifacts: _BugArtifacts | None = None,
) -> None:
    payload = [
        "-- BUG\n",
        f"-- database: {db_name}\n",
        f"-- round: {round_label}\n",
        f"-- mode: {mode}\n",
        f"-- stmt_idx: {stmt_idx}\n",
        f"-- mutation_idx: {mut_idx}\n",
        f"-- detail: {detail}\n",
    ]
    if artifacts is not None:
        payload.extend([
            f"-- bundle_dir: {artifacts.bundle_dir}\n",
            f"-- original_plan: {artifacts.original_plan_path or '<unavailable>'}\n",
            f"-- mutated_plan: {artifacts.mutated_plan_path or '<unavailable>'}\n",
            f"-- snapshots: {artifacts.snapshot_paths or ['<unavailable>']}\n",
        ])
    payload.extend([
        "ORIGINAL:\n",
        original.strip().rstrip(";") + ";\n",
        "MUTATED:\n",
        mutated.strip().rstrip(";") + ";\n\n",
    ])
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(payload)
    if total_path is not None:
        total_path.parent.mkdir(parents=True, exist_ok=True)
        with open(total_path, "a", encoding="utf-8") as f:
            f.writelines(payload)


def _is_exec_error(detail: str) -> bool:
    return detail.startswith("EXEC_ERROR:")


def _is_crash_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(pattern in lowered for pattern in _CRASH_PATTERNS)


def _write_crash(
    path: Path,
    total_path: Path | None,
    db_name: str,
    round_label: str,
    mode: str,
    original: str,
    mutated: str,
    detail: str,
    *,
    stmt_idx: int,
    mut_idx: int,
) -> None:
    payload = [
        "-- CRASH\n",
        f"-- database: {db_name}\n",
        f"-- round: {round_label}\n",
        f"-- mode: {mode}\n",
        f"-- stmt_idx: {stmt_idx}\n",
        f"-- mutation_idx: {mut_idx}\n",
        f"-- detail: {detail}\n",
        "ORIGINAL:\n",
        original.strip().rstrip(";") + ";\n",
        "MUTATED:\n",
        mutated.strip().rstrip(";") + ";\n\n",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(payload)
    if total_path is not None:
        total_path.parent.mkdir(parents=True, exist_ok=True)
        with open(total_path, "a", encoding="utf-8") as f:
            f.writelines(payload)


def _write_skipped(path: Path, db_name: str, round_label: str, mode: str, original: str, mutated: str, detail: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("-- SKIPPED\n")
        f.write(f"-- database: {db_name}\n")
        f.write(f"-- round: {round_label}\n")
        f.write(f"-- mode: {mode}\n")
        f.write(f"-- detail: {detail}\n")
        f.write("ORIGINAL:\n")
        f.write(original.strip().rstrip(";") + ";\n")
        f.write("MUTATED:\n")
        f.write(mutated.strip().rstrip(";") + ";\n\n")
