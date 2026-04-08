import os

from init_env.core.dialect import dialect_option, get_dialect_config
from init_env.utils.db_connect import connect_db, connection_driver


def _drain_result_set(cur) -> None:
    if getattr(cur, "description", None) is None:
        return
    fetchall = getattr(cur, "fetchall", None)
    if callable(fetchall):
        try:
            fetchall()
        except Exception:
            pass


def _split_sql(sql: str) -> list:
    statements = []
    buf = []
    in_single = False
    in_double = False
    in_backtick = False
    escape = False

    for ch in sql:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == "'" and not in_double and not in_backtick:
            in_single = not in_single
        elif ch == '"' and not in_single and not in_backtick:
            in_double = not in_double
        elif ch == "`" and not in_single and not in_double:
            in_backtick = not in_backtick

        if ch == ";" and not in_single and not in_double and not in_backtick:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _needs_server_level_connection(dialect: str, sql_texts: list[str]) -> bool:
    if connection_driver(dialect) == "duckdb":
        return False
    if not dialect_option(dialect, "uses_database_statements", True):
        return False

    markers = (
        "CREATE DATABASE",
        "DROP DATABASE",
        "USE ",
    )
    for sql in sql_texts:
        upper_sql = sql.upper()
        if any(marker in upper_sql for marker in markers):
            return True
    return False


def execute_sql_files(
    dialect: str,
    paths: list[str],
    *,
    database: str | None = None,
    raise_on_error: bool = False,
) -> dict:
    cfg = get_dialect_config(dialect)
    driver = connection_driver(dialect)
    target_database = database or cfg.database or None
    sql_payloads = [(path, open(path, "r", encoding="utf-8").read()) for path in paths]
    bootstrap_connect = _needs_server_level_connection(
        dialect, [sql for _, sql in sql_payloads]
    )
    connect_database = None if bootstrap_connect else target_database

    if driver == "duckdb":
        print(f"[execute] connect driver=duckdb file={connect_database}")
    else:
        print(
            f"[execute] connect host={cfg.host} port={cfg.port or 3306} "
            f"user={cfg.user} db={connect_database or ''}"
        )

    try:
        conn = connect_db(dialect, database=connect_database)
    except Exception:
        if driver == "duckdb" or bootstrap_connect or not target_database:
            raise
        print(
            f"[execute] connect to db={target_database} failed, retry without database"
        )
        conn = connect_db(dialect, database=None)
    total_failures = 0
    total_success = 0
    total_affected = 0
    first_failure: tuple[str, str, str] | None = None
    file_summaries: list[dict] = []
    try:
        cur = conn.cursor()
        try:
            for path, sql in sql_payloads:
                print(f"[execute] file: {path}")
                count = 0
                affected = 0
                failures = 0
                for stmt in _split_sql(sql):
                    try:
                        cur.execute(stmt)
                        _drain_result_set(cur)
                        count += 1
                        total_success += 1
                        rowcount = getattr(cur, "rowcount", None)
                        if rowcount is not None and rowcount >= 0:
                            affected += rowcount
                            total_affected += rowcount
                        if driver == "monetdb":
                            conn.commit()
                        print("[execute] stmt ok")
                    except Exception as exc:
                        failures += 1
                        total_failures += 1
                        if first_failure is None:
                            first_failure = (path, stmt, str(exc))
                        if driver == "monetdb":
                            rollback = getattr(conn, "rollback", None)
                            if callable(rollback):
                                try:
                                    rollback()
                                except Exception:
                                    pass
                        print(f"[execute] stmt failed: {exc}")
                print(
                    f"[execute] statements executed: {count} failed: {failures} affected rows: {affected}"
                )
                print(f"[execute] file done: {path}")
                file_summaries.append(
                    {
                        "path": path,
                        "success": count,
                        "failed": failures,
                        "affected": affected,
                    }
                )
        finally:
            close = getattr(cur, "close", None)
            if callable(close):
                close()
        commit = getattr(conn, "commit", None)
        if callable(commit):
            commit()
    finally:
        conn.close()
        print("[execute] connection closed")
    summary = {
        "success": total_success,
        "failed": total_failures,
        "affected": total_affected,
        "files": file_summaries,
        "first_failure": None,
    }
    if first_failure is not None:
        path, stmt, error = first_failure
        summary["first_failure"] = {
            "path": path,
            "error": error,
            "stmt": stmt[:300],
        }
    if raise_on_error and total_failures:
        assert summary["first_failure"] is not None
        raise RuntimeError(
            "SQL execution failed during rebuild: "
            f"path={summary['first_failure']['path']} "
            f"error={summary['first_failure']['error']} "
            f"stmt={summary['first_failure']['stmt']}"
        )
    return summary
