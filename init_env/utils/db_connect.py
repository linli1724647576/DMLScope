import os
import re
from pathlib import Path

from init_env.core.dialect import dialect_option, get_database_name, get_dialect_config, normalize_dialect
from init_env.utils.monetdb_admin import monetdb_round_database_name


def connection_driver(dialect: str) -> str:
    d = normalize_dialect(dialect)
    return str(dialect_option(d, "driver", "mysql"))


def _translate_wsl_path(path: str) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
    if not match:
        return path
    drive = match.group(1).upper()
    tail = match.group(2).replace("/", "\\")
    return f"{drive}:\\{tail}"


def resolve_duckdb_database_path(path: str) -> str:
    if not path:
        raise ValueError("DuckDB database path is empty.")
    translated = _translate_wsl_path(path)
    return str(Path(translated))


def resolve_round_database(
    dialect: str,
    *,
    round_num: int | None = None,
    out_dir: str | None = None,
    database: str | None = None,
) -> str:
    d = normalize_dialect(dialect)
    cfg = get_dialect_config(d)

    if database:
        return resolve_duckdb_database_path(database) if connection_driver(d) == "duckdb" else database

    if connection_driver(d) == "duckdb":
        if out_dir:
            return resolve_duckdb_database_path(str(Path(out_dir) / "test.db"))
        if cfg.database:
            return resolve_duckdb_database_path(cfg.database)
        raise ValueError("DuckDB database path is empty; provide out_dir or explicit database.")

    if connection_driver(d) == "monetdb":
        if round_num is not None:
            return monetdb_round_database_name(round_num, base_database=cfg.database or "mtdb")
        if cfg.database:
            return cfg.database
        raise ValueError("Database is empty for MonetDB; provide round_num or explicit database.")

    if dialect_option(d, "fixed_database", False) and cfg.database:
        return cfg.database

    if round_num is not None:
        return get_database_name(round_num)
    if cfg.database:
        return cfg.database
    raise ValueError(f"Database is empty for dialect={d}.")


def reset_duckdb_database(path: str) -> None:
    db_path = Path(resolve_duckdb_database_path(path))
    wal_path = Path(str(db_path) + ".wal")
    for candidate in (db_path, wal_path):
        if candidate.exists():
            candidate.unlink()


def connect_db(dialect: str, *, database: str | None = None, read_only: bool = False):
    d = normalize_dialect(dialect)
    cfg = get_dialect_config(d)
    driver = connection_driver(d)

    if driver == "duckdb":
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("duckdb is required to connect to DuckDB databases.") from exc

        target = resolve_duckdb_database_path(database or cfg.database)
        parent = Path(target).parent
        if not read_only and parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(target, read_only=read_only)

    if driver == "monetdb":
        try:
            import pymonetdb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pymonetdb is required to connect to MonetDB databases.") from exc

        target_database = database or cfg.database or None
        if not target_database:
            raise ValueError("Database is empty for MonetDB; please fill dialect config before execution.")

        return pymonetdb.connect(
            username=cfg.user,
            password=cfg.password,
            hostname=cfg.host,
            port=cfg.port or 50000,
            database=target_database,
            autocommit=False,
        )

    if not cfg.host:
        raise ValueError("Database host is empty; please fill dialect config before execution.")

    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymysql is required to connect to MySQL-compatible databases.") from exc

    return pymysql.connect(
        host=cfg.host,
        port=cfg.port or 3306,
        user=cfg.user,
        password=cfg.password,
        database=database or cfg.database or None,
        autocommit=False,
        charset="utf8mb4",
    )
