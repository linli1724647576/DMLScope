import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from init_env.builders.ddl_builder import build_ddl
from init_env.builders.dml_builder import build_dml_with_meta
from init_env.core.dialect import normalize_dialect
from init_env.core.schema import get_schema
from init_env.utils.db_connect import connection_driver, reset_duckdb_database, resolve_round_database
from init_env.utils.io_utils import write_file
from init_env.utils.monetdb_admin import ensure_monetdb_database
from init_env.utils.sql_executor import execute_sql_files


def _ensure_root_on_path() -> str:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT


def generate_round(
    dialect: str,
    round_num: int = 1,
    out_dir: str | None = None,
    *,
    container_name: str | None = None,
    join_where_target_only: bool = False,
    preprocess_join_types: bool = False,
) -> None:
    d = normalize_dialect(dialect)
    if out_dir is None:
        out_dir = os.path.join(ROOT, "env-sql", f"round-{round_num:03d}")
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)

    print(f"[round] start dialect={d} round={round_num} out_dir={out_dir}")
    ddl_sql = build_ddl(dialect=d, round_num=round_num)
    dml_sql, table_ids = build_dml_with_meta(d)

    ddl_path = os.path.join(out_dir, "ddl.sql")
    dml_path = os.path.join(out_dir, "dml.sql")
    round_database = resolve_round_database(d, round_num=round_num, out_dir=out_dir)
    print("[round] write ddl/dml")
    write_file(ddl_path, ddl_sql)
    write_file(dml_path, dml_sql)

    if connection_driver(d) == "duckdb":
        print(f"[round] reset duckdb file: {round_database}")
        reset_duckdb_database(round_database)
    elif connection_driver(d) == "monetdb":
        monetdb_container = container_name or "monetdbtest"
        print(
            f"[round] ensure monetdb database: {round_database} "
            f"container={monetdb_container}"
        )
        ensure_monetdb_database(
            round_database,
            container_name=monetdb_container,
            destroy_existing=True,
        )

    print("[round] execute ddl/dml")
    execute_sql_files(d, [ddl_path, dml_path], database=round_database)

    _ensure_root_on_path()
    from mutate_DML.original_dml_generator import generate_original_dml
    from mutate_DML.approx_mutation import ApproximateMutator
    from mutate_DML.sql_stream import split_sql_stream

    print("[round] generate original DML")
    schema = get_schema()
    ori_error_path = os.path.join(out_dir, "ori-DML.errors.log")
    raw_ori_sql = generate_original_dml(
        schema,
        table_ids,
        d,
        count=1000,
        error_log_path=ori_error_path,
        join_where_target_only=join_where_target_only,
    )
    mutator = ApproximateMutator(dialect=d, preprocess_join_types=preprocess_join_types)
    ori_sql = "\n".join(
        mutator.preprocess_statement(stmt).strip().rstrip(";") + ";"
        for stmt in split_sql_stream([raw_ori_sql])
    ) + "\n"
    ori_path = os.path.join(out_dir, "ori-DML.sql")
    write_file(ori_path, ori_sql)
    print(f"[round] done ori-DML: {ori_path}")
    print(f"[round] ori-DML error log: {ori_error_path}")

    from mutate_DML.approx_mutation import mutate_file

    mut_path = os.path.join(out_dir, "mut-DML.sql")
    print("[round] mutate original DML")
    mutated_count = mutate_file(
        ori_path,
        mut_path,
        dialect=d,
        preprocess_join_types=preprocess_join_types,
    )
    print(f"[round] done mut-DML: {mut_path} (count={mutated_count})")

    # Placeholder: compare original vs mutated results.
