import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

INIT_ENV = os.path.join(ROOT, "init_env")
if INIT_ENV not in sys.path:
    sys.path.insert(0, INIT_ENV)

from init_env.core.dialect import ALLOWED_DIALECTS, get_database_name, get_dialect_config
from init_env.pipeline.generate_round1 import generate_round
from init_env.utils.db_connect import connect_db, connection_driver, resolve_round_database
from init_env.utils.monetdb_admin import ensure_monetdb_database
from init_env.utils.sql_executor import execute_sql_files
from mutate_DML.result_compare import compare_files_and_log


def _root_path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def _resolve_out_dir(out_dir: str, round_num: int) -> str:
    if not out_dir:
        return _root_path("env-sql", f"round-{round_num:03d}")

    base = out_dir if os.path.isabs(out_dir) else _root_path(out_dir)
    tail = os.path.basename(os.path.normpath(base))
    if tail.startswith("round-"):
        return base
    return os.path.join(base, f"round-{round_num:03d}")


def _resolve_out_root(out_dir: str, start_round: int) -> str:
    resolved = _resolve_out_dir(out_dir, start_round)
    tail = os.path.basename(os.path.normpath(resolved))
    if tail.startswith("round-"):
        return os.path.dirname(resolved)
    return resolved


def _clear_bug_dir(bug_root: str = "bug") -> None:
    if not os.path.isdir(bug_root):
        return
    for name in os.listdir(bug_root):
        path = os.path.join(bug_root, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(path)
        except OSError:
            pass


def _parse_crashed_count(detail: str) -> int:
    match = re.search(r"\bcrashed=(\d+)\b", detail or "")
    return int(match.group(1)) if match else 0


def _round_label(round_num: int) -> str:
    return f"round-{round_num:03d}"


def _default_container_name(dialect: str) -> str:
    return f"{dialect}test"


def _default_state_path(dialect: str) -> str:
    state_dir = _root_path("runtime")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{dialect}.run_state.json")


def _decode_subprocess_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _wsl_root_exec_cmd(shell_script: str) -> list[str]:
    return [
        "wsl",
        "-d",
        "Ubuntu-22.04",
        "-u",
        "root",
        "--cd",
        "/",
        "--exec",
        "/bin/sh",
        "-lc",
        shell_script,
    ]


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_command_capture(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        timeout=timeout,
        check=False,
    )
    result.stdout = _decode_subprocess_output(result.stdout)
    result.stderr = _decode_subprocess_output(result.stderr)
    return result


def _load_run_state(path: str) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _save_run_state(path: str, **payload) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = dict(payload)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _recovery_log_paths(bug_root: str, round_num: int) -> tuple[str, str]:
    round_dir = os.path.join(bug_root, _round_label(round_num))
    os.makedirs(round_dir, exist_ok=True)
    return (
        os.path.join(round_dir, "recovery.txt"),
        os.path.join(bug_root, "total_recovery.log"),
    )


def _append_recovery_log(
    bug_root: str,
    *,
    round_num: int,
    dialect: str,
    container_name: str,
    action: str,
    container_state: dict | None = None,
    db_state: dict | None = None,
    detail: str = "",
) -> None:
    payload = [
        "-- RECOVERY\n",
        f"-- time: {datetime.now(timezone.utc).isoformat()}\n",
        f"-- round: {_round_label(round_num)}\n",
        f"-- dialect: {dialect}\n",
        f"-- container: {container_name}\n",
        f"-- action: {action}\n",
    ]
    if detail:
        payload.append(f"-- detail: {detail}\n")
    if container_state is not None:
        payload.append(
            f"-- container_state: {json.dumps(container_state, ensure_ascii=True, sort_keys=True)}\n"
        )
    if db_state is not None:
        payload.append(
            f"-- db_state: {json.dumps(db_state, ensure_ascii=True, sort_keys=True)}\n"
        )
    payload.append("\n")

    round_path, total_path = _recovery_log_paths(bug_root, round_num)
    for path in (round_path, total_path):
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(payload)


def _docker_container_state(container_name: str) -> dict:
    cmd = _wsl_root_exec_cmd(
        f"docker inspect {_shell_single_quote(container_name)} --format '{{{{json .State}}}}'"
    )
    try:
        result = _run_command_capture(cmd, timeout=20)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }

    raw = (result.stdout or "").strip()
    try:
        state = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"ok": False, "error": f"invalid inspect output: {raw}"}

    health = state.get("Health") or {}
    return {
        "ok": True,
        "status": state.get("Status"),
        "running": state.get("Running"),
        "restarting": state.get("Restarting"),
        "paused": state.get("Paused"),
        "dead": state.get("Dead"),
        "oom_killed": state.get("OOMKilled"),
        "exit_code": state.get("ExitCode"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "health": health.get("Status"),
    }


def _mysql_runtime_state(dialect: str, database: str) -> dict:
    cfg = get_dialect_config(dialect)
    state = {
        "ready": False,
        "database_exists": False,
        "table_count": None,
        "version": None,
        "hostname": None,
        "error": "",
    }
    try:
        import pymysql
    except ImportError as exc:
        state["error"] = f"pymysql import failed: {exc}"
        return state

    try:
        conn = pymysql.connect(
            host=cfg.host,
            port=cfg.port or 3306,
            user=cfg.user,
            password=cfg.password,
            database=None,
            autocommit=True,
            charset="utf8mb4",
            connect_timeout=3,
            read_timeout=3,
            write_timeout=3,
        )
    except Exception as exc:
        state["error"] = str(exc)
        return state

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION(), @@hostname")
            row = cur.fetchone() or (None, None)
            state["ready"] = True
            state["version"] = row[0]
            state["hostname"] = row[1]
            cur.execute("SHOW DATABASES LIKE %s", (database,))
            state["database_exists"] = cur.fetchone() is not None
            if state["database_exists"]:
                cur.execute(f"SHOW TABLES FROM `{database}`")
                state["table_count"] = len(cur.fetchall())
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        conn.close()
    return state


def _generic_runtime_state(dialect: str, database: str) -> dict:
    state = {
        "ready": False,
        "database_exists": None,
        "table_count": None,
        "version": None,
        "hostname": None,
        "error": "",
    }
    try:
        conn = connect_db(dialect, database=database)
    except Exception as exc:
        state["error"] = str(exc)
        return state

    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1")
            cur.fetchall()
            state["ready"] = True
            try:
                cur.execute("SELECT COUNT(*) FROM sys.tables")
                row = cur.fetchone()
                if row:
                    state["table_count"] = row[0]
                    state["database_exists"] = True
            except Exception:
                pass
        finally:
            close = getattr(cur, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return state


def _database_runtime_state(dialect: str, database: str) -> dict:
    if connection_driver(dialect) == "pymysql":
        return _mysql_runtime_state(dialect, database)
    return _generic_runtime_state(dialect, database)


def _restart_container(container_name: str, timeout_s: int) -> tuple[bool, str]:
    cmd = _wsl_root_exec_cmd(f"docker restart {_shell_single_quote(container_name)}")
    try:
        result = _run_command_capture(cmd, timeout=timeout_s)
    except Exception as exc:
        return False, str(exc)
    detail = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0, detail


def _wait_database_ready(
    dialect: str,
    database: str,
    *,
    timeout_s: int,
    interval_s: float,
) -> tuple[bool, dict]:
    deadline = time.time() + timeout_s
    last_state = _database_runtime_state(dialect, database)
    while time.time() < deadline:
        last_state = _database_runtime_state(dialect, database)
        if last_state.get("ready"):
            return True, last_state
        time.sleep(interval_s)
    return False, last_state


def _rebuild_round_state(
    dialect: str,
    out_dir: str,
    *,
    database: str | None = None,
    container_name: str | None = None,
) -> dict:
    ddl_path = os.path.join(out_dir, "ddl.sql")
    dml_path = os.path.join(out_dir, "dml.sql")
    missing = [path for path in (ddl_path, dml_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing rebuild files: {missing}")
    if connection_driver(dialect) == "monetdb":
        ensure_monetdb_database(
            database or resolve_round_database(dialect, out_dir=out_dir),
            container_name=container_name or _default_container_name(dialect),
            destroy_existing=True,
        )
    return execute_sql_files(
        dialect,
        [ddl_path, dml_path],
        database=database,
        raise_on_error=False,
    )


def _recover_mysql_after_crash(
    *,
    dialect: str,
    round_num: int,
    out_dir: str,
    bug_root: str,
    container_name: str,
    restart_timeout_s: int,
    ready_timeout_s: int,
) -> None:
    database = get_database_name(round_num)

    pre_container_state = _docker_container_state(container_name)
    pre_mysql_state = _mysql_runtime_state(dialect, database)
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="pre_restart_status",
        container_state=pre_container_state,
        db_state=pre_mysql_state,
    )

    restarted, restart_detail = _restart_container(container_name, restart_timeout_s)
    post_restart_container_state = _docker_container_state(container_name)
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="post_restart_status",
        container_state=post_restart_container_state,
        detail=restart_detail,
    )
    if not restarted:
        raise RuntimeError(
            f"MySQL container restart failed: container={container_name} detail={restart_detail}"
        )

    if connection_driver(dialect) == "monetdb":
        ensure_monetdb_database(
            database,
            container_name=container_name,
            destroy_existing=False,
            timeout_s=ready_timeout_s,
        )
        ready = True
        ready_state = _database_runtime_state(dialect, database)
    else:
        ready, ready_state = _wait_database_ready(
            dialect,
            database,
            timeout_s=ready_timeout_s,
            interval_s=2.0,
        )
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="mysql_ready_check",
        container_state=_docker_container_state(container_name),
        db_state=ready_state,
        detail=f"ready={ready}",
    )
    if not ready:
        raise RuntimeError(
            f"MySQL did not become ready after restart: container={container_name} state={ready_state}"
        )

    rebuild_summary = _rebuild_round_state(dialect, out_dir, database=database)
    rebuilt_state = _mysql_runtime_state(dialect, database)
    rebuild_detail = (
        f"rebuild_complete success={rebuild_summary['success']} "
        f"failed={rebuild_summary['failed']} affected={rebuild_summary['affected']}"
    )
    if rebuild_summary.get("first_failure") is not None:
        first_failure = rebuild_summary["first_failure"]
        rebuild_detail += (
            f" first_failure_path={first_failure['path']} "
            f"first_failure_error={first_failure['error']}"
        )
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="post_rebuild_status",
        container_state=_docker_container_state(container_name),
        db_state=rebuilt_state,
        detail=rebuild_detail,
    )


def _recover_database_after_crash(
    *,
    dialect: str,
    round_num: int,
    out_dir: str,
    bug_root: str,
    container_name: str,
    restart_timeout_s: int,
    ready_timeout_s: int,
) -> None:
    if connection_driver(dialect) == "pymysql":
        _recover_mysql_after_crash(
            dialect=dialect,
            round_num=round_num,
            out_dir=out_dir,
            bug_root=bug_root,
            container_name=container_name,
            restart_timeout_s=restart_timeout_s,
            ready_timeout_s=ready_timeout_s,
        )
        return

    database = resolve_round_database(dialect, round_num=round_num, out_dir=out_dir)
    pre_container_state = _docker_container_state(container_name)
    pre_db_state = _database_runtime_state(dialect, database)
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="pre_restart_status",
        container_state=pre_container_state,
        db_state=pre_db_state,
    )

    restarted, restart_detail = _restart_container(container_name, restart_timeout_s)
    post_restart_container_state = _docker_container_state(container_name)
    post_restart_db_state = _database_runtime_state(dialect, database)
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="post_restart_status",
        container_state=post_restart_container_state,
        db_state=post_restart_db_state,
        detail=restart_detail,
    )

    if not restarted and post_restart_db_state.get("ready"):
        _append_recovery_log(
            bug_root,
            round_num=round_num,
            dialect=dialect,
            container_name=container_name,
            action="restart_degraded_continue",
            container_state=post_restart_container_state,
            db_state=post_restart_db_state,
            detail=f"restart_failed_but_db_ready detail={restart_detail}",
        )
        ready = True
        ready_state = post_restart_db_state
    elif not restarted:
        raise RuntimeError(
            f"Container restart failed: dialect={dialect} container={container_name} detail={restart_detail}"
        )
    else:
        if connection_driver(dialect) == "monetdb":
            ensure_monetdb_database(
                database,
                container_name=container_name,
                destroy_existing=False,
                timeout_s=ready_timeout_s,
            )
            ready = True
            ready_state = _database_runtime_state(dialect, database)
        else:
            ready, ready_state = _wait_database_ready(
                dialect,
                database,
                timeout_s=ready_timeout_s,
                interval_s=2.0,
            )
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="db_ready_check",
        container_state=_docker_container_state(container_name),
        db_state=ready_state,
        detail=f"ready={ready}",
    )
    if not ready:
        raise RuntimeError(
            f"Database did not become ready after restart: dialect={dialect} container={container_name} state={ready_state}"
        )

    rebuild_summary = _rebuild_round_state(
        dialect,
        out_dir,
        database=database,
        container_name=container_name,
    )
    rebuilt_state = _database_runtime_state(dialect, database)
    rebuild_detail = (
        f"rebuild_complete success={rebuild_summary['success']} "
        f"failed={rebuild_summary['failed']} affected={rebuild_summary['affected']}"
    )
    if rebuild_summary.get("first_failure") is not None:
        first_failure = rebuild_summary["first_failure"]
        rebuild_detail += (
            f" first_failure_path={first_failure['path']} "
            f"first_failure_error={first_failure['error']}"
        )
    _append_recovery_log(
        bug_root,
        round_num=round_num,
        dialect=dialect,
        container_name=container_name,
        action="post_rebuild_status",
        container_state=_docker_container_state(container_name),
        db_state=rebuilt_state,
        detail=rebuild_detail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate round init DDL/DML SQL files.")
    parser.add_argument("--dialect", required=True, choices=sorted(ALLOWED_DIALECTS))
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--duration-hours", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state-file", default="")
    parser.add_argument(
        "--preprocess-join-types",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mysql-container", default="")
    parser.add_argument("--restart-timeout-seconds", type=int, default=120)
    parser.add_argument("--mysql-ready-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    mysql_container = args.mysql_container or _default_container_name(args.dialect)
    state_path = args.state_file or _default_state_path(args.dialect)

    start_round = args.round
    out_root = _resolve_out_root(args.out_dir, start_round)
    saved_state = _load_run_state(state_path) if args.resume else None
    if args.resume and saved_state:
        saved_round = int(saved_state.get("next_round") or start_round)
        saved_out_root = saved_state.get("out_root")
        if saved_state.get("dialect") == args.dialect and (
            not saved_out_root or os.path.normcase(saved_out_root) == os.path.normcase(out_root)
        ):
            start_round = max(start_round, saved_round)
    if not args.resume:
        _clear_bug_dir(_root_path("bug"))
        _clear_bug_dir(out_root)
    else:
        print(
            f"[main] resume enabled dialect={args.dialect} start_round={start_round} "
            f"state_file={state_path}"
        )
    _save_run_state(
        state_path,
        dialect=args.dialect,
        out_root=out_root,
        next_round=start_round,
        status="initialized",
    )

    duration_hours = args.duration_hours
    end_time = None
    if duration_hours and duration_hours > 0:
        end_time = time.time() + duration_hours * 3600

    current_round = start_round
    while True:
        if end_time is not None and time.time() >= end_time:
            print("[main] time budget reached, stop")
            _save_run_state(
                state_path,
                dialect=args.dialect,
                out_root=out_root,
                next_round=current_round,
                status="time_budget_reached",
            )
            break

        out_dir = _resolve_out_dir(args.out_dir, current_round)
        round_database = resolve_round_database(
            args.dialect,
            round_num=current_round,
            out_dir=out_dir,
        )
        _save_run_state(
            state_path,
            dialect=args.dialect,
            out_root=out_root,
            next_round=current_round,
            status="round_started",
            current_round=current_round,
            database=round_database,
            out_dir=out_dir,
        )
        print(
            f"[main] start dialect={args.dialect} round={current_round} out_dir={out_dir} "
            f"preprocess_join_types={args.preprocess_join_types}"
        )
        generate_round(
            dialect=args.dialect,
            round_num=current_round,
            out_dir=out_dir,
            container_name=mysql_container,
            preprocess_join_types=args.preprocess_join_types,
        )

        ori_path = os.path.join(out_dir, "ori-DML.sql")
        print("[main] compare original vs mutated")
        compare_result = compare_files_and_log(
            ori_path,
            dialect=args.dialect,
            database=round_database,
            round_num=current_round,
            bug_root=_root_path("bug"),
            preprocess_join_types=args.preprocess_join_types,
        )
        print(f"[main] compare done: {compare_result.detail}")

        crashed = _parse_crashed_count(compare_result.detail)
        if crashed > 0:
            print(
                f"[main] crash detected, start recovery round={current_round} "
                f"container={mysql_container}"
            )
            _recover_database_after_crash(
                dialect=args.dialect,
                round_num=current_round,
                out_dir=out_dir,
                bug_root=_root_path("bug"),
                container_name=mysql_container,
                restart_timeout_s=args.restart_timeout_seconds,
                ready_timeout_s=args.mysql_ready_timeout_seconds,
            )
            print(f"[main] recovery complete round={current_round}")
            _save_run_state(
                state_path,
                dialect=args.dialect,
                out_root=out_root,
                next_round=current_round + 1,
                status="round_recovered",
                current_round=current_round,
                database=round_database,
                out_dir=out_dir,
                detail=compare_result.detail,
            )
        else:
            _save_run_state(
                state_path,
                dialect=args.dialect,
                out_root=out_root,
                next_round=current_round + 1,
                status="round_completed",
                current_round=current_round,
                database=round_database,
                out_dir=out_dir,
                detail=compare_result.detail,
            )

        current_round += 1

        if end_time is None:
            break

    _save_run_state(
        state_path,
        dialect=args.dialect,
        out_root=out_root,
        next_round=current_round,
        status="stopped",
    )
    print("[main] done")


if __name__ == "__main__":
    main()
