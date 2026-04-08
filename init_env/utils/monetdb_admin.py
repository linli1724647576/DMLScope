import subprocess
import time

from init_env.core.dialect import get_dialect_config


def monetdb_round_database_name(round_num: int, *, base_database: str = "mtdb") -> str:
    base = (base_database or "mtdb").strip() or "mtdb"
    return f"{base}_r{round_num:03d}"


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "gbk"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    return text.replace("\x00", "")


def _clean_wsl_noise(text: str) -> str:
    if not text:
        return ""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("wsl:"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


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


def _run_shell(shell_script: str, *, timeout_s: int) -> subprocess.CompletedProcess:
    result = subprocess.run(
        _wsl_root_exec_cmd(shell_script),
        capture_output=True,
        text=False,
        timeout=timeout_s,
        check=False,
    )
    result.stdout = _clean_wsl_noise(_decode_output(result.stdout))
    result.stderr = _clean_wsl_noise(_decode_output(result.stderr))
    return result


def _ensure_monetdbd_started(container_name: str, *, timeout_s: int) -> None:
    shell_script = (
        f"docker exec {_shell_single_quote(container_name)} "
        "sh -lc 'mkdir -p /var/monetdb5/dbfarm; monetdbd start /var/monetdb5/dbfarm >/dev/null 2>&1 || true'"
    )
    _run_shell(shell_script, timeout_s=timeout_s)


def _needs_admin_retry(detail: str) -> bool:
    lower_detail = (detail or "").lower()
    return any(
        marker in lower_detail
        for marker in (
            "cannot connect",
            "connection refused",
            "challenge error",
            "please start monetdbd",
        )
    )


def _run_monetdb_admin(
    container_name: str,
    args: list[str],
    *,
    timeout_s: int = 30,
) -> tuple[bool, str]:
    quoted_args = " ".join(_shell_single_quote(arg) for arg in args)
    shell_script = (
        f"docker exec {_shell_single_quote(container_name)} "
        f"monetdb {quoted_args}"
    )

    _ensure_monetdbd_started(container_name, timeout_s=timeout_s)
    result = _run_shell(shell_script, timeout_s=timeout_s)
    detail = result.stdout or result.stderr
    if result.returncode == 0:
        return True, detail

    if _needs_admin_retry(detail):
        time.sleep(1.0)
        _ensure_monetdbd_started(container_name, timeout_s=timeout_s)
        retry = _run_shell(shell_script, timeout_s=timeout_s)
        retry_detail = retry.stdout or retry.stderr
        return retry.returncode == 0, retry_detail

    return False, detail


def _wait_until_connectable(database: str, *, timeout_s: int = 30) -> None:
    from init_env.utils.db_connect import connect_db

    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            conn = connect_db("monetdb", database=database)
            cur = conn.cursor()
            try:
                cur.execute("SELECT 1")
                cur.fetchall()
            finally:
                close = getattr(cur, "close", None)
                if callable(close):
                    close()
                close = getattr(conn, "close", None)
                if callable(close):
                    close()
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(
        f"MonetDB database did not become connectable: database={database} detail={last_error}"
    )


def ensure_monetdb_database(
    database: str,
    *,
    container_name: str = "monetdbtest",
    destroy_existing: bool = False,
    timeout_s: int = 45,
) -> None:
    cfg = get_dialect_config("monetdb")
    password = cfg.password or "monetdb"

    if destroy_existing:
        ok, detail = _run_monetdb_admin(container_name, ["destroy", "-f", database], timeout_s=timeout_s)
        if not ok and "no such database" not in detail.lower():
            raise RuntimeError(
                f"MonetDB destroy failed: database={database} container={container_name} detail={detail}"
            )

    ok, detail = _run_monetdb_admin(
        container_name,
        ["create", "-p", password, database],
        timeout_s=timeout_s,
    )
    lower_detail = detail.lower()
    if not ok and "already exists" not in lower_detail:
        raise RuntimeError(
            f"MonetDB create failed: database={database} container={container_name} detail={detail}"
        )

    ok, detail = _run_monetdb_admin(container_name, ["release", database], timeout_s=timeout_s)
    lower_detail = detail.lower()
    if not ok and "not under maintenance" not in lower_detail:
        raise RuntimeError(
            f"MonetDB release failed: database={database} container={container_name} detail={detail}"
        )

    _wait_until_connectable(database, timeout_s=timeout_s)
