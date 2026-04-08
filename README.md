# DMLScope

DMLScope is a DML fuzzing framework for generating original DML statements, mutating them, executing both versions on a target DBMS, and comparing the results to detect logic bugs, crashes, and behavioral inconsistencies.

## Environment Setup

### Python

Use Python 3.11 or newer. The project is pure Python, but several runtime drivers are required depending on the target database.

Recommended setup on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install sqlglot duckdb pymysql pymonetdb
```

Minimum package guidance:

- `sqlglot`: SQL parsing and rewriting
- `duckdb`: required for the `duckdb` dialect
- `pymysql`: required for MySQL-compatible dialects
- `pymonetdb`: required for the `monetdb` dialect

If you only plan to run DuckDB locally, this is sufficient:

```powershell
pip install sqlglot duckdb
```

### Database Backends

#### DuckDB

DuckDB is the easiest backend to run because it uses a local database file and does not require a server.

No extra infrastructure is needed beyond the Python package:

```powershell
pip install duckdb
```

#### MySQL-Compatible Dialects

For `mysql`, `mariadb`, `oceanbase`, `percona`, `polardb`, and `tidb`, DMLScope connects through `pymysql` using the host/port/user/password values defined in [`init_env/core/dialect.py`](./init_env/core/dialect.py).

You must ensure that:

- the target server is already running
- the configured port is reachable from the machine running DMLScope
- the configured username/password is valid

#### MonetDB

MonetDB support expects:

- `pymonetdb` installed in Python
- a MonetDB container accessible through Docker
- WSL available on Windows, because the recovery logic invokes Docker through `wsl`

The recovery and admin flow also assumes a container name such as `monetdbtest`.

## How To Run

### Basic Command

Show all CLI options:

```powershell
python main.py --help
```

Run one DuckDB round:

```powershell
python main.py --dialect duckdb --round 1
```

Run one DuckDB round into a custom output directory:

```powershell
python main.py --dialect duckdb --round 1 --out-dir env-sql
```

Run for multiple hours with resume support:

```powershell
python main.py --dialect duckdb --round 1 --duration-hours 4 --resume
```

Run MonetDB with an explicit container name:

```powershell
python main.py --dialect monetdb --round 1 --mysql-container monetdbtest
```

The `--mysql-container` flag is also used internally for restart and recovery of server-backed dialects.

### Useful Options

- `--dialect`: target database dialect, required
- `--round`: starting round number
- `--out-dir`: output root or explicit round directory
- `--duration-hours`: keep running rounds until the time budget is reached
- `--resume`: continue from the saved run state
- `--state-file`: custom path for the run state JSON file
- `--preprocess-join-types` / `--no-preprocess-join-types`: toggle SQL preprocessing before mutation and comparison
- `--restart-timeout-seconds`: restart timeout for crash recovery
- `--mysql-ready-timeout-seconds`: readiness timeout after restart

## Output Layout

Typical output paths:

- `env-sql/round-001/ddl.sql`
- `env-sql/round-001/dml.sql`
- `env-sql/round-001/ori-DML.sql`
- `env-sql/round-001/mut-DML.sql`
- `bug/round-001/`
- `runtime/<dialect>.run_state.json`

Bug and crash summaries are written under `bug/`, including:

- `bug/total_bug.log`
- `bug/total_crash.log`
- `bug/total_recovery.log`
