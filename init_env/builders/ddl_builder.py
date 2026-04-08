from init_env.core.dialect import dialect_option, get_database_name, normalize_dialect
from init_env.core.schema import get_schema


def _database_create_stmt(dialect: str, db_name: str) -> str:
    stmt = f"CREATE DATABASE {db_name}"
    if dialect_option(dialect, "database_mode_auto", False):
        stmt += " MODE='auto'"
    if dialect_option(dialect, "database_default_single_on", False):
        stmt += " DEFAULT_SINGLE='on'"
    return stmt + ";"


def _table_close_stmt(dialect: str) -> str:
    return ");"


def _foreign_key_clause(column: str, ref_table: str, ref_column: str) -> str:
    return f"  FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column})"


def _uses_database_statements(dialect: str) -> bool:
    return bool(dialect_option(dialect, "uses_database_statements", True))


def _is_monetdb(dialect: str) -> bool:
    return normalize_dialect(dialect) == "monetdb"


def _string_type(dialect: str, mysql_type: str) -> str:
    d = normalize_dialect(dialect)
    if d == "duckdb":
        return "VARCHAR"
    if d == "monetdb":
        if mysql_type.upper() in {"TINYTEXT", "MEDIUMTEXT", "LONGTEXT"}:
            return "CLOB"
    return mysql_type


def _binary_type(dialect: str, mysql_type: str) -> str:
    d = normalize_dialect(dialect)
    if d == "duckdb":
        return "BLOB"
    if d == "monetdb":
        return "BLOB"
    return mysql_type


def _datetime_type(dialect: str) -> str:
    if normalize_dialect(dialect) in {"duckdb", "monetdb"}:
        return "TIMESTAMP"
    return "DATETIME"


def _t2_c10_type(dialect: str) -> str:
    if dialect_option(dialect, "supports_enum_type", True):
        return "ENUM('value1','value2','value3')"
    return "VARCHAR(255)"


def _set_type(dialect: str, values: str) -> str:
    if dialect_option(dialect, "supports_set_type", True):
        return f"SET({values})"
    return "VARCHAR(255)"


def _year_type(dialect: str) -> str:
    if dialect_option(dialect, "supports_year_type", True):
        return "YEAR"
    return "SMALLINT"


def _float_type(dialect: str) -> str:
    if normalize_dialect(dialect) == "duckdb":
        return "DOUBLE"
    return "FLOAT(8,2)"


def _double_type(dialect: str) -> str:
    if normalize_dialect(dialect) in {"duckdb", "monetdb"}:
        return "DOUBLE"
    return "DOUBLE(12,4)"


def _json_type(dialect: str) -> str:
    if _is_monetdb(dialect):
        return "JSON"
    if dialect_option(dialect, "supports_json", True):
        return "JSON"
    return "VARCHAR(255)"


def build_ddl(dialect: str, round_num: int = 1) -> str:
    d = normalize_dialect(dialect)
    schema = get_schema()
    c11_type = "GEOMETRY" if dialect_option(d, "supports_geometry", True) else "VARCHAR(255)"
    supports_foreign_keys = bool(dialect_option(d, "supports_foreign_keys", True))
    db_name = get_database_name(round_num)
    t2_set_values = "'a','b','c','d'"
    t3_set_values = "'x','y','z'"
    statements = []

    if _uses_database_statements(d):
        statements.extend(
            [
                f"DROP DATABASE IF EXISTS {db_name};",
                _database_create_stmt(d, db_name),
                f"USE {db_name};",
                "",
            ]
        )

    statements.extend(
        [
            "DROP TABLE IF EXISTS t3;",
            "DROP TABLE IF EXISTS t2;",
            "DROP TABLE IF EXISTS t1;",
            "",
            "CREATE TABLE t1 (",
            "  c1 INT NOT NULL,",
            "  c2 VARCHAR(255) NULL,",
            "  c3 VARCHAR(255) NULL,",
            "  c4 INT NULL,",
            "  c5 DATE NULL,",
            "  c6 VARCHAR(10) NULL,",
            "  PRIMARY KEY (c1)",
            _table_close_stmt(d),
            "",
            "CREATE TABLE t2 (",
            "  c1 INT NOT NULL,",
            "  c2 INT NULL,",
            "  c3 DECIMAL(10,2) NULL,",
            "  c4 VARCHAR(50) NULL,",
            "  c5 DATE NULL,",
            f"  c6 {_string_type(d, 'MEDIUMTEXT')} NULL,",
            f"  c7 {_string_type(d, 'LONGTEXT')} NULL,",
            f"  c8 {_binary_type(d, 'MEDIUMBLOB')} NULL,",
            f"  c9 {_binary_type(d, 'LONGBLOB')} NULL,",
            f"  c10 {_t2_c10_type(d)} NULL,",
            f"  c11 {_set_type(d, t2_set_values)} NULL,",
            f"  c12 {_binary_type(d, 'BIT(8)')} NULL,",
            f"  c13 {_datetime_type(d)} NULL,",
            f"  c14 {_float_type(d)} NULL,",
            f"  c15 {_double_type(d)} NULL,",
            f"  c16 {_json_type(d)} NULL,",
            "  PRIMARY KEY (c1)" + ("," if supports_foreign_keys else ""),
            *([_foreign_key_clause("c2", "t1", "c1")] if supports_foreign_keys else []),
            _table_close_stmt(d),
            "",
            "CREATE TABLE t3 (",
            "  c1 INT NOT NULL,",
            "  c2 INT NULL,",
            "  c3 INT NULL,",
            f"  c4 {_year_type(d)} NULL,",
            f"  c5 {_datetime_type(d)} NULL,",
            "  c6 TINYINT NULL,",
            "  c7 SMALLINT NULL,",
            "  c8 INTEGER NULL,",
            "  c9 BIGINT NULL,",
            f"  c10 {_string_type(d, 'LONGTEXT')} NULL,",
            f"  c11 {c11_type} NULL,",
            f"  c12 {_string_type(d, 'TINYTEXT')} NULL,",
            f"  c13 {_binary_type(d, 'TINYBLOB')} NULL,",
            f"  c14 {_set_type(d, t3_set_values)} NULL,",
            "  c15 BOOLEAN NULL,",
            "  PRIMARY KEY (c1)" + ("," if supports_foreign_keys else ""),
            *(
                [
                    _foreign_key_clause("c2", "t1", "c1") + ",",
                    _foreign_key_clause("c3", "t2", "c1"),
                ]
                if supports_foreign_keys
                else []
            ),
            _table_close_stmt(d),
            "",
        ]
    )
    return "\n".join(statements)
