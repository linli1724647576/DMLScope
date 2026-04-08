class DialectConfig:
    def __init__(
        self,
        name: str,
        host: str = "",
        port: int | None = None,
        user: str = "",
        password: str = "",
        database: str = "",
        options: dict | None = None,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.options = options or {}


_DEFAULT_OPTIONS = {
    "driver": "mysql",
    "sqlglot_dialect": "mysql",
    "uses_database_statements": True,
    "fixed_database": False,
    "supports_geometry": True,
    "supports_json": True,
    "supports_enum_type": True,
    "supports_set_type": True,
    "supports_year_type": True,
    "supports_regexp": True,
    "supports_adddate": True,
    "supports_subdate": True,
    "supports_select_distinct": True,
    "supports_select_having": True,
    "supports_insert_ignore": True,
    "supports_insert_by_name": False,
    "supports_dml_returning": False,
    "insert_select_with_style": "before_select",
    "supports_update_low_priority": True,
    "supports_update_ignore": True,
    "supports_update_order_by_limit": True,
    "supports_update_join": True,
    "supports_update_from": False,
    "supports_delete_low_priority": True,
    "supports_delete_quick": True,
    "supports_delete_ignore": True,
    "supports_delete_order_by_limit": True,
    "supports_delete_multi_table": True,
    "supports_replace_low_priority": True,
    "supports_replace_delayed": True,
    "supports_replace_statement": True,
    "supports_replace_with": False,
    "supports_limit_offset": True,
    "supports_straight_join": True,
    "supports_right_join": True,
    "supports_intersect": True,
    "supports_except": True,
    "supports_dml_with": True,
    "supports_window_functions": False,
    "supports_foreign_keys": True,
    "supports_table_partitioning": False,
    "supports_dml_partition_clause": False,
    "supports_on_duplicate_key_update": True,
    "supports_subquery_in_join_on": True,
    "database_mode_auto": False,
    "database_default_single_on": False,
}


DIALECTS = {
    "mysql": DialectConfig(
        "mysql",
        host="127.0.0.1",
        port=13306,
        user="root",
        password="123456",
        database="",
        options={
            "supports_replace_delayed": False,
            "supports_window_functions": True,
            "supports_table_partitioning": True,
            "supports_dml_partition_clause": True,
        },
    ),
    "mariadb": DialectConfig(
        "mariadb",
        host="127.0.0.1",
        port=9901,
        user="root",
        password="123456",
        database="",
        options={
            "sqlglot_dialect": "mysql",
            "supports_dml_with": False,
            "supports_replace_delayed": False,
            "supports_window_functions": True,
            "supports_table_partitioning": True,
            "supports_dml_partition_clause": True,
        },
    ),
    "oceanbase": DialectConfig(
        "oceanbase",
        host="127.0.0.1",
        port=2881,
        user="root",
        password="",
        database="",
        options={
            "sqlglot_dialect": "mysql",
            "supports_update_low_priority": False,
            "supports_update_order_by_limit": False,
            "supports_delete_low_priority": False,
            "supports_delete_quick": False,
            "supports_delete_ignore": False,
            "supports_delete_order_by_limit": False,
            "supports_replace_low_priority": False,
            "supports_replace_delayed": False,
            "supports_limit_offset": False,
        },
    ),
    "duckdb": DialectConfig(
        "duckdb",
        host="",
        port=None,
        user="",
        password="",
        database="",
        options={
            "driver": "duckdb",
            "sqlglot_dialect": "duckdb",
            "uses_database_statements": False,
            "supports_geometry": False,
            "supports_enum_type": False,
            "supports_set_type": False,
            "supports_year_type": False,
            "supports_regexp": False,
            "supports_adddate": False,
            "supports_subdate": False,
            "supports_insert_ignore": True,
            "supports_insert_by_name": True,
            "supports_dml_returning": True,
            "supports_update_low_priority": False,
            "supports_update_ignore": False,
            "supports_update_order_by_limit": False,
            "supports_update_join": False,
            "supports_update_from": True,
            "supports_delete_low_priority": False,
            "supports_delete_quick": False,
            "supports_delete_ignore": False,
            "supports_delete_order_by_limit": False,
            "supports_delete_multi_table": False,
            "supports_replace_low_priority": False,
            "supports_replace_delayed": False,
            "supports_replace_statement": False,
            "supports_straight_join": False,
            "supports_window_functions": True,
            "supports_foreign_keys": False,
            "supports_dml_partition_clause": False,
            "supports_on_duplicate_key_update": False,
        },
    ),
    "monetdb": DialectConfig(
        "monetdb",
        host="127.0.0.1",
        port=50000,
        user="monetdb",
        password="monetdb",
        database="mtdb",
        options={
            "driver": "monetdb",
            "sqlglot_dialect": "postgres",
            "uses_database_statements": False,
            "fixed_database": False,
            "supports_geometry": False,
            "supports_json": False,
            "supports_enum_type": False,
            "supports_set_type": False,
            "supports_year_type": False,
            "supports_regexp": False,
            "supports_adddate": False,
            "supports_subdate": False,
            "supports_insert_ignore": False,
            "supports_update_low_priority": False,
            "supports_update_ignore": False,
            "supports_update_order_by_limit": False,
            "supports_update_join": False,
            "supports_update_from": True,
            "supports_delete_low_priority": False,
            "supports_delete_quick": False,
            "supports_delete_ignore": False,
            "supports_delete_order_by_limit": False,
            "supports_delete_multi_table": False,
            "supports_replace_low_priority": False,
            "supports_replace_delayed": False,
            "supports_replace_statement": False,
            "supports_limit_offset": True,
            "supports_straight_join": False,
            "supports_window_functions": True,
            "supports_on_duplicate_key_update": False,
            "supports_foreign_keys": False,
            "supports_table_partitioning": False,
            "supports_dml_partition_clause": False,
        },
    ),
    "tidb": DialectConfig(
        "tidb",
        host="127.0.0.1",
        port=4000,
        user="root",
        password="123456",
        database="",
        options={
            "sqlglot_dialect": "mysql",
            "supports_geometry": False,
            "supports_update_low_priority": False,
            "supports_update_order_by_limit": False,
            "supports_delete_low_priority": False,
            "supports_delete_quick": False,
            "supports_delete_order_by_limit": False,
            "supports_limit_offset": False,
            "supports_straight_join": False,
            "supports_window_functions": True,
            "supports_subquery_in_join_on": False,
        },
    ),
    "polardb": DialectConfig(
        "polardb",
        host="127.0.0.1",
        port=8527,
        user="polardbx_root",
        password="123456",
        database="",
        options={
            "sqlglot_dialect": "mysql",
            "supports_update_ignore": False,
            "supports_update_order_by_limit": False,
            "supports_delete_order_by_limit": False,
            "supports_limit_offset": False,
            "supports_foreign_keys": False,
            "database_mode_auto": True,
            "database_default_single_on": True,
        },
    ),
    "percona": DialectConfig(
        "percona",
        host="127.0.0.1",
        port=23306,
        user="root",
        password="123456",
        database="",
        options={
            "sqlglot_dialect": "mysql",
            "supports_window_functions": True,
            "supports_table_partitioning": True,
            "supports_dml_partition_clause": True,
        },
    ),
}

ALLOWED_DIALECTS = set(DIALECTS.keys())


def normalize_dialect(dialect: str) -> str:
    d = dialect.strip().lower()
    if d not in ALLOWED_DIALECTS:
        raise ValueError(f"Unsupported dialect: {dialect}")
    return d


def get_dialect_config(dialect: str) -> DialectConfig:
    d = normalize_dialect(dialect)
    return DIALECTS[d]


def dialect_option(dialect: str, name: str, default=None):
    d = normalize_dialect(dialect)
    if name in DIALECTS[d].options:
        return DIALECTS[d].options[name]
    return _DEFAULT_OPTIONS.get(name, default)


def get_sqlglot_dialect(dialect: str) -> str:
    return str(dialect_option(dialect, "sqlglot_dialect", "mysql"))


def get_supported_join_types(dialect: str) -> list[str]:
    types = ["JOIN", "INNER JOIN", "LEFT JOIN", "CROSS JOIN"]
    if dialect_option(dialect, "supports_right_join", True):
        types.append("RIGHT JOIN")
    if dialect_option(dialect, "supports_straight_join", True):
        types.append("STRAIGHT_JOIN")
    return types


def get_supported_set_ops(dialect: str) -> list[str]:
    ops = ["UNION", "UNION ALL"]
    if dialect_option(dialect, "supports_intersect", True):
        ops.append("INTERSECT")
    if dialect_option(dialect, "supports_except", True):
        ops.append("EXCEPT")
    return ops



def supports_update_tail(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_update_order_by_limit", True))


def supports_delete_tail(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_delete_order_by_limit", True))


def supports_limit_offset(dialect: str) -> bool:
    return bool(dialect_option(dialect, "supports_limit_offset", True))

def get_database_name(round_num: int) -> str:
    return f"databse{round_num}"
