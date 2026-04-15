from init_env.core.value_gen import (
    BINARY_TYPES,
    DATETIME_TYPES,
    DATE_TYPES,
    GEOM_TYPES,
    JSON_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
)


BOOLEAN_TYPES = {"bool", "boolean"}
STRICT_NUMERIC_TYPES = set(NUMERIC_TYPES) - {"bool"}
SAFE_GENERAL_TYPE_TAGS = ("string", "numeric", "date", "datetime")

NUMERIC_TYPE_ALIASES = {
    "integer",
    "smallint",
    "tinyint",
    "mediumint",
    "int2",
    "int4",
    "int8",
    "bigserial",
    "serial",
    "serial2",
    "serial4",
    "serial8",
    "decimal",
    "numeric",
    "number",
    "real",
    "float4",
    "float8",
    "money",
}

STRING_TYPE_ALIASES = {
    "char",
    "varchar",
    "text",
    "tinytext",
    "mediumtext",
    "longtext",
    "nchar",
    "nvarchar",
    "bpchar",
    "citext",
    "clob",
    "uuid",
    "enum",
    "set",
}

DATE_TYPE_ALIASES = {"date32"}

DATETIME_TYPE_ALIASES = {
    "timestamp",
    "timestamptz",
    "timestampz",
    "datetime2",
    "smalldatetime",
}


def _normalize_col_type(col_type: str | None) -> str:
    return (col_type or "").lower().split("(", 1)[0].strip()


def is_boolean_type(col_type: str | None) -> bool:
    return _normalize_col_type(col_type) in BOOLEAN_TYPES


def is_binary_type(col_type: str | None) -> bool:
    normalized = _normalize_col_type(col_type)
    return (
        normalized in BINARY_TYPES
        or "blob" in normalized
        or "binary" in normalized
        or normalized == "bit"
    )


def column_type_tag(col_type: str | None) -> str:
    normalized = _normalize_col_type(col_type)
    if normalized in GEOM_TYPES:
        return "geom"
    if normalized in JSON_TYPES:
        return "json"
    if is_binary_type(normalized):
        return "binary"
    if is_boolean_type(normalized):
        return "boolean"
    if normalized in DATE_TYPES or normalized in DATE_TYPE_ALIASES:
        return "date"
    if normalized in DATETIME_TYPES or normalized in DATETIME_TYPE_ALIASES:
        return "datetime"
    if normalized in STRING_TYPES or normalized in STRING_TYPE_ALIASES:
        return "string"
    if normalized in STRICT_NUMERIC_TYPES or normalized in NUMERIC_TYPE_ALIASES:
        return "numeric"
    if (
        normalized.endswith("int")
        or normalized.startswith("decimal")
        or normalized.startswith("numeric")
        or normalized.startswith("float")
        or normalized.startswith("double")
    ):
        return "numeric"
    return "string"


def columns_by_type_tag(
    schema: dict,
    table: str,
    type_tag: str,
    *,
    include_pk: bool = False,
) -> list[str]:
    pk = schema[table]["pk"]
    return [
        col
        for col, col_type in schema[table]["columns"].items()
        if (include_pk or col != pk) and column_type_tag(col_type) == type_tag
    ]


def available_general_type_tags(schema: dict, table: str) -> list[str]:
    tags = [tag for tag in SAFE_GENERAL_TYPE_TAGS if columns_by_type_tag(schema, table, tag)]
    return tags or ["numeric"]
