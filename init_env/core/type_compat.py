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
    if normalized in DATE_TYPES:
        return "date"
    if normalized in DATETIME_TYPES:
        return "datetime"
    if normalized in STRING_TYPES:
        return "string"
    if normalized in STRICT_NUMERIC_TYPES:
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
