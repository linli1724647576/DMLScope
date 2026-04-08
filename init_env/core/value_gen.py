import json
import random

from init_env.core.dialect import dialect_option, normalize_dialect
from init_env.core.rand_utils import geom_literal, rand_date, rand_datetime, rand_hex, rand_hex_body, rand_str, sql_str

NUMERIC_TYPES = {"int", "dec", "float", "double", "bigint", "year", "bool"}
STRING_TYPES = {"str"}
DATE_TYPES = {"date"}
DATETIME_TYPES = {"datetime"}
JSON_TYPES = {"json"}
BINARY_TYPES = {"bin"}
GEOM_TYPES = {"geom"}


def gen_value(col_type: str, dialect: str, col_meta: dict | None = None) -> str:
    if col_type in STRING_TYPES:
        meta = col_meta or {}
        enum_values = meta.get("enum_values")
        if enum_values:
            return sql_str(random.choice(enum_values))
        set_values = meta.get("set_values")
        if set_values:
            count = random.randint(1, len(set_values))
            chosen = random.sample(set_values, count)
            return sql_str(",".join(chosen))
        max_len = meta.get("max_len")
        if max_len:
            upper = max(1, int(max_len))
            upper = min(upper, 64)
            return sql_str(rand_str(1, upper))
        return sql_str(rand_str(5, 12))
    if col_type in NUMERIC_TYPES:
        meta = col_meta or {}
        if "min" in meta and "max" in meta:
            return str(random.randint(int(meta["min"]), int(meta["max"])))
        if col_type == "bool":
            return str(random.randint(0, 1))
        if col_type == "year":
            return str(random.randint(1970, 2030))
        return str(random.randint(1, 1000))
    if col_type in DATE_TYPES:
        return sql_str(rand_date())
    if col_type in DATETIME_TYPES:
        return sql_str(rand_datetime())
    if col_type in JSON_TYPES:
        return sql_str(json.dumps({"k": rand_str(3, 6), "n": random.randint(1, 100)}))
    if col_type in BINARY_TYPES:
        meta = col_meta or {}
        max_bytes = meta.get("max_bytes")
        nbytes = int(max_bytes) if max_bytes else 6
        if normalize_dialect(dialect) == "duckdb":
            return f"FROM_HEX('{rand_hex_body(nbytes)}')"
        if max_bytes:
            return rand_hex(nbytes)
        return rand_hex(nbytes)
    if col_type in GEOM_TYPES:
        d = normalize_dialect(dialect)
        if not dialect_option(d, "supports_geometry", True):
            return sql_str(rand_str(5, 10))
        return geom_literal()
    return sql_str(rand_str(5, 12))
