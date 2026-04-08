import random

from init_env.core.rand_utils import maybe_null
from init_env.core.schema_utils import column_meta, column_nullable, fk_columns
from init_env.core.value_gen import gen_value


def build_row(schema: dict, table: str, pk_value: int, table_ids: dict, dialect: str) -> dict:
    row = {}
    pk_col = schema[table]["pk"]
    fk_map = fk_columns(schema, table)

    for col, col_type in schema[table]["columns"].items():
        if col == pk_col:
            row[col] = str(pk_value)
            continue

        if col in fk_map:
            ref = fk_map[col]
            ref_table = ref["ref_table"]
            row[col] = str(random.choice(table_ids[ref_table]))
            continue

        meta = column_meta(schema, table, col)
        nullable = column_nullable(schema, table, col)
        value = gen_value(col_type, dialect, meta)
        row[col] = maybe_null(value, nullable)

    return row
