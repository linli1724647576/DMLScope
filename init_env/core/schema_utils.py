from collections import deque


def get_table_order(schema: dict) -> list:
    indeg = {t: 0 for t in schema}
    deps = {t: [] for t in schema}

    for table, meta in schema.items():
        for fk in meta.get("foreign_keys", []):
            ref = fk["ref_table"]
            deps[ref].append(table)
            indeg[table] += 1

    q = deque([t for t, d in indeg.items() if d == 0])
    order = []
    while q:
        t = q.popleft()
        order.append(t)
        for nxt in deps[t]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(schema):
        return list(schema.keys())
    return order


def fk_columns(schema: dict, table: str) -> dict:
    return {fk["column"]: fk for fk in schema[table].get("foreign_keys", [])}


def non_pk_columns(schema: dict, table: str) -> list:
    cols = list(schema[table]["columns"].keys())
    pk = schema[table]["pk"]
    return [c for c in cols if c != pk]


def columns_by_type(schema: dict, table: str, types: set) -> list:
    return [
        c
        for c, t in schema[table]["columns"].items()
        if t in types and c != schema[table]["pk"]
    ]

def column_meta(schema: dict, table: str, col: str) -> dict:
    return schema.get(table, {}).get("column_meta", {}).get(col, {})


def column_nullable(schema: dict, table: str, col: str) -> bool:
    meta = schema.get(table, {}).get("column_meta", {}).get(col, {})
    return meta.get("nullable", True)


def table_partition_names(schema: dict, table: str) -> list[str]:
    return list(schema.get(table, {}).get("partitioning", {}).get("names", []))


def table_partitioning(schema: dict, table: str) -> dict:
    return dict(schema.get(table, {}).get("partitioning", {}))
