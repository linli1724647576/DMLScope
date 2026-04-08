import re


def validate_update(stmt: str) -> None:
    s = stmt.strip().rstrip(";")
    if not re.search(r"\bUPDATE\b", s, re.IGNORECASE):
        raise ValueError(f"UPDATE keyword not found: {stmt}")
    if " SET " not in s.upper():
        raise ValueError(f"Missing SET in UPDATE: {stmt}")
    if " WHERE " not in s.upper():
        raise ValueError(f"Missing WHERE in UPDATE: {stmt}")


def test_update_variant(stmt: str) -> None:
    validate_update(stmt)


def test_update_variants(variants: list) -> None:
    if not variants:
        raise ValueError("No UPDATE variants to test")
    for stmt in variants:
        test_update_variant(stmt)