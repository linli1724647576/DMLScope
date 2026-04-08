import re


def validate_delete(stmt: str) -> None:
    s = stmt.strip().rstrip(";")
    if not re.search(r"\bDELETE\b", s, re.IGNORECASE):
        raise ValueError(f"DELETE keyword not found: {stmt}")
    if " FROM " not in s.upper():
        raise ValueError(f"Missing FROM in DELETE: {stmt}")
    if " WHERE " not in s.upper():
        raise ValueError(f"Missing WHERE in DELETE: {stmt}")


def test_delete_variant(stmt: str) -> None:
    validate_delete(stmt)


def test_delete_variants(variants: list) -> None:
    if not variants:
        raise ValueError("No DELETE variants to test")
    for stmt in variants:
        test_delete_variant(stmt)