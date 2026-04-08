import re


def validate_replace(stmt: str) -> None:
    s = stmt.strip().rstrip(";")
    upper = s.upper()
    if upper.startswith("REPLACE "):
        return
    if upper.startswith("WITH ") and re.search(r"\bREPLACE\b", upper, flags=re.IGNORECASE):
        return
    if not upper.startswith("REPLACE "):
        raise ValueError(f"Invalid REPLACE: {stmt}")


def test_replace_variant(stmt: str) -> None:
    validate_replace(stmt)


def test_replace_variants(variants: list) -> None:
    if not variants:
        raise ValueError("No REPLACE variants to test")
    for stmt in variants:
        test_replace_variant(stmt)
