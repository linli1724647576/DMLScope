import re


def normalize_sqlglot_input(sql: str, dialect: str) -> str:
    text = sql or ""
    if dialect.lower().strip() != "mysql":
        return text
    return re.sub(
        r"(?i)\bUNSIGNED(?:\s+INTEGER|\s+INT)?\b",
        "BIGINT",
        text,
    )
