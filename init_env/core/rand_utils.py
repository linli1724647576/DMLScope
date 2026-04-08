import random
import string
from datetime import date, datetime, timedelta

ALPHABET = string.ascii_lowercase
HEX_DIGITS = "0123456789ABCDEF"


def rand_str(min_len=5, max_len=12):
    n = random.randint(min_len, max_len)
    return "".join(random.choice(ALPHABET) for _ in range(n))


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def rand_date() -> str:
    start = date(2000, 1, 1)
    delta_days = random.randint(0, 9500)
    return (start + timedelta(days=delta_days)).isoformat()


def rand_datetime() -> str:
    start = datetime(2000, 1, 1, 0, 0, 0)
    delta_seconds = random.randint(0, 9500 * 24 * 3600)
    dt = start + timedelta(seconds=delta_seconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def rand_time() -> str:
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"



def rand_hex(nbytes=8) -> str:
    return "X'" + "".join(random.choice(HEX_DIGITS) for _ in range(nbytes * 2)) + "'"


def rand_hex_body(nbytes=8) -> str:
    return "".join(random.choice(HEX_DIGITS) for _ in range(nbytes * 2))


def maybe_null(val: str, nullable: bool, null_ratio=0.2) -> str:
    if nullable and random.random() < null_ratio:
        return "NULL"
    return val


def geom_literal() -> str:
    # Simple POINT geometry literal for MySQL-compatible dialects.
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    return f"ST_GeomFromText('POINT({x} {y})')"
