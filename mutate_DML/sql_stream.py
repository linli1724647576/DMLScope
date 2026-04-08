from typing import Iterable


def split_sql_stream(lines: Iterable[str]) -> Iterable[str]:
    buf = []
    in_single = False
    in_double = False
    in_backtick = False
    escape = False

    for line in lines:
        for ch in line:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == "\\":
                buf.append(ch)
                escape = True
                continue
            if ch == "'" and not in_double and not in_backtick:
                in_single = not in_single
            elif ch == '"' and not in_single and not in_backtick:
                in_double = not in_double
            elif ch == "`" and not in_single and not in_double:
                in_backtick = not in_backtick

            if ch == ";" and not in_single and not in_double and not in_backtick:
                stmt = "".join(buf).strip()
                if stmt:
                    yield stmt
                buf = []
            else:
                buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        yield tail
