## Full Documentation of Mutation Rules for Approximate DML Synthesis

Transforming construct `C1` into `C2` achieves under-approximation, while transforming `C2` into `C1` achieves over-approximation.

| Type | `C1` | `C2` |
|---|---|---|
| Relation | `SELECT ...` | `SELECT DISTINCT ...` |
|  | `A UNION ALL B` | `A UNION B` |
|  | `A UNION B` | `A INTERSECT B` |
|  | `A` | `A EXCEPT B` |
|  | `A EXCEPT B` | `A UNION B` |
|  | `A INTERSECT B` | `A UNION B` |
| Predicate | `P` | `FALSE` |
|  | `TRUE` | `P` |
|  | `P AND Q` | `P OR Q` |
|  | `JOIN ... ON P` | `JOIN ... ON TRUE` |
|  | `P` | `P OR TRUE` |
|  | `P` | `P AND FALSE` |
|  | `P OR Q` | `P AND Q` |
|  | `x IN (v1, v2, ..., vn)` | `x IN (v1, v2, ..., v(n-1))` |
|  | `x LIKE pattern` | `x LIKE relaxed(pattern)` |
|  | `x REGEXP pattern` | `x REGEXP relaxed(pattern)` |
| Comparison | `a > b` | `a >= b` |
|  | `a < b` | `a <= b` |
|  | `x BETWEEN l AND r` | `x >= l` |
|  | `x BETWEEN l AND r` | `x <= r` |
|  | `a = b` | `a >= b` |
|  | `a = b` | `a <= b` |
| Join / Structure | `INNER JOIN` | `LEFT JOIN` |
|  | `INNER JOIN` | `RIGHT JOIN` |
|  | `HAVING P` | `HAVING TRUE` |
|  | `GROUP BY ... HAVING P` | `GROUP BY ...` |
|  | `WHERE P` | `WHERE FALSE` |
|  | `WHERE P` | `(remove WHERE)` |
