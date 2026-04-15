## Full Documentation of Mutation Rules for Approximate DML Synthesis

Transforming construct `C1` into `C2` achieves under-approximation, while transforming `C2` into `C1` achieves over-approximation.

| Type             | `C1`                        | `C2`                         |
| ---------------- | --------------------------- | ---------------------------- |
| Relation         | `SELECT ...`                | `SELECT DISTINCT ...`        |
|                  | `A UNION ALL B`             | `A UNION B`                  |
|                  | `A UNION B`                 | `A INTERSECT B`              |
|                  | `A`                         | `A EXCEPT B`                 |
|                  | `A UNION B`                 | `A EXCEPT B`                 |
| Predicate        | `P`                         | `FALSE`                      |
|                  | `TRUE`                      | `P`                          |
|                  | `P OR Q`                    | `P AND Q`                    |
|                  | `JOIN ... ON TRUE`          | `JOIN ... ON P`              |
|                  | `P OR TRUE`                 | `P`                          |
|                  | `P`                         | `P AND FALSE`                |
|                  | `x IN (v1, v2, ..., vn)`    | `x IN (v1, v2, ..., v(n-1))` |
|                  | `x LIKE relaxed(pattern)`   | `x LIKE pattern`             |
|                  | `x REGEXP relaxed(pattern)` | `x REGEXP pattern`           |
| Comparison       | `a >= b`                    | `a > b`                      |
|                  | `a <= b`                    | `a < b`                      |
|                  | `x >= l`                    | `x BETWEEN l AND r`          |
|                  | `x <= r`                    | `x BETWEEN l AND r`          |
|                  | `a >= b`                    | `a = b`                      |
|                  | `a <= b`                    | `a = b`                      |
| Join / Structure | `LEFT JOIN`                 | `INNER JOIN`                 |
|                  | `RIGHT JOIN`                | `INNER JOIN`                 |
|                  | `HAVING TRUE`               | `HAVING P`                   |
|                  | `GROUP BY ...`              | `GROUP BY ... HAVING P`      |
|                  | `WHERE FALSE`               | `WHERE P`                    |
|                  | `(remove WHERE)`            | `WHERE P`                    |
