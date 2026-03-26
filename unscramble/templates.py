from __future__ import annotations

import random


def _py_factorial(n_name: str, fn_name: str) -> str:
    return f"""def {fn_name}({n_name}: int) -> int:
    if {n_name} < 0:
        raise ValueError("n must be >= 0")
    out = 1
    for i in range(2, {n_name} + 1):
        out *= i
    return out


def main() -> None:
    print({fn_name}(5))


if __name__ == "__main__":
    main()
"""


def _py_two_sum(arr: str, target: str, fn: str) -> str:
    return f"""from typing import List, Tuple


def {fn}({arr}: List[int], {target}: int) -> Tuple[int, int]:
    seen = {{}}
    for i, x in enumerate({arr}):
        y = {target} - x
        if y in seen:
            return (seen[y], i)
        seen[x] = i
    raise ValueError("no solution")
"""


def _py_csv_sum(path: str, fn: str) -> str:
    return f"""import csv


def {fn}({path}: str) -> float:
    total = 0.0
    with open({path}, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += float(row["amount"])
    return total
"""


def _py_lru_cache(fn: str) -> str:
    return f"""from functools import lru_cache


@lru_cache(maxsize=256)
def {fn}(n: int) -> int:
    if n <= 1:
        return n
    return {fn}(n - 1) + {fn}(n - 2)
"""


def _js_debounce(fn: str) -> str:
    return f"""export function {fn}(func, waitMs) {{
  let timeoutId = null;
  return function(...args) {{
    if (timeoutId !== null) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), waitMs);
  }};
}}
"""


def _js_group_by(fn: str) -> str:
    return f"""export function {fn}(items, keyFn) {{
  const out = new Map();
  for (const it of items) {{
    const k = keyFn(it);
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(it);
  }}
  return out;
}}
"""


def _ts_result_type() -> str:
    return """export type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };

export function ok<T, E = never>(value: T): Result<T, E> {
  return { ok: true, value };
}

export function err<T = never, E = unknown>(error: E): Result<T, E> {
  return { ok: false, error };
}
"""


def _bash_backup(fn: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{fn}() {{
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}}

main() {{
  if [[ $# -ne 2 ]]; then
    echo "usage: $0 <src> <dst>" >&2
    exit 2
  fi
  {fn} "$1" "$2"
}}

main "$@"
"""


def _java_factorial(cls: str) -> str:
    return f"""public class {cls} {{
  public static long factorial(int n) {{
    if (n < 0) throw new IllegalArgumentException("n must be >= 0");
    long out = 1L;
    for (int i = 2; i <= n; i++) out *= i;
    return out;
  }}

  public static void main(String[] args) {{
    System.out.println(factorial(5));
  }}
}}
"""


def _cpp_factorial() -> str:
    return """#include <iostream>
#include <stdexcept>

long long factorial(int n) {
  if (n < 0) throw std::invalid_argument("n must be >= 0");
  long long out = 1;
  for (int i = 2; i <= n; i++) out *= i;
  return out;
}

int main() {
  std::cout << factorial(5) << std::endl;
  return 0;
}
"""


def _go_factorial() -> str:
    return """package main

import "fmt"

func factorial(n int) int {
    if n < 0 {
        return 0
    }
    out := 1
    for i := 2; i <= n; i++ {
        out *= i
    }
    return out
}

func main() {
    fmt.Println(factorial(5))
}
"""


def _rust_factorial() -> str:
    return """fn factorial(n: i64) -> i64 {
    if n < 0 {
        return 0;
    }
    let mut out: i64 = 1;
    let mut i: i64 = 2;
    while i <= n {
        out *= i;
        i += 1;
    }
    out
}

fn main() {
    println!(\"{}\", factorial(5));
}
"""


def _php_factorial(fn: str) -> str:
    return f"""<?php
declare(strict_types=1);

function {fn}(int $n): int {{
  if ($n < 0) {{
    throw new InvalidArgumentException(\"n must be >= 0\");
  }}
  $out = 1;
  for ($i = 2; $i <= $n; $i++) {{
    $out *= $i;
  }}
  return $out;
}}

echo {fn}(5) . PHP_EOL;
"""


def sample_clean_code(rng: random.Random, *, language: str) -> str:
    language = language.lower()
    if language == "python":
        which = rng.choice(["factorial", "two_sum", "csv_sum", "fib_cache"])
        if which == "factorial":
            return _py_factorial(rng.choice(["n", "num", "count"]), rng.choice(["factorial", "fact", "compute_factorial"]))
        if which == "two_sum":
            return _py_two_sum(rng.choice(["nums", "arr", "values"]), rng.choice(["target", "t"]), rng.choice(["two_sum", "find_pair"]))
        if which == "csv_sum":
            return _py_csv_sum(rng.choice(["path", "csv_path", "filename"]), rng.choice(["sum_amounts", "total_amount"]))
        return _py_lru_cache(rng.choice(["fib", "fibonacci"]))

    if language == "javascript":
        return rng.choice([_js_debounce("debounce"), _js_group_by("groupBy")])

    if language == "typescript":
        return _ts_result_type()

    if language == "bash":
        return _bash_backup(rng.choice(["backup", "copy_file"]))

    if language == "java":
        return _java_factorial(rng.choice(["Main", "App", "FactorialApp"]))

    if language == "cpp":
        return _cpp_factorial()

    if language == "go":
        return _go_factorial()

    if language == "rust":
        return _rust_factorial()

    if language == "php":
        return _php_factorial(rng.choice(["factorial", "fact"]))

    raise ValueError(f"unsupported language: {language}")


LANGUAGES = ["python", "javascript", "typescript", "bash", "java", "cpp", "go", "rust", "php"]

