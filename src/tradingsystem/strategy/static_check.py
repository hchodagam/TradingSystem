"""AST check of strategy code before it is ever run.

This is defence in depth, NOT a security boundary: Python offers too many reflective routes for a
denylist to be complete. LLM-written code must not run outside OS-level isolation (no-network,
read-only, low-privilege container) - see the Plan 4 note in the ledger.
"""
from __future__ import annotations

import ast
import re

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "math"})
ALLOWED_FROM = frozenset({"math"})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "memoryview", "help", "exit", "quit", "type", "object", "super",
})
_FILE_ATTR = re.compile(
    r"^(read_.*|to_(csv|excel|parquet|pickle|json|hdf|sql|feather|stata|html|xml|latex|markdown|clipboard|orc"
    r"|string|gbq)"
    r"|save|savez|savez_compressed|savetxt|load|loadtxt|genfromtxt|fromfile|fromregex|tofile|memmap|dump|dumps"
    r"|DataSource|ExcelWriter|ExcelFile|HDFStore|style|eval|query)$"
)
# attribute names that name modules or reach interpreter internals
BANNED_ATTRS = frozenset({
    "sys", "os", "builtins", "ctypes", "ctypeslib", "f2py", "distutils", "testing", "importlib", "subprocess",
    "socket", "io", "lib", "shutil", "pathlib", "core", "util", "compat", "api", "plotting", "errors",
    "modules", "environ", "system", "popen", "spawn", "loader", "spec",
})
_FRAME_ATTR = re.compile(r"^(gi|cr|ag|f|tb|co)_")


def _dunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


def static_check(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]
    out: list[str] = []

    def bad(node: ast.AST, msg: str) -> None:
        out.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    bad(node, f"import of {a.name!r} not allowed (only: import {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_FROM:
                bad(node, f"'from {node.module or '.'} import' not allowed (only 'from math import ...')")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or _dunder(node.id):
                bad(node, f"use of {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                bad(node, f"private attribute {a!r} not allowed")
            elif _FILE_ATTR.match(a):
                bad(node, f"file/IO/eval attribute {a!r} not allowed")
            elif a in BANNED_ATTRS:
                bad(node, f"attribute {a!r} reaches modules or internals; not allowed")
            elif _FRAME_ATTR.match(a):
                bad(node, f"frame/code attribute {a!r} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad(node, "global/nonlocal not allowed")
    return out
