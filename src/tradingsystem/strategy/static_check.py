"""AST check of strategy code before it is ever run."""
from __future__ import annotations

import ast
import re

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "math"})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "memoryview", "help", "exit", "quit",
})
_FILE_ATTR = re.compile(
    r"^(read_.*|to_(csv|excel|parquet|pickle|json|hdf|sql|feather|stata|html|xml|latex|markdown|clipboard|orc)"
    r"|save|savez|savez_compressed|savetxt|load|loadtxt|genfromtxt|fromfile|tofile|memmap"
    r"|DataSource|ExcelWriter|HDFStore|ctypeslib|f2py|distutils|testing)$"
)
BANNED_CHAIN_ATTRS = frozenset({"io", "lib"})  # pd.io, np.lib


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
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    bad(node, f"import of {a.name!r} not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                bad(node, f"import from {node.module or '.'!r} not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or _dunder(node.id):
                bad(node, f"use of {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute):
            if _dunder(node.attr):
                bad(node, f"dunder attribute {node.attr!r} not allowed")
            elif _FILE_ATTR.match(node.attr):
                bad(node, f"file/IO attribute {node.attr!r} not allowed")
            elif node.attr in BANNED_CHAIN_ATTRS and isinstance(node.value, ast.Name):
                bad(node, f"{node.value.id}.{node.attr} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad(node, "global/nonlocal not allowed")
    return out
