"""Load a strategy file without executing it: SPEC via ast.literal_eval, hash of normalised source."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tradingsystem.strategy.spec import StrategySpec


class StrategyFormatError(Exception):
    pass


@dataclass(frozen=True)
class LoadedStrategy:
    spec: StrategySpec
    code: str
    hash: str
    path: Path


def load_strategy(path: Path) -> LoadedStrategy:
    code = Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    try:
        tree = ast.parse(code, filename=str(path))
    except SyntaxError as e:
        raise StrategyFormatError(f"syntax error at line {e.lineno}: {e.msg}") from e

    spec_node = None
    has_signals = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets):
            spec_node = node.value
        if isinstance(node, ast.FunctionDef) and node.name == "signals":
            args = [a.arg for a in node.args.args]
            if len(args) != 2:
                raise StrategyFormatError(f"signals must take (bars, p); found ({', '.join(args)})")
            has_signals = True
    if spec_node is None:
        raise StrategyFormatError("no top-level SPEC = {...} assignment")
    if not has_signals:
        raise StrategyFormatError("no top-level def signals(bars, p)")
    try:
        raw = ast.literal_eval(spec_node)
    except ValueError as e:
        raise StrategyFormatError("SPEC must be a literal dict (no calls or names)") from e
    try:
        spec = StrategySpec.model_validate(raw)
    except ValidationError as e:
        raise StrategyFormatError(f"SPEC invalid: {e}") from e

    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return LoadedStrategy(spec=spec, code=code, hash=digest, path=Path(path))
