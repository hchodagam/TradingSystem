"""Parameter grids for walk-forward fitting and sensitivity."""
from __future__ import annotations

import itertools

import numpy as np

from tradingsystem.strategy.spec import Params, ParamSpec, StrategySpec


def _clean(p: ParamSpec, v: float):
    lo, hi = p.range
    v = min(max(v, lo), hi)
    return int(round(v)) if p.is_int else round(float(v), 10)


def axis(p: ParamSpec, n: int) -> list:
    lo, hi = p.range
    vals = [_clean(p, v) for v in np.linspace(lo, hi, n)]
    return list(dict.fromkeys(vals))


def param_grid(spec: StrategySpec, n_per_param: int, max_combos: int, seed: int) -> list[Params]:
    """Linear grid per param; seeded random sample when the product exceeds max_combos.
    The defaults are always first."""
    names = list(spec.params)
    defaults = spec.defaults()
    combos = [tuple(c) for c in itertools.product(*(axis(spec.params[k], n_per_param) for k in names))]
    dkey = tuple(defaults[k] for k in names)
    combos = [c for c in combos if c != dkey]
    if len(combos) + 1 > max_combos:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(combos), size=max_combos - 1, replace=False))
        combos = [combos[i] for i in idx]
    return [defaults] + [Params(zip(names, c, strict=True)) for c in combos]


def neighbours(spec: StrategySpec, step_pct: float) -> list[Params]:
    """One param at a time at default +/- {step/2, step} x range width (clamped, deduped, != default)."""
    base = spec.defaults()
    out: list[Params] = []
    seen = {tuple(sorted(base.items()))}
    for k, p in spec.params.items():
        width = p.range[1] - p.range[0]
        for f in (-step_pct, -step_pct / 2, step_pct / 2, step_pct):
            q = Params(base)
            q[k] = _clean(p, p.default + f * width)
            key = tuple(sorted(q.items()))
            if key not in seen:
                seen.add(key)
                out.append(q)
    return out
