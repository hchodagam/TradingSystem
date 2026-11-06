"""Params: kept free of pydantic so the sandbox child stays cheap to start."""


class Params(dict):
    """Strategy parameters with attribute access: p.rsi_len."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None
