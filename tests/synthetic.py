"""Re-export: synthetic bars live in the package so the CLI's synthetic mode can use them."""
from tradingsystem.synthetic import SyntheticStore, make_bars

__all__ = ["SyntheticStore", "make_bars"]
