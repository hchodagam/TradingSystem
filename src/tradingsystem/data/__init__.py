"""Market data: Dhan adapter, Parquet cache, corporate-action adjustment, validation.

Bars (the frame every unit passes around): MultiIndex (date: Timestamp, symbol: str), sorted,
float columns open, high, low, close, volume.
"""
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
