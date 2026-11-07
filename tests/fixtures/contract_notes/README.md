# Real Dhan contract notes

Add one JSON file per contract note (one trading day). `tests/test_costs.py` checks the cost model against every file here to the paisa.

```json
{
  "fills": [
    {"symbol": "INFY", "date": "2024-03-01", "side": "BUY", "qty": 10, "price": 1650.25}
  ],
  "expected": {"stt": 17, "exchange": 0.49, "sebi": 0.02, "stamp": 2, "dp": 0, "gst": 0.09, "total": 19.60}
}
```

Copy the values from the note itself. If a test fails, fix `config/costs.yaml` or the rounding rule in `src/tradingsystem/costs.py`; never edit the expected values.
