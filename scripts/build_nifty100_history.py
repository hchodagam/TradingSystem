"""Rebuild point-in-time Nifty 100 membership (reference/nifty100_changes.csv) backwards from today's constituents and NSE notices.

Each event: (effective date, notice, [excluded], [included]) in the notice's own symbols. Symbols are mapped to
today's NSE symbol via NSE's symbolchange.csv. Walking newest -> oldest, an event's included symbols must be members
after it and its excluded ones must not; the count must stay 100 (101-104 only while a DVR or a demerger spin-off
is temporarily included). Writes reference/nifty100_changes.csv: the snapshot at START, then every change.

Sources: NSE Indices press releases (niftyindices.com/Press_Release/<notice>.pdf) and NSE's symbol change file
(nsearchives.nseindia.com/content/equities/symbolchange.csv; pass its path, default data/symbolchange.csv).
Usage: uv run python scripts/build_nifty100_history.py [path/to/symbolchange.csv]
"""
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

here = Path(__file__).parent
SYMBOLCHANGE = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else Path('data/symbolchange.csv')
START = date(2012, 1, 1)

E = [  # (effective, notice, excluded, included)
    ("2012-04-27", "ind_prs14032012", "HDIL IFCI IOB", "DIVISLAB GSKCONS PETRONET"),
    ("2012-09-28", "ind_prs16082012", "ANDHRABANK BEL STER", "APOLLOHOSP BAJAJHLDNG GODREJCP"),
    ("2013-04-01", "ind_prs13022013", "BIOCON GMRINFRA TORNTPOWER WIPRO", "BAJAJFINSV NMDC TATAGLOBAL UBL"),
    ("2013-09-27", "ind_prs27082013", "ASHOKLEY INDHOTEL RELINFRA", "M&MFIN OIL WIPRO"),
    ("2014-03-28", "ind_prs27022014", "IDBI JPASSOCIAT", "INFRATEL INGVYSYABK"),
    ("2014-09-19", "ind_prs20082014", "MPHASIS MCDOWELL-N", "AUROPHARMA MOTHERSUMI"),
    ("2014-11-28", "ind_prs18112014", "RANBAXY", "RELINFRA"),
    ("2015-02-02", "ind_prs21012015", "INGVYSYABK", "EICHERMOT"),
    ("2015-03-27", "ind_prs20022015", "DLF JINDALSTEL", "IBULHSGFIN MCDOWELL-N"),
    ("2015-05-29", "ind_prs29042015", "ADANIENT IDFC", "BRITANNIA MRF"),
    ("2015-09-28", "ind_prs24082015", "CROMPGREAV", "ASHOKLEY"),
    ("2015-09-28", "ind_prs12082015", "RPOWER UNIONBANK", "MARICO SUNDARMFIN"),
    ("2015-10-19", "ind_prs18092015", "ABIRLANUVO", "IOC"),
    ("2016-04-01", "ind_prs22022016_2",
     "BAJAJHLDNG BANKINDIA CANBK EXIDEIND FEDERALBNK M&MFIN MRF PETRONET RELCAPITAL RELINFRA SUNDARMFIN TATACHEM "
     "TATAGLOBAL",
     "ABB BAJFINANCE BEL CADILAHC CASTROLIND DLF EMAMILTD HINDZINC NHPC PIDILITIND PGHH SHREECEM TATAMTRDVR "
     "TORNTPHARM"),
    ("2016-09-30", "ind_prs12082016", "RCOM RECLTD", "INDIGO PEL"),
    ("2016-11-15", "ind_prs17102016", "CAIRN", "HAVELLS"),
    ("2017-03-31", "ind_prs16022017", "APOLLOHOSP BHARATFORG CASTROLIND", "ICICIPRULI PETRONET RECLTD"),
    ("2017-05-26", "ind_prs27042017", "GRASIM", "SUNTV"),
    ("2017-09-29", "ind_prs28082017", "DIVISLAB UBL", "DMART MRF"),
    ("2018-04-02", "ind_prs21022018", "GSKCONS GLAXO GLENMARK TATAPOWER TORNTPHARM", "ABCAPITAL GICRE GRASIM L&TFH SBILIFE"),
    ("2018-09-28", "ind_prs28082018", "CUMMINSIND EMAMILTD PFC PNB RECLTD", "BANDHANBNK BIOCON HDFCLIFE ICICIGI NIACL"),
    ("2019-03-29", "ind_prs25022019", "ABCAPITAL BEL LICHSGFIN OIL SUNTV", "BAJAJHLDNG DIVISLAB HDFCAMC PAGEIND UBL"),
    ("2019-09-27", "ind_prs28082019", "ABB BHEL MRF SAIL", "BERGEPAINT NESTLEIND PFC PNB"),
    ("2020-03-19", "ind_prs16032020", "YESBANK", "ADANITRANS"),
    ("2020-06-26", "ind_prs10062020", "ASHOKLEY IBULHSGFIN L&TFH NIACL IDEA TATAMTRDVR",
     "ABBOTINDIA IGL NAUKRI MUTHOOTFIN TORNTPHARM"),
    ("2020-07-31", "ind_prs02072020_1", "VEDL", "SBICARD"),
    ("2020-09-25", "ind_prs20082020", "NHPC PAGEIND SRTRANSFIN ZEEL", "ADANIGREEN ALKEM LTI TATACONSUM"),
    ("2021-03-31", "ind_prs23022021", "BANKBARODA CONCOR GICRE HINDZINC OFSS PFC",
     "ADANIENT APOLLOHOSP JUBLFOOD MRF VEDL YESBANK"),
    ("2021-06-30", "ind_prs15062021", "MOTHERSUMI", "GLAND"),
    ("2021-09-30", "ind_prs23082021", "ABBOTINDIA ALKEM MRF PETRONET UBL", "BANKBARODA CHOLAFIN JINDALSTEL PIIND SAIL"),
    ("2022-03-31", "ind_prs24022022_1", "AUROPHARMA HINDPETRO IGL JINDALSTEL YESBANK", "NYKAA MINDTREE PAYTM SRF ZOMATO"),
    ("2022-08-08", "ind_prs11072022", "NMDC PEL", "LICI TATAPOWER"),
    ("2022-09-30", "ind_prs01092022", "JUBLFOOD LUPIN MINDTREE PNB SAIL ZYDUSLIFE", "ATGL BEL HAL IRCTC MPHASIS MOTHERSON"),
    ("2023-03-31", "ind_prs17022023_1", "BANDHANBNK BIOCON GLAND MPHASIS PAYTM", "ABB AWL CANBK PAGEIND VBL"),
    ("2023-07-13", "ind_prs04072023", "HDFC", "JINDALSTEL"),
    ("2023-07-20", "ind_prs17072023", "", "JIOFIN"),
    ("2023-09-07", "ind_prs05092023", "JIOFIN", ""),
    ("2023-09-29", "ind_prs17082023", "ACC NYKAA HDFCAMC INDUSTOWER PAGEIND",
     "PNB SHRIRAMFIN TRENT TVSMOTOR ZYDUSLIFE TATAMTRDVR"),
    ("2024-03-28", "ind_prs28022024", "AWL MUTHOOTFIN PIIND PGHH UPL", "ADANIPOWER IRFC JIOFIN PFC RECLTD"),
    ("2024-08-30", "ind_prs23082024_1", "TATAMTRDVR", ""),
    ("2024-09-30", "ind_prs23082024", "BERGEPAINT COLPAL MARICO SBICARD SRF", "BHEL JSWENERGY LODHA NHPC UNIONBANK"),
    ("2025-01-06", "ind_prs30122024", "", "ITCHOTELS"),
    ("2025-02-10", "ind_prs06022025_1", "ITCHOTELS", ""),
    ("2025-03-28", "ind_prs21022025", "ATGL BHEL IRCTC NHPC UNIONBANK", "BAJAJHFL CGPOWER HYUNDAI INDHOTEL SWIGGY"),
    ("2025-04-07", "ind_prs02042025", "", "ENRIN"),
    ("2025-06-27", "ind_prs25062025_2", "ENRIN", ""),
    ("2025-09-30", "ind_prs22082025", "DABUR HEROMOTOCO ICICIPRULI INDUSINDBK SWIGGY",
     "HINDZINC MAXHEALTH MAZDOCK ENRIN SOLARINDS"),
    ("2025-10-14", "ind_prs07102025", "", "TMCV"),
    ("2025-11-17", "ind_prs13112025", "TMCV", ""),
    ("2025-12-05", "ind_prs28112025", "", "KWIL"),
    ("2026-02-24", "ind_prs20022026_1", "KWIL", ""),
    ("2026-03-30", "ind_prs23022026", "BAJAJHFL HAVELLS ICICIGI NAUKRI JSWENERGY LICI",
     "CUMMINSIND HDFCAMC MUTHOOTFIN TATACAP TMCV UNIONBANK"),
    ("2026-04-30", "ind_prs23042026", "", "VAML VEDPOWER VOGL VISL"),
    ("2026-06-19", "ind_prs17062026", "VEDPOWER VISL", ""),
    ("2026-06-23", "ind_prs19062026", "VAML", ""),
    ("2026-06-24", "ind_prs22062026", "VOGL", ""),
    ("2026-09-30", "ind_prs10082026", "INDHOTEL LODHA RECLTD SHREECEM UNITDSPR", "BSE POWERINDIA POLYCAB VAML IDEA"),
]

# NSE symbol changes, chained to today's symbol
renames: dict[str, str] = {}
with open(SYMBOLCHANGE, newline="", encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        old, new = r[" OLD SYMBOL" if " OLD SYMBOL" in r else "OLD SYMBOL"].strip(), r["NEW SYMBOL"].strip()
        if old and new and old != new:
            renames[old] = new


def current(sym: str) -> str:
    seen = set()
    while sym in renames and sym not in seen:
        seen.add(sym)
        sym = renames[sym]
    return sym


with open(Path("reference/nifty100_current.csv"), newline="", encoding="utf-8") as f:  # NSE list as of 2026-11-07
    today = {r["symbol"] for r in csv.DictReader(f)}
assert len(today) == 100, len(today)

members = set(today)
problems = []
history = []  # (date, set after event)
for eff, notice, exc, inc in sorted(E, key=lambda e: e[0], reverse=True):
    x = {current(s) for s in exc.split()}
    i = {current(s) for s in inc.split()}
    if not i <= members:
        problems.append(f"{eff} {notice}: included but not a member afterwards: {sorted(i - members)}")
    if x & members:
        problems.append(f"{eff} {notice}: excluded but still a member afterwards: {sorted(x & members)}")
    history.append((eff, len(members)))
    members = (members - i) | x
    if not 100 <= len(members) <= 104:
        problems.append(f"before {eff} {notice}: {len(members)} members")
history.append((START.isoformat(), len(members)))

print("members at", START, ":", len(members))
print("problems:", len(problems))
for p in problems:
    print("  ", p)
mapped = sorted({s: current(s) for e in E for s in (e[2] + " " + e[3]).split() if current(s) != s}.items())
print("renamed in notices -> today:", mapped)
if problems and "--force" not in sys.argv:
    sys.exit(1)

rows = [(START.isoformat(), s, "add") for s in sorted(members)]
for eff, notice, exc, inc in sorted(E, key=lambda e: e[0]):
    rows += [(eff, current(s), "remove") for s in exc.split()]
    rows += [(eff, current(s), "add") for s in inc.split()]
with open("reference/nifty100_changes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "symbol", "action"])
    w.writerows(rows)
counts = defaultdict(int)
for eff, n in history:
    counts[n] += 1
print("wrote", len(rows), "rows; member-count histogram after events:", dict(counts))
print("all symbols ever:", len({r[1] for r in rows}))
