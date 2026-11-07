import pytest

from tradingsystem.strategy.static_check import static_check

ESCAPES = {
    "import_os": "import os",
    "import_socket": "import socket",
    "from_import": "from subprocess import run",
    "relative_import": "from . import x",
    "dunder_import": "x = __import__('os')",
    "open_write": "open('x', 'w')",
    "exec": "exec('1')",
    "eval": "eval('1')",
    "getattr": "getattr(1, 'real')",
    "to_csv": "bars.to_csv('x.csv')",
    "to_pickle": "bars.to_pickle('x')",
    "read_csv": "pd.read_csv('x')",
    "np_save": "np.save('x', a)",
    "np_load": "np.load('x')",
    "tofile": "a.tofile('x')",
    "pd_io": "pd.io.common",
    "np_lib": "np.lib.format",
    "datasource": "np.DataSource()",
    "excelwriter": "pd.ExcelWriter('x')",
    "dunder_escape": "().__class__.__bases__",
    "dunder_name": "__builtins__",
    # final-review escapes (C1)
    "sys_modules_chain": 'm = pd.core.common.sys.modules["os"]',
    "submodule_from_import": "from pandas.util._print_versions import os",
    "submodule_import": "import pandas.io.common",
    "ctypeslib": "import numpy.ctypeslib as c",
    "frame_walk": "g = (x for x in []); g.gi_frame.f_back.f_globals",
    "traceback_frame": "e.__traceback__",
    "tb_frame": "t.tb_frame",
    "to_string_path": "bars.to_string('x.txt')",
    "ndarray_dump": "a.dump('x')",
    "fromregex": "np.fromregex('x', 'r', [])",
    "excelfile": "pd.ExcelFile('x')",
    "styler": "bars.style.to_string('x')",
    "private_attr": "pd._libs",
    "pd_eval": "pd.eval('1 + 1')",
    "df_query": "bars.query('close > 1')",
    "from_pandas": "from pandas import read_csv",
}


@pytest.mark.parametrize("name", list(ESCAPES))
def test_static_rejects(name):
    assert static_check(ESCAPES[name]), name


def test_allowed_code_is_clean():
    code = (
        "import pandas as pd\nimport numpy as np\nimport math\n"
        "def signals(bars, p):\n"
        "    c = bars['close']\n"
        "    out = pd.DataFrame(index=bars.index)\n"
        "    out['entry'] = c.to_numpy() > math.pi\n"
        "    out['exit'] = c.rolling(3).mean().to_frame()['close'] < 1\n"
        "    out['stop'] = np.nan\n"
        "    return out\n"
    )
    assert static_check(code) == []


def test_from_math_import_allowed():
    assert static_check("from math import sqrt, log\nimport numpy as np\n") == []


def test_violation_mentions_line_number():
    v = static_check("x = 1\nimport os\n")
    assert "line 2" in v[0]
