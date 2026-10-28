#!/usr/bin/env python3
"""Convert `crypto_regime_xgboost.py` into `crypto_regime_xgboost.ipynb`.

The module is written in jupytext "percent" format, so the .py file is the single source of
truth and the notebook is a generated artefact. Two notebook-only cells are added at the top
(Colab install hints and the config cheat sheet) and the trailing `__main__` guard is dropped.

Usage:
    python build_notebook.py                 # writes crypto_regime_xgboost.ipynb
    python build_notebook.py --check         # verify the committed notebook is up to date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "crypto_regime_xgboost.py"
NOTEBOOK = ROOT / "crypto_regime_xgboost.ipynb"

INTRO = """# Crypto Regime Prediction with XGBoost — Range vs Trend (4h timeframe)

Predicts, on every closed 4h candle, **the probability that the next H hours will be a
trending rather than a ranging regime**, for H in {4, 8, 12, 16, 24}.

**How to run on Google Colab**

1. `Runtime -> Change runtime type -> Python 3`, CPU is enough (a GPU adds nothing for XGBoost here).
2. `Runtime -> Run all`. The first cell installs/upgrades nothing unless needed; Colab already
   ships `numpy`, `pandas`, `scikit-learn`, `xgboost` and `matplotlib`.
3. Data is downloaded automatically from `data.binance.vision` (no API key, no rate limit).
   First run takes a few minutes; later runs reuse the `data/cache` folder.
4. Scroll to **section 13** for the results tables and **section 12** for the live
   "70% probability of trending over the next 12h"-style output.

Everything is plain Python, so the same file runs unchanged on a laptop:
`python crypto_regime_xgboost.py`.
"""

INSTALL = """# Colab: everything used below is preinstalled. Uncomment only if an import fails.
# !pip -q install xgboost scikit-learn pandas numpy matplotlib

import sys, platform
print(platform.python_version())
for mod in ("numpy", "pandas", "sklearn", "xgboost", "matplotlib"):
    try:
        m = __import__(mod)
        print(f"{mod:12s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:12s} MISSING -> pip install {mod}")
"""

CHEATSHEET = """## The knobs that matter most (all live in `Config`, section 1)

```python
Config(
    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),   # more coins -> more rows -> better model
    start="2019-01-01",                          # longer history -> more regime variety
    horizons_hours=(4, 8, 12, 16, 24),           # what you asked for
    label_method="efficiency",                   # "efficiency" | "r2" | "sign_persistence"
    label_quantile=0.5,                          # 0.65 -> only the most directional windows
    label_absolute_threshold=None,               # e.g. 0.35 to fix the ER cut-off in advance
    min_abs_fwd_return_quantile=None,            # e.g. 0.9 to ignore "dead" markets
    test_frac=0.20,                              # last 20% of history = untouched holdout
    n_cv_splits=4,                               # purged walk-forward folds
    run_hpo=True,                                # purged-CV random search
    calibration="isotonic",                      # kept only if it improves held-out OOF
)
```
"""


def parse_percent(text: str):
    """Split jupytext percent source into notebook cells."""
    cells, buf, kind = [], [], None

    def flush():
        nonlocal buf, kind
        if kind is None:
            buf = []
            return
        src = "\n".join(buf).strip("\n")
        if kind == "markdown":
            # drop the leading "# " of comment-style markdown
            src = "\n".join(re.sub(r"^# ?", "", ln) for ln in src.splitlines())
        if src.strip():
            cells.append((kind, src))
        buf = []

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        if kind is None:
            kind = "code"      # leading code before the first marker
        buf.append(line)
    flush()

    # the module's trailing __main__ guard is meaningless inside a notebook
    cleaned = []
    for k, src in cells:
        if k == "code":
            src = re.sub(
                r'(?:^|\n)\s*if __name__ == "__main__":\n(?:[ \t]+.*\n?)*\Z', "\n", src
            ).rstrip()
            if not src:
                continue
        cleaned.append((k, src))
    return cleaned


def build() -> dict:
    cells = [
        ("markdown", INTRO),
        ("code", INSTALL),
        ("markdown", CHEATSHEET),
    ]
    cells += parse_percent(SOURCE.read_text())

    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": f"cell-{i:03d}",
                             "metadata": {}, "source": lines})
        else:
            nb_cells.append({"cell_type": "code", "id": f"cell-{i:03d}", "metadata": {},
                             "execution_count": None, "outputs": [], "source": lines})

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the notebook differs from the generated version")
    args = ap.parse_args()

    nb = build()
    payload = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if note := (NOTEBOOK.read_text() if NOTEBOOK.exists() else ""):
            if note != payload:
                print(f"{NOTEBOOK.name} is stale -> run `python build_notebook.py`")
                return 1
        print("notebook is up to date")
        return 0

    NOTEBOOK.write_text(payload)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = len(nb["cells"]) - n_code
    print(f"wrote {NOTEBOOK.name}: {len(nb['cells'])} cells ({n_code} code, {n_md} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
