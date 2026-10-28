#!/usr/bin/env python3
"""End-to-end self-test. No internet required.

Checks:
  1. the generated notebook parses and its code cells all execute in order
     (catches copy/paste or build_notebook regressions);
  2. the leakage audits pass;
  3. the pipeline produces sane metrics on a synthetic 3-regime market;
  4. a live prediction can be produced and serialises to the expected columns.

Run:  python smoke_test.py            # ~1-2 minutes
      python smoke_test.py --quick    # fewer horizons/rows
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def notebook_namespace() -> dict:
    """Execute every notebook code cell except the final `main(CONFIG)` run."""
    nb = json.loads((ROOT / "crypto_regime_xgboost.ipynb").read_text())
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # the run cell would hit the network; it is exercised separately below
    code_cells = [c for c in code_cells if "OUT = main(CONFIG)" not in c]
    # dataclasses resolve their module through sys.modules, so exec into a real module
    mod = types.ModuleType("notebook_test")
    sys.modules[mod.__name__] = mod
    ns: dict = mod.__dict__
    ns["__name__"] = "notebook_test"
    for i, src in enumerate(code_cells):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"notebook code cell {i} failed: {e}") from e
    assert "main" in ns and "Config" in ns, "notebook did not define main/Config"
    return ns


def main(quick: bool = False) -> int:
    print("=" * 70)
    print("1) executing notebook code cells")
    print("=" * 70)
    ns = notebook_namespace()
    print(f"   ok: {len(ns)} names defined, dataclasses/pandas/xgboost imported")

    Config = ns["Config"]
    run_main = ns["main"]

    print("=" * 70)
    print("2) pipeline on a synthetic 3-regime market (offline)")
    print("=" * 70)
    cfg = Config(
        symbols=("SYNTHA", "SYNTHB"),
        market_symbol="SYNTHA",
        data_source="synthetic",
        start="2019-01-01",
        horizons_hours=(4, 12, 24) if quick else (4, 8, 12, 16, 24),
        run_hpo=False,
        hpo_iter=2,
        n_cv_splits=3,
        bootstrap_reps=60,
        run_permutation_test=True,
        save_figures=False,
        out_dir="/tmp/crypto_regime_smoke",
    )
    cfg.params = {**cfg.params, "n_estimators": 400 if quick else 700}
    cfg.early_stopping_rounds = 50

    out = run_main(cfg)
    summary = out["summary"]

    # ---------------- assertions ----------------
    print("=" * 70)
    print("3) assertions")
    print("=" * 70)

    audit = out["causality_audit"]
    assert audit["pass"].all(), f"causality audit failed:\n{audit[~audit['pass']]}"
    print(f"   ok: {len(audit)} features are causal (unchanged when future bars are removed)")

    assert set(summary["horizon_hours"]) == set(cfg.horizons_hours), "missing horizons"
    for col in ("accuracy", "roc_auc", "log_loss", "brier", "mcc", "balanced_acc"):
        assert summary[col].notna().all(), f"{col} has NaNs"
    print(f"   ok: metrics present for horizons {list(summary['horizon_hours'])}")

    # a real (regime-switching) signal must be learnable: AUC clearly above chance
    best_auc = summary["roc_auc"].max()
    assert best_auc > 0.55, f"synthetic signal not learned (best AUC={best_auc:.3f})"
    print(f"   ok: best test AUC on synthetic data = {best_auc:.3f} (> 0.55)")

    # balanced classes by construction -> accuracy must be meaningfully above 50%
    assert summary["accuracy"].max() > 0.53, "no skill over a coin flip on synthetic data"
    print(f"   ok: best test accuracy = {summary['accuracy'].max():.3f}")

    over = out["overfitting"]
    assert (over["test_minus_cv"] < 0.10).all(), f"test much worse than CV:\n{over}"
    print("   ok: test log loss is within 0.10 of purged-CV log loss (no big generalisation gap)")

    live = out["live"]
    assert not live.empty, "no live predictions produced"
    assert set(live["regime"]) <= {"TRENDING", "RANGING"}
    assert live["p_trending"].between(0, 1).all()
    print(f"   ok: {len(live)} live predictions, e.g. "
          f"{live.iloc[0]['symbol']} {live.iloc[0]['horizon_hours']}h -> "
          f"{live.iloc[0]['p_trending']:.2%} trending")

    for h, res in out["results"].items():
        t = res["threshold_analysis"]
        assert len(t) >= 5 and t["coverage"].between(0, 1).all(), "bad decision table"
    print("   ok: decision tables (P >= t -> coverage/precision/lift) computed for every horizon")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
