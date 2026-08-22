"""
Regression test for a bug found and fixed this session: web/app.py's
_gather_cheap_enrichments() imported a `PredictionLedger` class that never
existed in data/prediction_ledger.py (that module only has module-level
functions). Wrapped in a bare `except: pass`, this meant /api/decision,
/api/decision-brief, and /api/portfolio-brief always silently got
calibration=None in production - the calibration section of every Decision
Report/Brief always rendered INSUFFICIENT_HISTORY, even with real matured
predictions in the ledger.

Run: python3 tests/test_app_calibration_bugfix.py

Offline/deterministic - data.prediction_ledger's real functions are
monkeypatched with sentinel-returning stubs, so this proves the wiring
(right import, right function names, right kwarg names) reaches all the
way through _gather_cheap_enrichments() instead of being silently
swallowed by the bare except - which is exactly how the original bug hid
for as long as it did.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.prediction_ledger as prediction_ledger_module
import web.app as app_module

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_no_predictionledger_class_reference_remains():
    import inspect
    source = inspect.getsource(app_module._gather_cheap_enrichments)
    check("no reference to the nonexistent PredictionLedger class remains",
          "PredictionLedger" not in source, source)


def test_module_has_no_predictionledger_class():
    check("data.prediction_ledger has no PredictionLedger class (confirms the bug was real)",
          not hasattr(prediction_ledger_module, "PredictionLedger"))


SENTINEL_CALIBRATION = {"overall": {"win_rate": 0.62}, "_sentinel": "calibration"}
SENTINEL_SUMMARY = {"total": 42, "_sentinel": "summary"}


def test_enrichments_actually_reach_the_real_functions():
    # _gather_cheap_enrichments also calls xsection.ranking.run_ranking() for the
    # "production-pilot" universe, a real network-hitting call (yfinance) totally
    # unrelated to the calibration bug this test checks - patch it out so this
    # test stays fast and offline instead of accidentally exercising that path.
    with patch.object(prediction_ledger_module, "calibration_report", return_value=SENTINEL_CALIBRATION) as mock_cal, \
         patch.object(prediction_ledger_module, "summary", return_value=SENTINEL_SUMMARY) as mock_summary, \
         patch("xsection.ranking.run_ranking", side_effect=RuntimeError("should not be called by this test")):
        calibration, prediction_summary, _xsec = app_module._gather_cheap_enrichments({}, "TEST")

    check("calibration is the real function's return value, not None",
          calibration == SENTINEL_CALIBRATION, repr(calibration))
    check("prediction_summary is the real function's return value, not None",
          prediction_summary == SENTINEL_SUMMARY, repr(prediction_summary))
    check("calibration_report was called with the correct kwarg name (horizon=, not horizon_days=)",
          mock_cal.call_args.kwargs.get("horizon") == 20, str(mock_cal.call_args))
    check("summary was called at all", mock_summary.called)


def test_portfolio_brief_shared_enrichments_also_fixed():
    import inspect
    source = inspect.getsource(app_module.portfolio_brief_endpoint) \
        if hasattr(app_module, "portfolio_brief_endpoint") else None
    if source is None:
        # Route function name may differ - locate by scanning the module source instead.
        source = Path(app_module.__file__).read_text()
    check("no PredictionLedger reference remains anywhere in web/app.py",
          "PredictionLedger" not in source)


if __name__ == "__main__":
    test_no_predictionledger_class_reference_remains()
    test_module_has_no_predictionledger_class()
    test_enrichments_actually_reach_the_real_functions()
    test_portfolio_brief_shared_enrichments_also_fixed()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — PredictionLedger bug fix: calibration/summary genuinely wired through")
