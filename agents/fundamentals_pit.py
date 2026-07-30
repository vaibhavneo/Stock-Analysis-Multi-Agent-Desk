"""
Point-in-time fundamentals for the live analysis — evidence-backed.

The 7-agent pipeline's fundamentals came from yfinance: the CURRENT (restated)
view, with no provenance. This module is the honest alternative — it pulls
as-reported fundamentals from EDGAR through the FinancialDataGateway and computes
ratios as EvidenceLedger claims, so every number a report shows can be traced to
the exact SEC filing it came from.

Design choices that matter:
  - Every ratio is recorded via ledger.record_claim(), which REFUSES empty
    evidence. An unsupported fundamental number is structurally impossible here.
  - `as_of` is honored end to end: ask for 2020-01-01 and you get what a filer
    had disclosed by then, not today's restatement.
  - Non-filers (ETFs, ADRs, crypto, most foreign issuers) return a clearly
    LABELLED "unavailable" result — never a fabricated fallback. The live app
    shows "not an SEC filer" rather than a wrong number.

This does NOT replace the yfinance fundamentals in market_data.py yet (that is a
larger migration touching the live agents); it is an additive, evidence-backed
panel the app can show alongside them, and the foundation the eventual swap
builds on.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Ratios we compute, each as (name, formula string, numerator, denominator).
# The formula string's variable names match the evidence keys so explain() reads
# as arithmetic a human can re-check against the filings.
_RATIOS = [
    ("net_margin", "net_income / revenue", "net_income", "revenue"),
    ("gross_margin", "gross_profit / revenue", "gross_profit", "revenue"),
    ("operating_margin", "operating_income / revenue", "operating_income", "revenue"),
    ("return_on_equity", "net_income / equity", "net_income", "equity"),
    ("return_on_assets", "net_income / assets", "net_income", "assets"),
    ("debt_to_equity", "long_term_debt / equity", "long_term_debt", "equity"),
]

# Raw concepts surfaced directly (as claims with a trivial identity formula) so
# the app can show the as-reported line items, each traceable to its filing.
_RAW_CONCEPTS = ["revenue", "net_income", "assets", "equity", "operating_cash_flow"]


def analyze_fundamentals_pit(
    ticker: str,
    as_of: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evidence-backed point-in-time fundamentals for one ticker.

    Returns a dict the web layer can render directly:
      {
        available: bool,          # False => not an SEC filer (labelled, not faked)
        reason: str|None,         # why, when unavailable
        as_of, as_of_honored,     # the PIT guarantee, surfaced
        concepts: {name: {value, unit, filed, accession, claim_id}},
        ratios:   {name: {value, formula, claim_id, inputs_filed}},
        provider, warnings,
      }
    Never raises for the ordinary "no data" case — the live dashboard must
    degrade to an honest empty panel, not a 500.
    """
    from data import ledger
    from financial_data import GatewayError, get
    from financial_data.providers.edgar import NotConfiguredError

    out: Dict[str, Any] = {
        "ticker": ticker.upper(), "available": False, "reason": None,
        "as_of": as_of, "as_of_honored": False,
        "concepts": {}, "ratios": {}, "provider": None, "warnings": [],
    }

    try:
        res = get("fundamentals_pit", ticker, as_of=as_of,
                  concepts=list({c for c in _RAW_CONCEPTS} |
                                {n for _, _, n, d in _RATIOS} |
                                {d for _, _, n, d in _RATIOS}))
    except NotConfiguredError as e:
        out["reason"] = f"EDGAR not configured: {str(e).splitlines()[0]}"
        return out
    except GatewayError as e:
        out["reason"] = f"gateway error: {e}"
        return out
    except Exception as e:                      # network etc. — honest, not fatal
        out["reason"] = f"unavailable: {type(e).__name__}: {str(e)[:80]}"
        return out

    out["provider"] = res.get("provider")
    out["as_of_honored"] = res.get("as_of_honored", False)
    out["warnings"] = res.get("warnings", [])

    if not res["data"]:
        # Distinguish "not a filer" from "we simply have nothing" — both honest,
        # but they mean different things to a user.
        unavail = res.get("unavailable") or []
        reason = unavail[0]["reason"] if unavail else "no us-gaap fundamentals"
        out["reason"] = f"not available ({reason})"
        return out

    # Most-recent surviving datum per concept (data is already PIT-filtered and
    # restatement-collapsed by the gateway).
    latest: Dict[str, Dict[str, Any]] = {}
    for d in res["data"]:
        c = d.get("concept")
        cur = latest.get(c)
        if cur is None or str(d.get("period_end") or "") > str(cur.get("period_end") or ""):
            latest[c] = d

    if not latest:
        out["reason"] = "no usable concepts"
        return out

    out["available"] = True

    # Raw concepts as identity claims (value traces to its own filing).
    for name in _RAW_CONCEPTS:
        d = latest.get(name)
        if not d:
            continue
        cid = _safe_claim(ledger, name, d["value"], f"as-reported {name}",
                          {name: d}, ticker, d.get("unit"), as_of,
                          res.get("as_of_honored", False), run_id)
        out["concepts"][name] = {
            "value": d["value"], "unit": d.get("unit"),
            "period_end": d.get("period_end"), "filed": d.get("available_at"),
            "accession": (d.get("source") or {}).get("document"),
            "us_gaap_tag": (d.get("source") or {}).get("ref"),
            "claim_id": cid,
        }

    # Derived ratios — each an evidence-backed claim over two datums.
    for name, formula, num_key, den_key in _RATIOS:
        num, den = latest.get(num_key), latest.get(den_key)
        if not num or not den:
            continue
        try:
            dv = float(den["value"])
            if dv == 0:
                continue
            value = float(num["value"]) / dv
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        cid = _safe_claim(ledger, name, value, formula,
                          {num_key: num, den_key: den}, ticker, "ratio", as_of,
                          res.get("as_of_honored", False), run_id)
        out["ratios"][name] = {
            "value": round(value, 6), "formula": formula, "claim_id": cid,
            "inputs_filed": {num_key: num.get("available_at"),
                             den_key: den.get("available_at")},
        }

    return out


def _safe_claim(ledger, name, value, formula, evidence, ticker, unit, as_of,
                honored, run_id) -> Optional[str]:
    """Record a claim; never let a ledger hiccup break a user's analysis.

    A failure here costs traceability for that one number, not the whole report,
    so it degrades to claim_id=None (the value still shows, just without a stored
    audit row) rather than raising.
    """
    try:
        return ledger.record_claim(
            name, float(value) if value is not None else None, formula, evidence,
            symbol=ticker.upper(), unit=unit, as_of=as_of,
            as_of_honored=bool(honored), run_id=run_id)
    except Exception:
        return None
