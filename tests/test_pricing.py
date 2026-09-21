#!/usr/bin/env python3
"""The local derivatives engine.

This engine exists because OptionsPilot covers 31 equity names and nothing
else. Everything here is MODEL-priced off realized volatility, which is a
weaker claim than a quote — so the tests care about two separate things:

  1. Is the math right? Checked against published Black-Scholes values, and
     against FINITE DIFFERENCES of the engine's own price function. A greek
     that disagrees with the numerical derivative of the price it claims to
     differentiate is wrong no matter how good the formula looks.

  2. Does it refuse to lie? An implied vol that cannot exist returns None
     rather than a clamped boundary; a structure with unbounded loss raises
     rather than building; probabilities are labelled risk-neutral.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas.pricing import black_scholes as bs
from mas.pricing import structures as st
from mas.pricing.payoff import evaluate, payoff_at, net_cost

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


# ── Published reference values ────────────────────────────────────────────

def test_matches_published_black_scholes_values():
    """S=100 K=100 T=1 r=5% sigma=20% q=0 -> C=10.4506, P=5.5735."""
    c = bs.price("call", 100, 100, 1.0, 0.05, 0.20)
    p = bs.price("put", 100, 100, 1.0, 0.05, 0.20)
    check("call matches reference", abs(c - 10.450584) < 1e-5, f"{c:.6f}")
    check("put matches reference", abs(p - 5.573526) < 1e-5, f"{p:.6f}")


def test_put_call_parity_holds_across_a_grid():
    """C - P = S*e^-qT - K*e^-rT. Parity is an arbitrage identity, not an
    approximation: any violation is a bug in one of the two branches."""
    worst = 0.0
    for S in (50, 100, 250):
        for K in (60, 100, 200):
            for T in (0.08, 0.5, 2.0):
                for sig in (0.1, 0.35, 0.9):
                    for r in (0.0, 0.05):
                        for q in (0.0, 0.03):
                            c = bs.price("call", S, K, T, r, sig, q)
                            p = bs.price("put", S, K, T, r, sig, q)
                            lhs = c - p
                            rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
                            worst = max(worst, abs(lhs - rhs))
    check("parity holds to 1e-9 everywhere", worst < 1e-9, f"worst={worst:.2e}")


# ── Greeks vs finite differences ──────────────────────────────────────────

def test_delta_matches_the_numerical_derivative():
    h = 1e-5
    worst = 0.0
    for kind in ("call", "put"):
        for S in (80, 100, 130):
            for sig in (0.15, 0.45):
                num = ((bs.price(kind, S + h, 100, 0.75, 0.04, sig)
                        - bs.price(kind, S - h, 100, 0.75, 0.04, sig)) / (2 * h))
                ana = bs.greeks(kind, S, 100, 0.75, 0.04, sig).delta
                worst = max(worst, abs(num - ana))
    check("delta == dPrice/dS", worst < 1e-5, f"worst={worst:.2e}")


def test_gamma_matches_the_second_derivative():
    h = 1e-3
    worst = 0.0
    for S in (85, 100, 120):
        for sig in (0.2, 0.5):
            num = ((bs.price("call", S + h, 100, 0.75, 0.04, sig)
                    - 2 * bs.price("call", S, 100, 0.75, 0.04, sig)
                    + bs.price("call", S - h, 100, 0.75, 0.04, sig)) / (h * h))
            ana = bs.greeks("call", S, 100, 0.75, 0.04, sig).gamma
            worst = max(worst, abs(num - ana))
    check("gamma == d2Price/dS2", worst < 1e-4, f"worst={worst:.2e}")


def test_vega_is_per_volatility_point_not_per_unit():
    """The 100x error that still looks like a plausible number."""
    h = 1e-6
    num_per_unit = ((bs.price("call", 100, 100, 1.0, 0.05, 0.20 + h)
                     - bs.price("call", 100, 100, 1.0, 0.05, 0.20 - h)) / (2 * h))
    ana = bs.greeks("call", 100, 100, 1.0, 0.05, 0.20).vega
    check("vega is the per-unit derivative / 100",
          abs(ana - num_per_unit / 100.0) < 1e-6, f"{ana} vs {num_per_unit/100:.6f}")


def test_theta_is_per_calendar_day_not_per_year():
    """The 365x error, same family."""
    h = 1e-6
    num_per_year = -((bs.price("call", 100, 100, 1.0 + h, 0.05, 0.20)
                      - bs.price("call", 100, 100, 1.0 - h, 0.05, 0.20)) / (2 * h))
    ana = bs.greeks("call", 100, 100, 1.0, 0.05, 0.20).theta
    check("theta is the per-year derivative / 365",
          abs(ana - num_per_year / 365.0) < 1e-6, f"{ana} vs {num_per_year/365:.8f}")


def test_long_option_greek_signs():
    for kind in ("call", "put"):
        g = bs.greeks(kind, 100, 100, 0.5, 0.03, 0.3)
        check(f"{kind} gamma > 0", g.gamma > 0)
        check(f"{kind} vega > 0", g.vega > 0)
        check(f"{kind} theta < 0 (long options decay)", g.theta < 0, g.theta)
    check("call delta in (0,1)", 0 < bs.greeks("call", 100, 100, .5, .03, .3).delta < 1)
    check("put delta in (-1,0)", -1 < bs.greeks("put", 100, 100, .5, .03, .3).delta < 0)


# ── Implied volatility ────────────────────────────────────────────────────

def test_implied_vol_round_trips():
    worst = 0.0
    for kind in ("call", "put"):
        for K in (80, 100, 125):
            for sig in (0.08, 0.25, 0.6, 1.2):
                px = bs.price(kind, 100, K, 0.6, 0.04, sig)
                got = bs.implied_volatility(kind, px, 100, K, 0.6, 0.04)
                check(f"{kind} K={K} sig={sig} solves", got is not None)
                if got is not None:
                    worst = max(worst, abs(got - sig))
    check("implied vol recovers input to 1e-6", worst < 1e-6, f"worst={worst:.2e}")


def test_impossible_prices_return_none_not_a_clamped_number():
    """A quote below intrinsic or above the no-arbitrage bound has NO implied
    volatility. Returning a boundary value would turn bad data into a
    confident 500% vol that the rest of the system would price off."""
    check("below intrinsic -> None",
          bs.implied_volatility("call", 0.01, 100, 50, 1.0, 0.05) is None)
    check("above upper bound -> None",
          bs.implied_volatility("call", 101.0, 100, 50, 1.0, 0.05) is None)
    check("zero price -> None",
          bs.implied_volatility("call", 0.0, 100, 100, 1.0, 0.05) is None)
    check("negative price -> None",
          bs.implied_volatility("call", -5.0, 100, 100, 1.0, 0.05) is None)


def test_expiry_and_zero_vol_collapse_to_intrinsic():
    check("at expiry a call is intrinsic",
          abs(bs.price("call", 120, 100, 0.0, 0.05, 0.3) - 20.0) < 1e-9)
    check("at expiry an OTM put is worthless",
          abs(bs.price("put", 120, 100, 0.0, 0.05, 0.3)) < 1e-9)
    check("zero vol call is discounted intrinsic",
          bs.price("call", 120, 100, 1.0, 0.0, 0.0) > 19.99)


# ── Structures ────────────────────────────────────────────────────────────

def test_unbounded_loss_is_refused_at_construction():
    """A naked short call has no ceiling. It is rejected from the leg set
    alone, before any pricing happens."""
    legs = [st.Leg(st.CALL, 100, -1)]
    check("naked short call is inadmissible", not st.is_admissible(legs))
    check("and the reason names the mechanism",
          "without limit" in (st.unbounded_reason(legs) or ""))
    try:
        st._mk("naked", st.DIRECTIONAL, "BEARISH", legs, 0.25, "x")
        check("building it raises", False)
    except st.UnboundedLoss:
        check("building it raises", True)


def test_every_named_structure_is_admissible():
    args = {
        "long_call": (100,), "long_put": (100,),
        "bull_call_spread": (100, 110), "bear_put_spread": (100, 90),
        "bull_put_spread": (95, 90), "bear_call_spread": (105, 110),
        "long_straddle": (100,), "long_strangle": (95, 105),
        "iron_condor": (85, 95, 105, 115), "iron_butterfly": (90, 100, 110),
        "call_butterfly": (95, 100, 105), "put_butterfly": (95, 100, 105),
    }
    check("args cover the whole catalogue", set(args) == set(st.STRUCTURES))
    for name, a in args.items():
        s = st.build_structure(name, *a, expiry_years=0.25)
        check(f"{name} builds and is bounded", st.is_admissible(s.legs))


def test_unknown_structure_raises_rather_than_defaulting():
    try:
        st.build_structure("iron_condr", 85, 95, 105, 115, expiry_years=0.25)
        check("typo raises", False)
    except KeyError:
        check("typo raises", True)


# ── Payoff economics ──────────────────────────────────────────────────────

def test_vertical_spread_economics_are_internally_consistent():
    s = st.build_structure("bull_call_spread", 100, 110, expiry_years=0.25)
    e = evaluate(s, S=100, r=0.05, sigma=0.30)
    width = (110 - 100) * 100
    check("debit, not credit", e["direction_of_cash"] == "DEBIT")
    check("max loss is the debit paid",
          abs(e["max_loss"] + e["net_cost"]) < 0.01, f"{e['max_loss']} {e['net_cost']}")
    check("profit + loss = width of the spread",
          abs(e["max_profit"] - e["max_loss"] - width) < 0.01,
          f"{e['max_profit']} {e['max_loss']} {width}")
    check("exactly one breakeven", len(e["breakevens"]) == 1, e["breakevens"])
    check("breakeven = long strike + debit per unit",
          abs(e["breakevens"][0] - (100 + e["net_cost_per_unit"])) < 1e-6,
          f'{e["breakevens"][0]} vs {100 + e["net_cost_per_unit"]}')
    check("profit is capped", e["max_profit_unbounded"] is False)


def test_long_call_profit_is_unbounded_and_says_so():
    s = st.build_structure("long_call", 100, expiry_years=0.5)
    e = evaluate(s, S=100, r=0.04, sigma=0.35)
    check("unbounded flag set", e["max_profit_unbounded"] is True)
    check("max_profit is None, not a large number", e["max_profit"] is None)
    check("max loss is the premium", e["max_loss"] < 0)


def test_straddle_has_two_breakevens_and_a_loss_zone():
    s = st.build_structure("long_straddle", 100, expiry_years=0.25)
    e = evaluate(s, S=100, r=0.04, sigma=0.35)
    check("two breakevens", len(e["breakevens"]) == 2, e["breakevens"])
    lo, hi = e["breakevens"]
    check("they straddle the strike", lo < 100 < hi, e["breakevens"])
    check("worst case is at the strike",
          abs(payoff_at(s, 100) - 0.0) < 1e-9)
    check("two profitable regions", len(e["profitable_regions"]) == 2)


def test_iron_condor_is_a_credit_with_a_finite_floor():
    s = st.build_structure("iron_condor", 85, 95, 105, 115, expiry_years=0.25)
    e = evaluate(s, S=100, r=0.04, sigma=0.30)
    check("credit received", e["direction_of_cash"] == "CREDIT", e["net_cost"])
    check("loss is finite", e["max_loss"] is not None and e["max_loss"] > -10_000)
    check("profit is capped", e["max_profit_unbounded"] is False)
    check("two breakevens", len(e["breakevens"]) == 2, e["breakevens"])
    check("max loss = wing width - credit",
          abs(abs(e["max_loss"]) - ((95 - 85) * 100 - abs(e["net_cost"]))) < 0.01,
          f"{e['max_loss']} {e['net_cost']}")


def test_probability_of_profit_is_bounded_and_labelled_risk_neutral():
    for name, a in (("long_call", (100,)), ("iron_condor", (85, 95, 105, 115)),
                    ("long_straddle", (100,)), ("bull_put_spread", (95, 90))):
        s = st.build_structure(name, *a, expiry_years=0.25)
        e = evaluate(s, S=100, r=0.04, sigma=0.30)
        pop = e["probability_of_profit_risk_neutral"]
        check(f"{name} POP in [0,1]", 0.0 <= pop <= 1.0, pop)
        check(f"{name} names the measure",
              any("RISK-NEUTRAL" in c.upper() for c in e["caveats"]))


def test_high_probability_credit_spread_has_a_worse_payoff_than_a_debit_one():
    """The structural trade-off that makes POP alone a misleading number: the
    high-POP structure is the one with the bad risk/reward, by construction."""
    credit = evaluate(st.build_structure("bull_put_spread", 90, 85, expiry_years=0.25),
                      S=100, r=0.04, sigma=0.30)
    debit = evaluate(st.build_structure("bull_call_spread", 100, 110, expiry_years=0.25),
                     S=100, r=0.04, sigma=0.30)
    check("credit spread wins more often",
          credit["probability_of_profit_risk_neutral"]
          > debit["probability_of_profit_risk_neutral"])
    cr = credit["max_profit"] / abs(credit["max_loss"])
    dr = debit["max_profit"] / abs(debit["max_loss"])
    check("but pays far less when it does", cr < dr, f"{cr:.3f} vs {dr:.3f}")


def test_everything_is_labelled_model_priced():
    s = st.build_structure("bull_call_spread", 100, 110, expiry_years=0.25)
    e = evaluate(s, S=100, r=0.05, sigma=0.30)
    check("basis is MODEL", e["basis"] == bs.MODEL_BASIS, e["basis"])
    check("the realized-vs-implied caveat is attached",
          any("realized" in c.lower() and "implied" in c.lower() for c in e["caveats"]))


def test_contract_multiplier_scales_money_not_probability():
    s = st.build_structure("bull_call_spread", 100, 110, expiry_years=0.25)
    us = evaluate(s, S=100, r=0.05, sigma=0.30, contract_multiplier=100.0)
    per_unit = evaluate(s, S=100, r=0.05, sigma=0.30, contract_multiplier=1.0)
    check("cost scales",
          abs(us["net_cost"] - per_unit["net_cost_per_unit"] * 100) < 0.01,
          f'{us["net_cost"]} vs {per_unit["net_cost_per_unit"] * 100}')
    check("breakevens do NOT scale", us["breakevens"] == per_unit["breakevens"])
    check("probability does NOT scale",
          us["probability_of_profit_risk_neutral"]
          == per_unit["probability_of_profit_risk_neutral"])


def test_crypto_scale_inputs_price_without_blowing_up():
    """BTC at 5-figure spot with 60% vol — the regime the equity-shaped code
    was never exercised on."""
    s = st.build_structure("bull_call_spread", 85_000, 95_000, expiry_years=0.25)
    e = evaluate(s, S=81_000, r=0.04, sigma=0.60, contract_multiplier=1.0)
    check("prices", e["net_cost"] > 0)
    check("breakeven sits between the strikes",
          85_000 < e["breakevens"][0] < 95_000, e["breakevens"])
    check("POP is sane", 0.0 < e["probability_of_profit_risk_neutral"] < 0.5)


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
