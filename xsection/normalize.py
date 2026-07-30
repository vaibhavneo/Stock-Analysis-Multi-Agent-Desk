"""
Robust cross-sectional normalization (by ranking date).

Ordinary (mean/std) z-scores are unstable when a single outlier dominates a
cross-section of a dozen names — so this module uses OUTLIER-ROBUST transforms
and documents the exact formula for each:

  winsorize(x, p)      : clip to the [p, 1-p] empirical quantiles before scaling.
  percentile_rank(x)   : rank / (n-1) in [0,1]; fully outlier-immune (the mission's
                         preferred default for the cross-section).
  robust_z(x)          : (x - median) / (1.4826 * MAD); MAD = median(|x-median|).
                         1.4826 makes MAD a consistent std estimator for normals.
  sector_relative(x,s) : percentile_rank computed WITHIN each sector group, so a
                         name is scored against its peers, not the whole tape.

Higher-is-better orientation is applied per feature (e.g. valuation ratios and
risk are inverted so that a higher normalized value always means "more
attractive"), and every feature's direction is documented in factors.py.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def _finite(vals: List[Optional[float]]) -> List[float]:
    return [v for v in vals if v is not None and math.isfinite(v)]


def winsorize(values: List[Optional[float]], p: float = 0.05) -> List[Optional[float]]:
    """Clip the lowest and highest `k = floor(p*n)` finite values to the nearest
    retained value: clip range [xs[k], xs[n-1-k]]. Symmetric; on small samples
    where floor(p*n)==0 it is a no-op (too few points to trim), which is correct
    — you can't winsorize 5% of 12 names."""
    xs = sorted(_finite(values))
    n = len(xs)
    if n < 3:
        return list(values)
    k = int(p * n)
    if k < 1 or k >= n - k:
        return list(values)
    lo, hi = xs[k], xs[n - 1 - k]
    return [None if v is None or not math.isfinite(v) else min(max(v, lo), hi) for v in values]


def percentile_rank(values: List[Optional[float]]) -> List[Optional[float]]:
    """rank/(n-1) in [0,1] over the finite values; None stays None. Ties share
    the average rank. Outlier-immune."""
    idx = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len(idx) < 2:
        return [0.5 if i in idx else None for i in range(len(values))]
    order = sorted(idx, key=lambda i: values[i])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg / (len(order) - 1)
        i = j + 1
    return [ranks.get(i) for i in range(len(values))]


def robust_z(values: List[Optional[float]]) -> List[Optional[float]]:
    xs = _finite(values)
    if len(xs) < 3:
        return [0.0 if v is not None else None for v in values]
    med = sorted(xs)[len(xs) // 2]
    mad = sorted(abs(x - med) for x in xs)[len(xs) // 2]
    scale = 1.4826 * mad if mad > 0 else 1.0
    return [None if v is None or not math.isfinite(v) else (v - med) / scale for v in values]


def sector_relative_rank(values: List[Optional[float]], sectors: List[Optional[str]]) -> List[Optional[float]]:
    """percentile_rank computed WITHIN each sector group. Singleton sectors get
    0.5 (no peers to rank against)."""
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(sectors):
        groups.setdefault(s or "unknown", []).append(i)
    out: List[Optional[float]] = [None] * len(values)
    for _, idxs in groups.items():
        sub = percentile_rank([values[i] for i in idxs])
        for k, i in enumerate(idxs):
            out[i] = sub[k]
    return out


def normalize_feature(values: List[Optional[float]], sectors: List[Optional[str]],
                      higher_is_better: bool = True, winsor_pct: float = 0.05,
                      sector_relative: bool = False) -> List[Optional[float]]:
    """The pipeline used for every feature: winsorize -> (optional sector) percentile
    rank -> orient so higher = more attractive. Returns values in [0,1]."""
    w = winsorize(values, winsor_pct)
    pr = sector_relative_rank(w, sectors) if sector_relative else percentile_rank(w)
    if not higher_is_better:
        pr = [None if v is None else 1.0 - v for v in pr]
    return pr
