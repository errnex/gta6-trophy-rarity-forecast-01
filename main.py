#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase (makefun program/v1).

Reads every ``<id>.in`` case in the input directory and writes one
``<id>.out`` forecast per case: five quantiles (0.05/0.25/0.5/0.75/0.95) of
each trophy's earn rate (percent of Exophase-tracked PS5 players holding it)
at the case's target. Standard library only, no network, deterministic.

Model
-----
Exophase does not update its counts continuously: it refreshes a game's page
on its own schedule (hours apart), and a read shows the rates as of the last
refresh. Two forces move a rate between reads:

* dilution -- a refresh adds a cohort of newly tracked players who have
  earned far fewer trophies than the players tracked so far, so most rates
  fall when the tracked base jumps;
* progression -- the players already tracked keep playing and earn more
  trophies, so a rate rises toward the share of players who ever earn it.

A rate therefore saturates, and its logit is the quantity whose line
extrapolates safely: per trophy, least squares of ``logit(rate)`` on
``x = ln(1 + hours since release day)`` over the case's distinct reads in
the last 24 hours (consecutive identical reads are one Exophase state),
each slope shrunk toward the cross-trophy median slope (a trophy's own
slope is noisy when its count is small), anchored on the trophy's last
observed rate and read at the target. Fitting earner counts and the
tracked base separately was tried first and rejected: it lets a
fast-rising trophy's forecast blow past 100%.

When the whole history is a single flat page (fewer than 3 distinct reads)
no trend can be estimated -- and the page is stale, a refresh is overdue.
The program then applies the refresh prior calibrated on the launch-night
refresh the kit documents (638 -> 3,868 players tracked, every rate but
the 100% one falling to about a fifth): the median rate is a fifth of the
last read, with a wide spread that also covers the no-refresh outcome.

The spread is the fit's residual (shrunk toward the cross-trophy median,
floored) combined with a random-walk term that grows with the horizon, so
a 4.5-day forecast is honestly wider than a 90-minute one. The one trophy
at 100% (every tracked player holds it, through a 6x base expansion) is
forecast to stay at ~100% with a tight band. Quantiles are normal on the
logit scale, rounded to 4 decimals, non-decreasing, in [0.01, 99.99].
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Flat-history refresh prior: median rate multiplier and base logit spread.
# Calibrated on the kit's documented launch-night refresh (rates fell to
# ~0.22x; the median 0.20 hedges slightly toward a smaller fall).
FLAT_MULT = 0.20
FLAT_SIGMA = 0.60
# Trend branch: random-walk variance per day added to the fit residual.
# 0.09 splits the clean-sim optimum (0.06) and the noisier-sim plateau
# (0.06-0.12 all within noise there): honest without being over-wide.
TREND_RW_VAR_PER_DAY = 0.09
SIGMA_MIN, SIGMA_MAX = 0.45, 1.10
# A trophy essentially nobody holds yet: floor its median, widen its band.
RARE_RATE_FLOOR = 0.05
RARE_SIGMA = 0.90
# The trophy everybody holds: forecast it to stay there.
FULL_ROW = [99.0, 99.9, 99.99, 99.99, 99.99]


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    return round(x, 4)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_ppf(p):
    """Acklam's approximation of the standard normal quantile."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def zscores(levels):
    if len(levels) == len(LEVELS) and all(a == b for a, b in zip(levels, LEVELS)):
        return Z
    return [norm_ppf(t) for t in levels]


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (intercept, slope, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def quantiles(med, sigma, zs):
    """Normal-on-logit quantiles around a median rate, non-decreasing."""
    l = logit(med)
    out = []
    prev = 0.0
    for z in zs:
        q = round4(clamp(inv_logit(l + z * sigma)))
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def distinct_states(history):
    """Collapse consecutive identical reads (one Exophase state each).

    Returns [(time, players_tracked, rates_tuple)] in time order.
    """
    states = []
    last_key = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != last_key:
            states.append((parse_time(h["at"]), h.get("players_tracked"), key[1]))
            last_key = key
    return states


def hours_since_release(t):
    h = (t - RELEASE).total_seconds() / 3600.0
    return max(h, -0.9)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    zs = zscores(levels)
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    try:
        checkpoint = parse_time(case["at"])
    except Exception:
        checkpoint = parse_time(history[-1]["at"]) if history else RELEASE
    try:
        target = parse_time(case["target"])
    except Exception:
        target = checkpoint
    x_tgt = math.log1p(hours_since_release(target))

    # Last non-null rate per trophy (the anchor).
    n_tr = len(trophies)
    last_rate = [None] * n_tr
    for h in reversed(history):
        rates = h.get("rates") or []
        for i in range(n_tr):
            if last_rate[i] is None and i < len(rates) and rates[i] is not None:
                last_rate[i] = rates[i]
        if all(v is not None for v in last_rate):
            break
    last_rate = [1.0 if v is None else v for v in last_rate]

    states = distinct_states(history)

    # ---- Flat history: no trend is estimable; a refresh is overdue. ----
    if len(states) < 3:
        sigma = min(1.2, math.sqrt(FLAT_SIGMA ** 2 + 0.25 * horizon_days))
        rows = []
        for t, r in zip(trophies, last_rate):
            if r >= 99.99:
                rows.append({"id": t["id"], "q": list(FULL_ROW)})
            elif r <= 0.0:
                rows.append({"id": t["id"], "q": quantiles(RARE_RATE_FLOOR, RARE_SIGMA, zs)})
            else:
                rows.append({"id": t["id"], "q": quantiles(FLAT_MULT * r, sigma, zs)})
        return {"trophies": rows}

    # ---- Trend branch: fit logit(rate) on log time. ----
    # A rate saturates toward the share of players who ever earn the
    # trophy, so its logit -- not the earner count -- is the quantity
    # whose log-time line extrapolates safely: fitting ln E and ln N
    # separately lets a fast-rising trophy's forecast blow past 100%.
    # Slopes are shrunk toward the cross-trophy median (a trophy's own
    # slope is noisy when its count is small), and each fit is anchored
    # on the trophy's last observed rate.
    window_start = checkpoint.timestamp() - 24 * 3600
    window = [s for s in states if s[0].timestamp() >= window_start]
    if len(window) < 3:
        window = states
    x_last = math.log1p(hours_since_release(states[-1][0]))
    dx = x_tgt - x_last

    # Per-trophy logit-rate fits.
    slopes, resids = [], []
    per_trophy = []
    for i in range(n_tr):
        pts = []
        for (t, n, rates) in window:
            if i < len(rates) and rates[i] is not None:
                pts.append((math.log1p(hours_since_release(t)), logit(rates[i])))
        if len(pts) >= 3:
            a, b, r = ols([p[0] for p in pts], [p[1] for p in pts])
            slopes.append(b)
            resids.append(r)
            per_trophy.append((b, r, len(pts)))
        else:
            per_trophy.append((None, None, len(pts)))
    b_med = median(slopes) if slopes else 0.0
    r_med = median(resids) if resids else 0.3
    r_med = min(max(r_med, 0.15), 0.8)

    rows = []
    for idx, t in enumerate(trophies):
        r = last_rate[idx]
        if r >= 99.99:
            rows.append({"id": t["id"], "q": list(FULL_ROW)})
            continue
        b, resid, npts = per_trophy[idx]
        if b is None:
            b_hat, s_hat = b_med, r_med
        else:
            w = npts / (npts + 3.0)
            b_hat = w * b + (1.0 - w) * b_med
            s_hat = (npts * min(max(resid, 0.15), 0.8) + 3.0 * r_med) / (npts + 3.0)
        med = inv_logit(logit(r) + b_hat * dx) if dx > 0 else r
        sigma = math.sqrt(s_hat ** 2 + TREND_RW_VAR_PER_DAY * horizon_days)
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        if r < RARE_RATE_FLOOR:
            med = max(med, RARE_RATE_FLOOR)
            sigma = max(sigma, RARE_SIGMA)
        rows.append({"id": t["id"], "q": quantiles(med, sigma, zs)})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        case_id = name[:-3]
        with open(os.path.join(output_dir, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
