# Draft Portfolio Policy (Fee‑Aware Rule‑A I/J Tiering)

This is a **draft** policy for paper‑mode research only. It is derived from
walk‑forward (train→test) analysis on the canonical daily ledgers and includes
Kalshi maker/taker fees (Oct 2025 schedule).

## Objective
Beat the Kalshi mid **net of maker fees** while placing **fewer, higher‑quality**
trades than the “blind fade home favorites” baseline.

## Current best candidate (Phase‑1 pregame)
Trade **Rule A only** (Kalshi home favorite), but concentrate risk into the two
highest‑signal sub‑buckets:

- **A‑I (“sign flip”)**: `p_mid >= 0.50` and `p_primary <= p_mid - t` and `p_primary < 0.50` → buy **away**
- **A‑J (“no flip”)**: `p_mid >= 0.50` and `p_primary <= p_mid - t` and `p_primary >= 0.50` → buy **away**
  - Only trade A‑J when a confirmer also triggers A‑J at the same threshold (`primary=J` AND `confirmer=J`).

Recommended pairing:
- Primary model: **Grok**
- Confirmer: **market_proxy** (Odds API no‑vig baseline, time‑aligned)

Recommended sizing (tiered contracts):
- **A‑I:** 3 contracts
- **A‑J with confirmer:** 1 contract

This configuration is specifically designed to:
- Increase average edge per contract (fewer trades, lower fee drag),
- Preserve the strongest historical “fade home favorite” behavior, and
- Avoid the negative‑EV disagreement regimes (when Grok and market disagree).

## Fee model (important)
Kalshi fees are not linear and are rounded up to the nearest cent per order.
Backtests that ignore fees can overstate edge, especially for many small trades.

In this repo:
- Maker fee is modeled as `ceil_to_cent(0.0175 * contracts * P * (1-P))` dollars.
- Taker fee is modeled as `ceil_to_cent(0.07 * contracts * P * (1-P))` dollars.

## Command template (walk‑forward, fee‑aware)
All leagues, train→test by date, maker fees included:

`PYTHONPATH=. python chimera_v2c/tools/walkforward_rule_a_ij_tiered_policy.py --league all --start-date 2025-11-19 --end-date 2025-12-15 --primary grok --confirmer market_proxy --edge-thresholds 0.005 0.01 0.015 0.02 0.025 --threshold-select-mode max_net_pnl --units-i 3 --units-jj 1 --fee-mode maker`

Notes:
- `--threshold-select-mode max_net_pnl` picks the day’s threshold from prior days only.
- Use `--no-write` to print results only (no CSV).
- The tool reports a fee‑matched baseline: “always buy away when `p_mid >= 0.50`”.

## What not to do (for now)
- Do not trade Rule B or Rule D in production without fresh out‑of‑sample evidence (they have been negative/unstable in this dataset).
- Avoid “disagreement” trades (primary says fade but market says follow, or vice‑versa); these have been consistently negative in the current window.

## Appendix: older agreement‑stacking draft
The older draft (“agreement‑only stacking” across A/B/C/D) is still useful for
exploration, but it is **not fee‑aware** and is currently inferior to the
fee‑aware A‑I/A‑J tiered policy above for the goal of higher net PnL with fewer
trades. If you want to revisit it, use:

- `chimera_v2c/tools/walkforward_agreement_stacking_policy.py`
