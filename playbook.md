# The Swing Trading Playbook (NSE)

*A working reference for this system. Read it once end-to-end, then keep it open while you trade.*

---

## 0. The honest frame (read first)

- **No system guarantees 4% in 7–30 days.** Your uploaded strategy doc targets **20–25% CAGR** — not
  4%-per-trade certainty. Edge comes from **positive expectancy + tight risk + cutting losers fast.**
- **The 4% headline is a per-trade *target*, not a promise.** A good swing system wins ~50–60% of the
  time with winners ~1.5–2× the losers. Compounded over many trades that is excellent; any single
  trade can lose.
- **This tool's job is to tell you the truth.** When we backtested your three strategies on 10 years
  of real NSE daily data — with trading costs, 0.1% slippage, and *no look-ahead* — the results came
  in **far below the doc's claimed numbers** (e.g. Bull Trap's claimed 68% win / 2.34 profit factor
  did not reproduce; realistic profit factors landed near or below 1.0). That gap is the single most
  valuable output here: **published backtest stats are usually optimistic; verify before you risk money.**

---

## 1. How high-probability picks are chosen (the screener logic)

The screener scores every stock 0–100 by blending:

| Factor | Weight | What it rewards |
|---|---|---|
| **Volume conviction** | 30 | Volume ≥ 1.5–3× the 20-day average on the signal bar |
| **Reward : risk (to T1)** | 25 | Setups with ≥ 2R to first target |
| **Momentum alignment** | 20 | RSI pointing the trade's way (>50 long / <50 short) |
| **Relative strength vs NIFTY** | 25 | Leaders for longs, laggards for shorts (55-day RS) |

On top of that sits a **fundamental gate** (below) and an **earnings-in-N-days flag** so you never get
blindsided by a results-day gap inside your holding window.

**Rule of thumb:** trade only the top of the ranked list, only when the trade aligns with the sector
rotation view, and only when position sizing keeps your risk ≤ 1% of capital.

---

## 2. Fundamentals & volume in practice

### Fundamentals that matter for a 7–30 day swing (fewer than for investing)
The point is **avoiding blow-ups and event risk**, not deep valuation. The doc's gate (§2.2):
- **Debt-to-Equity < 1.5** — survives volatility without bankruptcy scares.
- **Market cap > ₹5,000 Cr** — enough liquidity to enter/exit cleanly.
- **Dividend yield 1–3%, positive free cash flow, promoter holding 20–40%** — not a distressed shell.
- **Check the earnings date** — avoid holding through a results announcement unless that's the thesis.

### Reading volume (the real signal)
- **Breakout volume:** a genuine breakout comes on volume ≥ 1.5× the 20-day average. No volume = suspect.
- **Reversal / climax volume:** a failed breakout that reverses on ≥ 2× volume = sellers in control
  (the Bull Trap). A capitulation flush on ≥ 4× volume then a high-volume green bar = climax reversal.
- **Volume dry-up on pullbacks:** in an uptrend, shrinking volume on the dip = healthy; a surge on the
  dip = distribution.
- **Delivery %** (available via NSE bhavcopy): high delivery % = real buyers taking stock home, not
  intraday churn. A breakout with high delivery % is far more trustworthy.
- **OBV / cumulative volume:** confirms whether volume is net accumulating or distributing.

---

## 3. The legal "information edge" (what to watch instead of insider tips)

> **Trading on unpublished price-sensitive information is illegal under SEBI (PIT) Regulations.**
> Do not seek or act on "hidden news" that isn't public. There is no edge worth a ban and a fine.

The *legal* equivalent is publicly-filed information that most retail traders never bother to read:

- **NSE/BSE corporate announcements & board-meeting calendar** — buybacks, splits, capex, order wins.
- **Earnings / results calendar** — know exactly when each holding reports.
- **Bulk & block deals** (published daily by the exchanges) — see when big players enter/exit.
- **Insider / SAST / PIT filings** — promoters and insiders must disclose their *own* legal trades;
  clusters of insider buying are a public, legal signal.
- **Shareholding-pattern changes** (quarterly) — rising FII/DII/promoter stakes.
- **FII/DII daily flow data** — the tide behind index moves.

Fold these in as a **confirmation layer**: take the screener's top setup, then check there's no adverse
announcement and no earnings landmine inside your window.

---

## 4. Top strategies for BEAR / SIDEWAYS markets

Trending strategies stall when the market chops. These three earn their keep:

1. **Bull Trap Reversal (short)** — *your Strategy 1.* Overheated, headline-driven ranges produce
   frequent failed breakouts. Fade them: short the rejection of a false breakout on heavy volume, stop
   above the breakout high, targets back into the range. Shines exactly when everyone else is chasing.
2. **Mean-reversion / range trading** — define the box (support & resistance), buy support / sell
   resistance, ATR-based stop just outside the box. Sideways markets *pay you* to fade the edges.
3. **Climax reversal + rotate to defensives** — *your Strategy 3* for capitulation bounces, plus when
   NIFTY < 200-DMA, rotate exposure to **FMCG/Pharma** (defensive) and, if you trade derivatives, hedge
   with index-futures shorts or protective puts.

Each is a preset in the **Backtest Lab** — verify the edge on the exact stocks and period you care about
before committing capital.

---

## 5. Risk & the "4%" reality (the part that actually makes money)

- **Position size (doc §5.1):** `shares = (capital × risk%) / |entry − stop|`. Risk a fixed **≤1–2%**
  of capital per trade so no single loss hurts. (₹10L @ 2%, ₹15 stop distance → 1,333 shares.)
- **VIX-scaled risk (doc §5.2):** VIX < 15 → 1% & favor breakouts · 15–20 → 2–3% all strategies ·
  20–25 → 1.5% & avoid reversals · > 25 → 0.5% or sidelines. The app reads live India VIX and suggests.
- **Drawdown circuit-breakers (doc §5.3):** pause on daily > 1.5%, cut size on weekly > 3%, **halt** on
  monthly > 8%, full review on quarterly > 15%.
- **Why expectancy beats win-rate:** a 40%-win system with 2.5R winners makes money; a 70%-win system
  with 0.5R winners loses. Always think in **R-multiples**, not hit-rate.
- **Event protocol (doc §5.4):** flatten before major scheduled events (RBI, Budget, elections);
  don't trade Day 1 of a shock; re-enter small.

---

## 6. Daily routine, journal & discipline (doc §7–8)

**Pre-market:** check global cues → run the screener → shortlist top 5–10 → mark entry/stop/targets.
**In-session:** enter *only* when all conditions are met; manage stops; never average down.
**Post-market:** log every trade (entry, stop, target, exit, reason, lesson).

**Do:** hard stops, trade only full-criteria signals, journal everything, respect position sizing,
scale winners, rest after 3 losses in a row.
**Don't:** chase candles (FOMO), ignore volume, overtrade (2–4 quality trades/week), hold losers,
average down, trade on breaking news, ignore macro/geopolitical risk.

---

*Strategies and rules in this playbook are transcribed from
`Swing_Trading_Strategies_Indian_Market.docx`. Where the document's exact thresholds produced almost
no signals on daily data, the system exposes a **strict** (doc-literal) and a **relaxed** (tradeable)
mode — see the Backtest Lab. Nothing here is investment advice; verify everything and trade your own risk.*
