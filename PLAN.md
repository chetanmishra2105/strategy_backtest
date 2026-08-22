# NSE Swing-Trading System — Full Plan

**Goal:** A repeatable, data-driven system to find NSE stocks with a high probability of a
meaningful move in a **7–30 day** swing horizon, and to *prove* the edge on historical data
before risking capital.

**Stack (decided):** `yfinance` for price data · Backtest engine **+** daily screener ·
delivered as a **Streamlit** dashboard · Python 3.11 in the existing `.venv`.

**Your three strategies** (from `Swing_Trading_Strategies_Indian_Market.docx`) are now first-class
citizens of this system. **Bull Trap Reversal is the priority**: it gets backtested first *and* a
dedicated UI page.

---

## 0. Read this first — two hard truths (so you don't lose money)

1. **No honest system guarantees 4% in 7–30 days.** Markets are probabilistic. Your own document
   agrees — it targets **20–25% CAGR**, not 4%-per-trade certainty. The edge comes from:
   - Trading **only setups with historically positive expectancy** (proven by backtest).
   - **Position sizing** so a single loss never costs more than ~1% of capital.
   - **Cutting losers fast, letting winners run** (reward ≥ 1.5× risk).
2. **"Hidden news that moves a stock before the public knows" = insider information = illegal**
   under SEBI's PIT regulations. We do **not** build that. We harvest **publicly filed** signals
   most retail traders ignore — bulk/block deals, delivery %, earnings/board-meeting calendars,
   corporate announcements, FII/DII flows. That is the *legal* information edge.

---

## 1. What gets built (system architecture)

```
backt_chtn/
├─ requirements.txt          # expanded dependency list
├─ app.py                    # Streamlit entry + multipage router
├─ pages/                    # Streamlit multipage tabs
│  ├─ 1_Bull_Trap_Reversal.py   # ⭐ DEDICATED PAGE (scanner + backtest + charts for Strategy 1)
│  ├─ 2_Screener.py             # composite screener across all strategies
│  ├─ 3_Backtest_Lab.py         # backtest any strategy / stock / index / interval
│  ├─ 4_Sector_Rotation.py      # RRG + relative-strength heatmap
│  └─ 5_Playbook.py             # renders playbook.md in-app
├─ nsewing/                  # the engine package
│  ├─ config.py              # universes (NIFTY 50/500), sector→stocks map, risk & cost settings
│  ├─ data.py                # yfinance fetch + on-disk parquet cache; symbol universe
│  ├─ indicators.py          # RSI(14), MACD(12,26,9), EMA/SMA, ATR, ADX, Bollinger, VolMA20, Stochastic
│  ├─ strategies.py          # ⭐ your 3 strategies as pluggable signal classes (see §4)
│  ├─ signals.py             # shared setup helpers (breakout, divergence, engulfing, climax)
│  ├─ screener.py            # rank the universe each day → scored candidate table per strategy
│  ├─ backtest.py            # vectorized backtester + metrics (CAGR, Sharpe, maxDD, win%, PF, expectancy)
│  ├─ risk.py                # ⭐ position sizing, VIX-scaled risk, drawdown triggers (from doc §5)
│  ├─ sectors.py             # sector indices; relative-strength heatmap + RRG rotation chart
│  ├─ fundamentals.py        # P/E, ROE, debt/equity, div yield, mcap, promoter holding, earnings date
│  └─ charts.py              # plotly candlesticks + indicator overlays + equity/drawdown curves
├─ data_cache/               # parquet OHLCV cache (gitignored)
└─ playbook.md               # written guide + your document's rules, also shown in-app
```

**Expanded `requirements.txt`:** `pandas numpy yfinance streamlit plotly ta scipy`
(optional: `nsepython` for delivery% & bulk-deals; `streamlit-aggrid` for nicer tables).

---

## 2. Data layer — *where the data comes from*

- `get_ohlcv(symbol, interval, period)` wraps **yfinance**: `.NS` for stocks (`RELIANCE.NS`),
  `^NSEI` / `^NSEBANK` for indices. Intervals: `1d`, `1wk`, `1h`. (Doc uses daily + occasional 4h;
  yfinance has no 4h — we use `1h` and resample to 4h where needed.)
- **Honest limitation:** yfinance gives *years* of daily/weekly but only ~**60–730 days** of hourly.
  → Daily = the backbone (matches your doc, which is daily-candle based); hourly = entry timing only.
- **Parquet cache** in `data_cache/`, keyed by symbol+interval; `refresh=True` forces an update.
- **India VIX** (`^INDIAVIX`) pulled too — drives the risk module (doc §5.2).

---

## 3. Stock selection / screener — *how high-probability picks are chosen*

The screener runs **each of your three strategies' scanner rules** (doc §6) across the universe and
returns a ranked table per strategy, layered with a shared quality gate:

| Layer | Check | Source |
|---|---|---|
| **Fundamental gate** | Div yield 1–3%, D/E < 1.5, mcap > ₹5,000 Cr, positive FCF, promoter 20–40% | Doc §2.2 / `fundamentals.py` |
| **Trend/RS context** | Price vs 50/200 EMA; stock RS vs NIFTY & vs its sector | `indicators.py` |
| **Strategy trigger** | The exact scanner conditions of the chosen strategy (§4 below) | `strategies.py` |
| **Volume confirmation** | Volume ratios vs 20-day MA per strategy (1.5× / 2.0× / 2.5× / 4.0×) | Doc §6 |
| **Event-risk flag** | Earnings inside horizon; India VIX regime | `fundamentals.py`, VIX |

Output columns: strategy, score, entry, hard SL, T1/T2/T3, R:R, volume ratio, RSI/MACD state,
earnings-in-N-days flag, VIX-adjusted suggested position size.

---

## 4. ⭐ Strategy library — your three strategies, coded from the document

Implemented in `strategies.py` as pluggable classes with a common interface
`generate_signals(df) -> entries/exits/stops/targets`, so the **same backtester and screener** run
all of them. Rules are transcribed **directly from your doc §6 scanner specs + §2/3/4 entry-exit**.

### Strategy 1 — BULL TRAP REVERSAL ⭐ (priority: backtested first + dedicated page)
- **Direction:** SHORT / downside mean-reversion. *(Doc §2.3 prints "LONG ENTRY", but SL is above the
  failed-breakout high and all targets are downside — mechanically a short. We implement the short and
  note the doc's label typo.)*
- **Scanner (doc §6.2):**
  - Price action: `Close[-1]>Open[-1]` and `High[-1]>High[-2]` (breakout candle); then
    `Close[0]<Close[-1]` and `Close[0]<Open[-1]` (reversal candle); `High[0]<=High[-1]` (rejection).
  - Volume: `Vol[-1] ≥ 1.5×VolMA20` (breakout) and `Vol[0] ≥ 2.0×VolMA20` (reversal).
  - Momentum: bearish RSI(14) divergence (`High[-1]>High[-2]` but `RSI[-1] ≤ RSI[-2]`); MACD < signal.
- **Entry:** 0.5–1% below the rejection candle close.
- **Hard SL:** highest point of the breakout attempt (above the failed resistance bar).
- **Targets:** T1 last swing low (50%), T2 previous support (25%), T3 accumulation zone (25%);
  trail SL to break-even after T1. Min R:R 1:2, target 1:3–1:4.
- **Hold:** 2–10 days (swing). *(Doc §7.1's "close all day trades" line contradicts the 2–10d
  backtest; we follow the swing hold. Flagged.)*

### Strategy 2 — ACCUMULATION BREAKOUT
- **Direction:** LONG breakout from tight low-volume consolidation.
- **Scanner (doc §6.3):** 4+ day range ≤ 1.5% with volume ≤ 0.8×VolMA20; then
  `Close[0] > max(High[-4:-1])`, `Vol[0] ≥ 2.5×VolMA20`, close in upper 20% of range; `RSI>50`, MACD>signal.
- **Entry:** 0.3% above breakout candle close. **SL:** low of the consolidation box.
- **Targets:** T1 mid-to-swing-high (40%), T2 previous resistance (35%), T3 major resistance/Fib (25%);
  trail to box mid after T1. Min R:R 1:2.5. **Hold:** 4–15 days.

### Strategy 3 — AGGRESSIVE VOLUME EXPANSION (Climax Reversal)
- **Direction:** LONG bounce off a selling climax.
- **Scanner (doc §6.4):** red climax `Vol[-1] ≥ 4×VolMA20`, `Low[-1] < 20MA[-2]−3%`; then green
  `Vol[0] ≥ 3×VolMA20`, `Close[0]>Close[-1]`; `RSI[-1]<30`, `RSI[0]>RSI[-1]`, Stochastic crossing above 20.
- **Entry:** next candle if volume stays elevated (0.3–0.5% above open). **SL:** low of reversal candle.
- **Targets:** T1 20-day MA (30%), T2 recent swing high (40%), T3 resistance (30%). Min R:R 1:3.5.
  **Hold:** 1–7 days.

*Each strategy ships with a preset in the Backtest Lab and appears in the Screener.*

---

## 5. Backtesting engine — *test any strategy/stock/index on daily/weekly/hourly, with charts*

- **Vectorized backtester**: feed it a strategy class + OHLCV; supports `1d`/`1wk`/`1h`.
- **Multi-target scaling exits** matching the doc: partial exits at T1/T2/T3 with the doc's %
  allocations, break-even trail after T1, and hard SL — plus a max-hold cap per strategy.
- **Realism guardrails:** per-trade cost model (brokerage + STT + **0.1% slippage per the doc**);
  **no look-ahead** (signals use data ≤ bar *t*, act at *t+1* open); one open position per symbol.
- **Metrics** (mirror the doc's report): total return, **CAGR**, Sharpe, Sortino, **max drawdown**,
  **win rate**, avg win / avg loss, **profit factor**, expectancy, best/worst trade, trade count,
  avg holding period.
- **Validation callout:** the doc claims Bull Trap 68% win / 25.3% CAGR / PF 2.34 on top-25 Nifty
  stocks, 2021–2024. Our first backtest **reproduces that setup** so you can see how close real,
  cost-adjusted, look-ahead-free results land vs the document's numbers (expect them to be lower —
  that's the honest test of the edge).
- **Charts:** equity curve **vs buy & hold**, drawdown curve, trade entry/exit markers on the
  candlestick, return-distribution histogram, monthly-returns heatmap.
- Runs on a **single symbol** or a **whole universe** (portfolio equity by aggregating trades).

---

## 6. Risk module (`risk.py`) — coded from your document §5

- **Position sizing:** `shares = (capital × risk%) / (entry − stop)` (doc §5.1 formula).
- **VIX-scaled risk (doc §5.2):** VIX<15 → 1% & favor breakouts; 15–20 → 2–3% all strategies;
  20–25 → 1.5% & avoid reversals; >25 → 0.5% & tight SL / sidelines. Auto-applied from `^INDIAVIX`.
- **Drawdown triggers (doc §5.3):** daily >1.5%, weekly >3%, monthly >8%, quarterly >15% → warnings
  surfaced in the UI and honored by the portfolio backtest.
- **Portfolio caps (doc §5.1):** max daily/weekly/monthly risk 1% / 3% / 8%.

---

## 7. UI — Streamlit multipage dashboard

- **⭐ Page 1 — Bull Trap Reversal (dedicated):** live scanner for the setup on your universe;
  per-stock annotated candlestick (breakout bar, reversal bar, SL & T1/T2/T3 lines, RSI-divergence &
  MACD panels); a one-click backtest of *this* strategy with the full metrics + equity/drawdown/trade
  charts; VIX-adjusted position-size calculator; the doc's exact rules shown alongside.
- **Page 2 — Screener:** run all strategies across NIFTY 50/500, ranked candidate table, fundamentals
  gate, earnings flag, per-row chart.
- **Page 3 — Backtest Lab:** pick strategy / symbol or universe / interval / date range / params;
  full stats + all charts; compare vs buy & hold and vs the doc's published numbers.
- **Page 4 — Sector Rotation:** RRG quadrant scatter + relative-strength heatmap + sector drill-down.
- **Page 5 — Playbook:** renders `playbook.md` in-app.
- **Sidebar (global):** capital, risk-per-trade %, current VIX regime, date range, data-refresh button.

---

## 8. The written Playbook (`playbook.md`, also Page 5)

- **8.1** The three strategies in plain language (concept, when each shines, exact rules) — from your doc.
- **8.2** Fundamentals & volume in practice: the doc's fundamental gate, and reading volume
  (breakout surge, reversal spike, climax, dry-up, delivery %, OBV).
- **8.3** Legal "information edge": NSE corporate announcements & earnings/board calendar, bulk/block
  deals, SAST/PIT insider filings, shareholding changes, FII/DII flows — and a blunt line on what
  crosses into illegal insider trading.
- **8.4** Bear / sideways playbook: Bull Trap Reversal (short) shines in overheated/false-breakout
  chop; Climax Reversal for capitulation bounces; range/support-resistance trading; rotate to
  defensives (FMCG/Pharma) and hedge with index futures when NIFTY < 200-DMA.
- **8.5** Risk & the "4% target" reality: doc §5 position sizing, VIX table, drawdown halts,
  geopolitical/event protocols, and why expectancy beats win-rate.
- **8.6** Daily routine, trade-journal template, monthly review, and Do's/Don'ts — from your doc §7–8.

---

## 9. Build order

1. Env + `data.py` (+ cache, + `^INDIAVIX`) → confirm fetch of RELIANCE / NIFTY daily+weekly+hourly.
2. `indicators.py` (RSI, MACD, VolMA20, ATR, Stochastic, ADX) + `signals.py` helpers.
3. `strategies.py` — **Bull Trap Reversal first**, then Accumulation Breakout, then Climax Reversal.
4. `backtest.py` + metrics + `charts.py` → **validate Bull Trap on top-25 Nifty, 2021–2024**, compare to doc.
5. `risk.py` (sizing, VIX scaling, drawdown triggers).
6. `screener.py` + `fundamentals.py`.
7. `sectors.py` (RRG + heatmap).
8. `app.py` + `pages/*` — **Bull Trap Reversal page first**, then the rest.
9. `playbook.md`.

---

## 10. Verification (end-to-end)

- **Data:** fetch `RELIANCE.NS` on `1d`/`1wk`/`1h`, `^NSEI`, `^INDIAVIX`; assert non-empty; cache written.
- **Bull Trap backtest (priority):** run on top-25 Nifty stocks, daily, 2021–2024, with 0.1% slippage;
  confirm metrics compute and print a **side-by-side vs the document's numbers** (68% win / 25.3% CAGR /
  PF 2.34 / -18.2% DD) — realistic gaps expected and explained.
- **No look-ahead check:** shifting signals one bar must not implausibly improve results.
- **Screener:** run all 3 strategies on NIFTY 50; ranked tables + earnings flags populate; charts render.
- **Sector rotation:** all sector tickers return data; RRG scatter + heatmap render.
- **App:** `streamlit run app.py` launches; the Bull Trap page + all other pages load without error.

---

## 11. Honesty notes & document discrepancies (we implement the correct logic)

- **Bull Trap is a SHORT**, despite the doc's "LONG ENTRY" label in §2.3 (SL above breakout high +
  downside targets prove it). Implemented as a short.
- **Swing, not intraday:** doc §7.1 says "close all day trades / no overnight," but the strategies'
  own backtests hold 2–15 days. We follow the multi-day swing hold.
- yfinance has **no native 4h** and only ~60–730d of hourly — daily is the backbone (which matches
  the doc's daily-candle design); hourly/4h is resampled and used only for entry timing.
- **Not** a live auto-trading bot; no broker order placement (a Kite/Upstox layer can be added later).
- **No** guaranteed returns; **no** insider/illegal data. The doc's published stats are a *claim* we
  will test, not a promise.

---
---

# ✅ WHAT IS BUILT (status: DONE) & HOW TO RUN

*This section was added after the build. Everything below is implemented, tested on live NSE data,
and working.*

## What the tool does now (in simple words)

A local web dashboard (Streamlit) with **5 pages**:

1. **⭐ Bull Trap Reversal** — your priority strategy on its own page. Scan for setups, backtest it,
   see an annotated chart, and get the exact number of shares to buy for your risk.
2. **🔍 Screener** — scans a list of stocks and ranks the best setups (score 0–100) for any of your
   3 strategies, with fundamentals + an "earnings coming soon" warning.
3. **🧪 Backtest Lab** — test any strategy on any stock or index, on daily/weekly/hourly data. Shows
   win rate, profit factor, CAGR, drawdown, equity curve vs buy-and-hold, and trade markers.
4. **🧭 Sector Rotation** — a rotation map (RRG) + heatmap showing which sectors are strong/weak, so
   you know where to focus. Click a sector to screen its stocks.
5. **📖 Playbook** — the written guide: how to pick stocks, read volume, the *legal* news sources to
   watch, best strategies for bear/sideways markets, and risk rules.

## The 3 strategies (from your Word document), all coded

| Strategy | Type | Status |
|---|---|---|
| **Bull Trap Reversal** | Short — fade a failed breakout | ✅ built + backtested first |
| **Accumulation Breakout** | Long — breakout from a quiet base | ✅ built |
| **Volume Climax Reversal** | Long — bounce off a selling climax | ✅ built |

Each has two modes (toggle in sidebar): **strict** (your doc's exact rules — gives almost no signals
on daily data) and **relaxed** (loosened to a tradeable number of signals; this is the default).

## Files created

```
backt_chtn/
├─ app.py                 # dashboard home — RUN THIS
├─ playbook.md            # the written trading guide
├─ requirements.txt       # dependencies (installed)
├─ pages/                 # the 5 dashboard pages
│  ├─ 1_Bull_Trap_Reversal.py
│  ├─ 2_Screener.py
│  ├─ 3_Backtest_Lab.py
│  ├─ 4_Sector_Rotation.py
│  └─ 5_Playbook.py
├─ nsewing/               # the engine
│  ├─ config.py           # stock lists, sectors, costs, risk settings
│  ├─ data.py             # downloads price data (yfinance) + caches it
│  ├─ indicators.py       # RSI, MACD, ATR, etc.
│  ├─ signals.py          # candlestick pattern helpers
│  ├─ strategies.py       # the 3 strategies
│  ├─ backtest.py         # the backtesting engine
│  ├─ risk.py             # position sizing + VIX + drawdown rules
│  ├─ screener.py         # ranks stocks
│  ├─ fundamentals.py     # P/E, debt, earnings date
│  ├─ sectors.py          # sector rotation (RRG + heatmap)
│  ├─ charts.py           # all the plotly charts
│  └─ ui.py               # shared dashboard controls
└─ data_cache/            # downloaded price data is stored here
```

## ▶️ HOW TO RUN

Open **PowerShell** in the project folder and run:

```powershell
cd C:\Users\mishrch\backt_chtn
.\.venv\Scripts\streamlit run app.py
```

Your browser opens at **http://localhost:8531**. Pick a page from the left sidebar
(start with **Bull Trap Reversal**). Set your capital and risk% in the sidebar.

To stop the app: press **Ctrl+C** in the PowerShell window.

**First run is slow** (it downloads price data); after that it uses the cache and is fast. Click
**"🔄 Refresh data cache"** in the sidebar to pull fresh data.

## ⚠️ Honest result from testing (important)

When the 3 strategies were backtested on **10 years of real NSE data** with trading costs, slippage,
and no look-ahead cheating, the results were **far below your document's claims**:

| Strategy | Doc claim (win% / profit factor) | Reality (win% / profit factor) |
|---|---|---|
| Bull Trap Reversal | 68% / 2.34 | ~25% / 0.39 |
| Accumulation Breakout | 64% / 2.18 | ~41% / **1.06** (≈break-even) |
| Volume Climax Reversal | 58% / 2.08 | ~31% / 0.62 |

**Takeaway:** the document's published numbers do not hold up under honest testing. Only
Accumulation Breakout is roughly break-even. Do **not** trade real money on these as-is — this is
exactly why we backtest first. Next steps to explore: test on intraday data via a broker API,
parameter-tune with walk-forward validation, or focus on the one strategy that's near break-even.

---
---

# ➕ UPDATE 2 — Williams %R strategy, Midcap/Smallcap universes, Performance page

*Added after the first build, per your request. All three items are implemented, tested on live
data, and working.*

## a) New strategy — Williams %R(280) Oversold (from your image)

The image was a Chartink-style filter: **buy when Williams %R(280) crosses below −90** (extremely
oversold), for large-cap stocks. This is now the **4th strategy** in the system.

- It runs on **one timeframe at a time** — the sidebar **Interval** dropdown (`1d` = Daily,
  `1wk` = Weekly) selects which. Pick the timeframe, the stocks get filtered by that timeframe's
  signal, and the backtest runs on it. (Exactly as you asked — no dual-timeframe mixing.)
- Long mean-reversion; stop below the recent low (capped at 2×ATR); targets at 2R / 3R / 4R;
  max hold 20 bars.
- `strict` mode = the image's literal WR(280)/−90; `relaxed` = WR(180)/−85 for more signals.

**How it performs (top-25 large caps, honest backtest with costs, no look-ahead):**

| Timeframe | Trades | Win rate | Profit factor | CAGR |
|---|---|---|---|---|
| **Daily** | 250 | ~28% | 0.62 | **≈ −4% / yr** |
| **Weekly** | 7 | ~29% | 0.86 | **≈ +1% / yr** |

*(Weekly has few trades because WR(280) needs ~280 weeks ≈ 5.4 years of history per signal.)*
Like the document's strategies, this is **not** a money-maker as-is on daily data — see it live in the
**Backtest Lab** by choosing "Williams %R Oversold" and switching the Interval.

## b) Added Midcap 50 + Smallcap 50 universes

Two new stock lists are now selectable in the sidebar **Universe** dropdown:

- **NIFTY Midcap 50** — 49/50 fetch cleanly.
- **NIFTY Smallcap 50** — 50/51 fetch cleanly.

Any strategy, screener, backtest, or the performance page can run on these just like the large-cap
lists — so you can compare how each strategy behaves across large / mid / small caps.

*Quick read (Accumulation Breakout, relaxed, full history): Midcap PF 0.78 / CAGR −7%,
Smallcap PF 0.99 / CAGR −1%. Smallcaps are the most break-even, but still not a proven edge.*

## c) New page — 📊 Strategy Performance (trailing 30 / 60 / 90 / 120 / 180 days)

A new dashboard page that runs **all 4 strategies** over the last **30, 60, 90, 120, and 180 days**
(windows are selectable) for whichever **Universe** you pick, and shows:

- A **heatmap** (strategy × window) for any metric — total return, annualised CAGR, win rate,
  profit factor, expectancy, or trade count.
- **Pivot tables** for return, win rate, trades, and profit factor side by side.
- A **trade-count view** so you can spot thin/noisy windows (a great number on 1 trade means nothing).

**Use it like this:** pick a universe → run → read the heatmap to see which strategy is working
*recently* and on *which cap segment*; then switch the universe (large/mid/small) and re-run to compare.

> Honest note: over short trailing windows all four strategies currently show mostly flat-to-negative
> total returns on large caps — consistent with the full-history finding. The page's value is telling
> you this *before* you commit capital, and letting you spot if/when a regime turns favourable.

## Files added/changed in this update
- `nsewing/strategies.py` — added `WilliamsROversold` (+ registered in `STRATEGIES`).
- `nsewing/indicators.py` — added `williams_r()`.
- `nsewing/config.py` — added `MIDCAP50`, `SMALLCAP50`, and both to `UNIVERSES`.
- `nsewing/backtest.py` — added `start_date` (trailing-window support).
- `nsewing/performance.py` — **new**: multi-window comparison engine.
- `nsewing/charts.py` — added `performance_heatmap()`.
- `pages/6_Strategy_Performance.py` — **new** dashboard page.

*(How to run is unchanged: `.\.venv\Scripts\streamlit run app.py` — the new page and strategy appear
automatically.)*

---
---

# 🛠️ UPDATE 3 — Fixed the same-day-exit bug + rebuilt Backtest Lab for real swing trading

*You spotted (from the trade log) that **every trade exited the same day**. That was a genuine bug.
Fixed, plus the Backtest Lab is rebuilt to show the full selection logic and real ₹ money.*

## The bug and the fix

**What was wrong:** the engine closed a trade the instant the day's **intraday low** touched the stop.
Stops sit ~3–5% from entry, but a single day's high–low range is often **5–8%** — so the *entry day
itself* dipped below the stop and killed the trade on normal noise. Max-hold of 7–15 bars made it worse.

**The fix (matches how you actually trade):**
- **Stops now trigger only on the daily CLOSE** beyond the stop — intraday wicks are ignored. No trade
  is ever stopped out on its entry bar.
- **Max hold is now up to 60 days** (slider), so trades get room to reach targets.
- Take-profit targets still fill intraday (realistic for a limit order).

**Proof it's fixed** (Accumulation Breakout, Top-25, daily):
| | Before | After (60-day hold) |
|---|---|---|
| Same-day exits (0 bars) | **every trade** | **0** |
| Median / avg hold | ~0–1 days | 14 / **35 days** |
| Profit factor | ~1.06 | **1.28** |
| CAGR | ~+1% | **+3.9%** |

Winners now run instead of being shaken out — so results improved, but they're still **honest** (this
did not manufacture an edge).

## The Backtest Lab is now a layered funnel (as you asked)

Choose your **timeframe** and **universe** in the sidebar, then the page shows selection **layer by layer**:

1. **Layer 1 — Universe & timeframe:** which stocks have enough data.
2. **Layer 2 — Signals:** which stocks fired the strategy.
3. **Layer 3 — Fundamental gate:** pass/fail (mcap > ₹5,000 Cr, D/E < 1.5) + ⚠ earnings-soon flag.
4. **Layer 4 — Sector rotation:** keeps only stocks whose **sector is in an uptrend** (Leading/Improving),
   and **shows the dropped ones** (Lagging/Weakening) with their quadrant — nothing hidden.
5. **Layer 5 — Final ranked candidates**, with a **"🔎 Why this stock?"** explainer that lists every
   entry condition (pass/fail, actual value vs threshold), the full trade plan (entry/stop/T1-T3/R:R/ATR),
   the sector quadrant, and the fundamentals.

## Real ₹ portfolio backtest + "how the money was divided"

The universe backtest now models **actual rupees**, not abstract R-multiples:
- Controls: **capital, risk % per trade, max concurrent positions (default 6), max hold (up to 60d).**
- **Position size = (capital × risk%) ÷ stop distance**, capped by available cash per open slot.
- **Rich trade log:** shares, ₹ invested, **days held**, ₹ P&L, % P&L, exit reason.
- **"💰 How the money was divided" panel:** pick any signal day and see which stocks fired, which were
  **TAKEN vs skipped and why** (no slot / low rank / not enough cash), and the cash left after each.
- **₹ equity curve** + drawdown. Verified: cash never goes negative; capital is never over-committed.

## Files (this update)
- **Edited:** `nsewing/backtest.py` (close-based stops via new shared `resolve_exit`; `max_hold_override`),
  `nsewing/charts.py` (`equity_curve_rupees`), `pages/3_Backtest_Lab.py` (full layered rebuild).
- **New:** `nsewing/portfolio.py` (₹ portfolio sim + allocation log), `nsewing/pipeline.py` (layered
  funnel), `nsewing/explain.py` ("why this stock?").

*Same run command. Open **Backtest Lab**, choose Portfolio mode, click "Build candidate funnel", then
"Run portfolio backtest".*

---
---

# 🛠️ UPDATE 4 — Fixed the "sector = None / everything trades" bug + index comparison chart

*On Midcap 50 (1d), every stock showed sector = None and quadrant = Unknown, so the sector gate let
**all** stocks through and the backtest showed a big loss. Fixed, plus an index-comparison chart added.*

## The sector bug — fixed

**Cause:** the sector→quadrant mapping only knew the hardcoded large-cap list in `config.SECTORS`.
Midcap/smallcap tickers weren't in it → sector `None` → quadrant `Unknown`, and the gate treated
`Unknown` as "pass", so nothing was filtered (everything traded → the −43% you saw).

**Fix:**
1. New `sectors.sector_for_symbol()` resolves a sector for **any** stock — first the large-cap map,
   then a **fallback to yfinance's own sector** (Technology→IT, Financial Services→Financials,
   Healthcare→Pharma, etc. via new `config.YF_SECTOR_MAP`).
2. The sector gate now **drops `Unknown` instead of passing it**, so the filter is actually meaningful.

**Result on Midcap 50:** every stock now gets a real sector + quadrant (MRF→Auto/Leading,
TATAELXSI→IT/Weakening…), **0 Unknowns**. The gate correctly keeps in-trend stocks and drops weak ones
(e.g. 22 kept / 19 dropped), so you no longer trade everything blindly.

## New: Strategy vs index comparison chart

After a portfolio backtest, the Backtest Lab now plots the **strategy's return against NIFTY 50,
NIFTY Midcap 150, and NIFTY Smallcap 250** (buy & hold) on one chart with distinct legends, plus a
total-return table. Verified tickers: `^NSEI`, `NIFTYMIDCAP150.NS`, `NIFTYSMLCAP250.NS`.

**Honest note:** with the sector gate now working, the midcap strategy turned from −43% to ~+46% over
full history — but the indices themselves returned far more (Smallcap 250 ~+289%) over the same window.
The comparison makes the benchmark explicit: **if the strategy trails buy & hold, the edge isn't there.**

## Files (this update)
- **Edited:** `nsewing/config.py` (`YF_SECTOR_MAP`, `COMPARE_INDICES`), `nsewing/sectors.py`
  (`sector_for_symbol`), `nsewing/pipeline.py` (use it + drop Unknown), `nsewing/performance.py`
  (`strategy_vs_indices`, `compare_returns_table`), `nsewing/charts.py` (`compare_curve`),
  `pages/3_Backtest_Lab.py` (comparison section).

---
---

# 🛠️ UPDATE 5 — Diagnosed the Midcap −48% + added a Reward:Risk filter

*You saw Midcap 50 return −48% (PF 0.75, avg hold 10 days). Investigated — it was **not** a code bug;
it's a **settings** effect. Fixed the guidance and added the R:R control you asked for.*

## Why Midcap 50 showed −48% (root cause)

The screenshot showed **Avg hold = 10 days**. The strategies aim for targets 2.5R–5R away, which need
**weeks** to reach. A **10-day max-hold** forces the position to be dumped at market on day 10 — usually
at a small loss — so you pile up time-stop losses and the curve bleeds. Proof (Midcap 50, same data):

| Max hold | Win rate | Profit factor | Return |
|---|---|---|---|
| **10 days** | 43% | **0.88** | **−24%** |
| 20 days | 40% | 1.06 | +14% |
| 60 days | 28% | 1.27 | **+48%** |

Same strategy, same stocks — only the hold cap changed. So the −48% was the **max-hold slider set too
low for swing trading**, not a bug. Fixes applied:
- The **Max hold slider now warns** when set below 30 days ("targets need weeks; a short cap turns
  winners into time-stop losses — try 45–60+").
- Help text on the slider explains this directly.

## New: Reward : Risk filter on the portfolio backtest (as you asked)

A **"Reward : Risk target"** dropdown now sits with the funnel controls: **Strategy default / 1:1 / 1:2 /
1:3 / 1:4 / 1:5**. When you pick a ratio, the profit target is placed at exactly that multiple of the
stop distance (single full-exit target), and it flows through the candidate levels **and** the portfolio
backtest. It shows the classic trade-off (Midcap 50, 60-day hold, sector gate on):

| R:R | Win rate | Profit factor | Return |
|---|---|---|---|
| 1:1 | 59% | 0.99 | −1% |
| 1:2 | 45% | 1.12 | +17% |
| 1:3 | 37% | 1.18 | +31% |
| 1:4 | 35% | 1.29 | **+52%** |

Higher R:R = fewer wins but bigger ones — and here the higher ratios are clearly better, which is a
genuinely useful thing to be able to test.

## How to reproduce a *positive* Midcap run
Backtest Lab → Portfolio mode → Midcap 50, 1d → Build funnel (sector gate ON) → set **Max hold 60**,
**Reward:Risk 1:3 or 1:4** → Run. (Still compare against the index chart — buy-and-hold is the bar.)

## Files (this update)
- **Edited:** `nsewing/strategies.py` (`rr_target` field + target override in `generate_signals`;
  `get_strategy(rr_target=…)`), `nsewing/pipeline.py` (`rr_target` param), `pages/3_Backtest_Lab.py`
  (R:R dropdown, short-max-hold warning, default hold 60).

---
---

# 🛠️ UPDATE 6 — Trade-log stop/target columns, entry-bar fix, and the honest verdict on −30%

*You asked why 1:2 still returns −30%, why trades hold only 0–4 days, and to add stop/target prices to
the log. Investigated all three; fixed the two that were bugs and am being straight about the third.*

## Added: stop & target price in the trade log ✅
The trade log now shows, per trade: entry price, **stop price**, **target price**, exit price, days held,
₹ invested, ₹ P&L, % P&L, and exit reason — so you can see the full plan vs the outcome at a glance.

## Fixed: bogus "0-day" trades ✅
Targets could fill on the **entry bar** itself (you buy at that bar's open, so acting on the same bar's
high is look-ahead). Now **neither stop nor target can fill on the entry bar** — both activate the next
bar. 0-day trades dropped from ~6 to ~0 (a lone one remains only as a weekend gap in the *calendar*-day
count, not a real same-day trade).

## Why trades still hold only a few days — NOT a bug (measured)
The stop sits ~**1.5 daily-ATRs** from entry and the target ~**3 ATRs**. A midcap that moves ~2.5% a day
simply *reaches* one of those levels within a few days — that's the strategy's geometry, not a
simulation error. The hold distribution is smooth (1,2,3,4,… days), and **stops slightly outnumber
targets**. A swing trade only lasts weeks if price doesn't hit either level first; with tight ATR stops,
it usually does.

## Why 1:2 still loses −30% — the real reason (measured, honest)
Two compounding facts, neither fixable by more tweaking:
1. **Realized reward:risk ≈ 1.4, not 2.0.** When a stop is hit, price has *closed through* it, so the
   exit is a bit worse than the planned stop (a realistic gap). Losers run ~−5% while the nominal risk
   was ~−3.75%. So a "1:2" plan behaves like ~1:1.4 after slippage-through-stop. *(The trade log now
   flags this explicitly.)*
2. **The strategy itself lacks a durable edge on this universe.** I prototyped an intraday hard-stop
   model too — it was **worse** (PF 0.78–0.86), because intraday stops reintroduce the shakeouts. No
   exit rule converts a break-even/negative edge into a positive one.

**Bottom line:** the −30% is the honest result of this strategy on midcaps, not a coding bug. To get a
positive curve you need a genuine edge — options: use a higher R:R (1:4 tested positive earlier because
its few big winners outweigh the overshoot), a different/《trending》 universe, longer holds with wider
stops, or a different strategy entirely. The tool is doing its most important job: **telling you the
truth before you risk money, instead of flattering the strategy.**

## Also fixed
- `indicators.adx()` no longer crashes on very short price series (guarded + try/except).

## Files (this update)
- **Edited:** `nsewing/backtest.py` (no entry-bar exits), `nsewing/portfolio.py` (stop/target in trade
  record), `nsewing/indicators.py` (ADX guard), `pages/3_Backtest_Lab.py` (trade-log columns + realized-
  R:R note).

---
---

# 🛠️ UPDATE 7 — Point-in-time sector gate (removed the look-ahead bias)

*You asked the right question: the backtest was trading the same 22 stocks that passed **today's**
fundamental + sector filters, applied across **all** history. That's look-ahead / survivorship bias.
Fixed the sector part properly and bounded the window for the fundamentals part.*

## What was biased
- The Layer 5 table shows each stock's **latest** signal (a snapshot for today's decision) — that part
  is fine.
- But the **backtest** re-ran the strategy and traded **every historical signal** of those stocks, while
  the selection filters were judged **as of today**. So it effectively asked "how would stocks that look
  good *now* have done in the past?" — flattering the filters.

## The fix (point-in-time sector gate)
- New `sectors.quadrant_history()` computes each sector's RRG quadrant as a **time series** (same
  RS-ratio/momentum math, classified at every date), and `sectors.quadrant_on(date)` reads the quadrant
  **as of a given date**.
- The portfolio sim now takes `sector_history` + `symbol_sectors` and, at **each historical signal**,
  only enters if the stock's sector was Leading/Improving **on that date**. Rejected entries are logged
  ("sector Pharma was Lagging on signal date") so it's transparent.
- The backtest universe is now the **fundamentals-only** survivors (`fund_pass`); the sector gate is
  applied per-date inside the sim rather than pre-filtered by today's rotation.

## Fundamentals: bounded to a recent window
yfinance only exposes **current** fundamentals (no historical point-in-time data), so those can't be made
truly point-in-time. Instead a **Backtest window** control (3 or 5 years, default 5) limits the test to
the recent span where today's fundamentals are a fair proxy; older history is skipped.

## Why this matters (measured on Midcap 50, 1:2, 60-day hold)
| | Trades | Win rate | Profit factor | Return |
|---|---|---|---|---|
| OLD (today's gate on all history) | 230 | 45.2% | 1.09 | **+15.9%** |
| **NEW (point-in-time, 5-yr)** | 124 | 37.9% | 0.81 | **−20.6%** |

The old +15.9% was **inflated by look-ahead** — 113 entries were correctly rejected once the sector was
checked as-of-date (it was Lagging/Weakening back then). The honest, unbiased result is **−20.6%**. This
is the whole point: **an unbiased backtest tells you the truth**, and it confirms this strategy doesn't
have a durable edge here as-is.

## Files (this update)
- **Edited:** `nsewing/sectors.py` (`quadrant_history`, `quadrant_on`, `_classify`), `nsewing/portfolio.py`
  (point-in-time sector gate params + per-date check + skip logging), `nsewing/pipeline.py` (return
  `fund_pass` + `sym2sec`; score all fundamentally-sound names), `pages/3_Backtest_Lab.py` (Backtest-window
  control, pass PIT sector data, use fundamentals-only universe).

---
---

# 🛠️ UPDATE 8 — % Stop-loss / Target exit method (fixes the 1–2 day holds)

*You saw many trades close in 1–2 days — too short for swing. Root cause: the strategies' ATR-based
stops sit ~1.5 ATR from entry, so daily noise hits them fast. Fix: a **percentage-based exit bracket**
with wider stops, which is now the default.*

## What changed
- **New default exit method: "% Stop-loss / Target"** (a checkbox, ON by default). It **disables the
  Reward:Risk option** while checked (as you asked).
- **Stop-loss %** dropdown: 5% / 8% / 10% / 12% / 15% — the stop is placed that % below entry
  (above, for shorts), computed from each trade's entry price.
- **Target = SL ×** dropdown: 1× / 1.5× / 2× / 3× — the target is that multiple of the stop %.
  Default **2×**, exactly as requested (e.g. 10% stop → 20% target).
- Uncheck the box to fall back to the Reward:Risk selector; unchecking disables the % dropdowns.
- Flows through the whole funnel: candidate levels **and** the point-in-time portfolio backtest use it.

## It fixes the short-hold problem (Midcap 50, 2-yr, point-in-time)
| Exit method | Median hold | Trades ≤ 2 days | Return |
|---|---|---|---|
| Strategy ATR stop (old) | 16 days | 10% | −26% |
| 5% SL / 10% target | 23 days | 4% | −26% |
| **10% SL / 20% target** | **50 days** | **2%** | **−17%** |

Wider % stops let winners breathe — median hold rises to ~50 days (real swing behaviour), the 1–2 day
shakeouts nearly vanish, and both win rate and return improve. *(Still negative here — the strategy's
edge problem remains — but the trade mechanics now match how you actually trade.)*

## How to use
Backtest Lab → Portfolio mode → keep **"Use % Stop-loss / Target"** checked → pick **Stop-loss %** and
**Target = SL ×** → Build funnel → set Max hold 60 & window → Run.

## Files (this update)
- **Edited:** `nsewing/strategies.py` (`pct_stop` / `pct_target_mult` fields + override in
  `generate_signals`; `get_strategy(...)` params), `nsewing/pipeline.py` (pass-through), 
  `pages/3_Backtest_Lab.py` (exit-method checkbox + SL%/target-multiple dropdowns, default on).

---
---

# 🛠️ UPDATE 9 — Long-only + fundamental QoQ-growth filter & ranking (no technicals)

*You asked to (1) focus only on **long** positions, and (2) filter/rank stocks purely on
**quarter-on-quarter fundamentals** — operating profit, EPS, revenue increasing — with weights, and
**no technical indicators** in that scoring.*

## Long-only
Short-side signals are now ignored everywhere in the funnel (`long_only=True`, default). Bull Trap
Reversal is a short strategy, so under long-only it correctly produces **no trades**; the long
strategies (Accumulation Breakout, Volume Climax, Williams %R) are unaffected. The Backtest Lab shows a
"📈 Long-only" note.

## Fundamental QoQ-growth score (purely fundamental — no MACD/RSI/etc.)
New `fundamentals.growth_score()` reads yfinance's **quarterly income statement** (~5 quarters) and scores
quarter-on-quarter growth of three metrics, weighted as you specified:
- **Operating Income 40% · EPS 35% · Revenue 25%.**
- Each metric is scored on **consistency + latest quarter**: 60% "how many of the last 4 QoQ steps rose"
  + 40% "is the latest quarter up" — so steady growers beat one-off spikes.
- Combined into a **0–100 growth score**.

**How it's used (hard filter + ranking):**
- A stock must score **≥ 50** (more growth than not, weighted) to be tradeable — weak names are dropped
  with reason "weak QoQ growth (NN)".
- Candidates are **ranked by this growth score**, not by any technical composite. The ranking contains
  no MACD/RSI/volume — exactly as requested.

## What it looks like (Midcap 50, long-only, growth on)
- **DALBHARAT** dropped — growth 21 (OpInc −17%, EPS −52%, revenue up 1/4). **CUMMINSIND** dropped —
  "weak QoQ growth (49.7)".
- **Ranked top:** TORNTPHARM 88.8, AUROPHARMA 85, ASTRAL 85, ZYDUSLIFE 82.7 — sorted purely on QoQ
  operating-profit + EPS + revenue growth.
- Layer 3 table now shows `growth_score`, `OpInc_QoQ`, `EPS_QoQ`, `Rev_up_qtrs` per stock.

*Note: yfinance only exposes ~5 quarters, so the score uses the last ~4 QoQ changes. The strategy's own
entry rules (which are technical) still decide the entry timing and appear in "Why this stock?"; the
**selection and ranking** are now purely fundamental, which is what you asked for.*

## Files (this update)
- **Edited:** `nsewing/fundamentals.py` (`growth_score` + QoQ helpers), `nsewing/pipeline.py`
  (`long_only` + `apply_growth`; growth gate in Layer 3; rank by growth; return `growth_map`),
  `pages/3_Backtest_Lab.py` (long-only note, "QoQ growth filter + rank" checkbox, growth columns in
  Layer 3).

---
---

# 🛠️ UPDATE 10 — Trailing stop-loss (₹ or %, activates once the trade goes green)

*You asked for a trailing stop/target that follows price up "once the stock starts giving a green
signal", by ₹1/₹2 steps. Implemented as a **trailing stop-loss** (stop-only, the standard well-behaved
form) with a choice of ₹ or %.*

## How it works
- **Activation:** the trail stays dormant until the trade is **in profit by the trail distance** (price
  ≥ entry + distance). That's the "once it goes green" trigger.
- **Ratchet:** it then follows the running **peak** up, keeping the stop `distance` below it, and **only
  ever moves up, never down**. It exits on a **daily close** below the trailed stop (same close-based
  rule as the fixed stop — no intraday-wick shakeouts).
- **Unit — your choice per run:** **₹ (1/2/5/10/20/50)** or **% (1/2/3/5%)**. The fixed profit target
  still applies; whichever (target or trailed stop) hits first wins.
- **Caveat surfaced in the UI:** a fixed ₹ trail is huge for a ₹12 stock and tiny for MRF (₹125k) —
  the % option scales fairly across price ranges.

## Verified
- **Ratchet unit test:** entry 100 → peak 120 with a ₹5 trail → stop ratchets to 115 → exits at the
  next close below 115, locking **+12%** (vs +1% with no trail). Stop never loosened.
- **No-trail regression:** with trailing off, results are **identical** to before (same trades/PF).
- **₹ vs %:** both work; the 3% trail on Top-25 shifted PF from 1.69→1.24 — showing the honest
  trade-off (a tight trail locks gains sooner but caps the big winners).
- App boots headless (health 200, zero errors).

## Where it lives (one source of truth)
All exit logic is in `backtest.resolve_exit`, which **both** the single-symbol and portfolio engines
call — so trailing was implemented once there and both inherit it.

## How to use
Backtest Lab → Portfolio mode → **Trailing stop-loss** section → tick "Enable trailing stop" → pick
**₹ or %** and the value → Build funnel → Run. Trailed exits show as `trail_stop` in the trade log.

## Files (this update)
- **Edited:** `nsewing/backtest.py` (`resolve_exit` trailing logic + `_simulate_one`/`run_backtest`
  forwarding), `nsewing/strategies.py` (`trail_amount`/`trail_is_pct` fields + `get_strategy`),
  `nsewing/portfolio.py` (forward trail params), `nsewing/pipeline.py` (pass-through),
  `pages/3_Backtest_Lab.py` (trailing controls).

---
---

# 🛠️ UPDATE 11 — One analysis window governs everything (funnel ↔ simulation now consistent)

*You spotted that Layer 5 showed a **2022** signal (TORNTPHARM) while the simulation ran on the last 2
years — the funnel and the backtest disagreed. Fixed: a single window now bounds the whole flow, and
Layer 5 shows exactly what the sim trades.*

## The inconsistency
Layer 5 recorded each stock's **single most-recent** Williams signal regardless of date (so TORNTPHARM
showed 2022-05-24), but the simulation only traded the last N years. Two different views of "the trades."

## The fix — everything inside one window, point-in-time
- **One "Analysis window (years)" selector at the top** of the Backtest Lab governs **Layers 2–5 AND the
  simulation**. (The old per-sim window selector and the redundant "signal look-back bars" slider are gone.)
- **Layer 2** now keeps **every** long signal within the window (not just the latest).
- **Layer 4/5 are per-signal:** for **each** signal, the sector's RRG quadrant is checked **as of that
  signal's own date** (reusing `sectors.quadrant_history` / `quadrant_on` from UPDATE 7). Only
  Leading/Improving pass. Layer 5 lists **every surviving in-window signal** — a stock appears once per
  signal date (names repeat) — which is **exactly the list the simulation trades**.
- **Fundamentals** stay current-data as a window-wide proxy (yfinance has no historical PIT financials) —
  disclosed. Signal timing and sector rotation are genuinely point-in-time.

## Verified (Midcap 50, Williams %R, 2-yr window)
- **No pre-window rows:** Layer 5 min signal_date = 2024-08-29 (cutoff 2024-08-16). The 2022 TORNTPHARM
  row is gone. ✓
- **Per-signal & repeats:** 66 in-window signals across 19 symbols (e.g. CONCOR 10, IDFCFIRSTB 9). ✓
- **Point-in-time sector gate:** 75 signals dropped because their sector was Lagging/Weakening **on that
  date** (e.g. BALKRISIND 2025-04-07 dropped, but its 2026-06-10 signal kept). ✓
- **Consistency:** sim traded 30 of the 66 (reduced only by max-positions / cash / one-per-name) —
  a proper subset of Layer 5, all inside the window. ✓
- Changing 2→1 yr shrinks Layer 2/5 and the sim identically. App boots (health 200).

## Files (this update)
- **Edited:** `nsewing/pipeline.py` (`window_years`; all in-window signals; per-signal point-in-time
  sector gate; Layer 5 = per-signal rows), `pages/3_Backtest_Lab.py` (window selector moved to top,
  drives funnel + sim; removed look-back slider & duplicate window; Layer 5 caption).

---
---

# 🛠️ UPDATE 12 — Strategies are now configurable (only Williams %R enabled)

*Requested: make each strategy switchable on/off from config, with **Williams %R enabled and the others
disabled**.*

## What changed
- **`nsewing/config.py`** gets a `STRATEGY_ENABLED` map (this is the existing central config file — no
  duplicate created):
  ```python
  STRATEGY_ENABLED = {
      "Bull Trap Reversal":    False,
      "Accumulation Breakout": False,
      "Volume Climax Reversal":False,
      "Williams %R Oversold":  True,
  }
  ```
  Flip any value to `True`/`False` to show/hide that strategy across the whole app.
- **`strategies.enabled_strategies()`** returns only the enabled names (registry order). Unknown/missing
  keys default to enabled so a newly-added strategy is never silently hidden; never returns empty (safety).
- **UI + engines use it:** the Strategy dropdowns in **Backtest Lab** and **Screener**, and the default
  strategy set in **Strategy Performance**, now list only enabled strategies. `STRATEGIES` (the class
  registry) is unchanged, so `get_strategy(...)` still works for any name.

## Verified
- `config.STRATEGY_ENABLED` loads; `enabled_strategies()` → `['Williams %R Oversold']`.
- App boots headless (health 200, zero errors) — every page's dropdown now shows Williams %R only.

## Files (this update)
- **Edited:** `nsewing/config.py` (`STRATEGY_ENABLED`), `nsewing/strategies.py` (`enabled_strategies()`),
  `pages/2_Screener.py` + `pages/3_Backtest_Lab.py` (dropdowns use it), `nsewing/performance.py`
  (default set uses it).

---
---

# 🟢 UPDATE 13 — Supertrend swing strategies (Plans A / B / C) — Plan B implemented

*Requested: a **long-only** swing system built around the **Supertrend** indicator plus more
confirmations, targeting an **average ~3%/month**, that (a) enters where **volume is rising**,
(b) uses **sector rotation** to pick trending stocks, (c) has **SL / target / trailing SL**, and
(d) checks **fundamentals**. Three plans were designed; **Plan B is implemented** as a new,
fully-separate strategy that plugs into the existing funnel — only the strategy changes; every
filtering + backtest step is reused unchanged.*

## ⚠️ The honest "3% monthly" reality (read first)
3%/month compounds to **~42.6%/year**. That is achievable **on average across a full year in
trending phases** — it is **not** deliverable *every* month. Realistic shape: good months +6–12%,
choppy months −2 to +2%, correction months −4 to −8% if rules are followed. The goal is a
**positive-expectancy, high-discipline system that averages 3%+**, with strict risk control keeping
bad months small. Anyone promising a *guaranteed* monthly 3% is not being honest. The backtest tells
the truth — trade on what it shows, not on the target.

## The shared engine (all three plans use it — nothing here changes per plan)
Only the **strategy class** differs. Every plan reuses the existing funnel exactly as-is:
Layer 1 data → **Layer 2 strategy signals** → Layer 3 fundamentals + QoQ-growth gate/rank →
Layer 4 **point-in-time sector-rotation gate (RRG)** → Layer 5 ranked candidates → **₹ portfolio
backtest** with SL / target / **trailing-stop** controls. So "sector rotation", "fundamentals", and
"SL/target/trailing" are already handled by the pipeline — a plan only supplies the **entry logic +
level logic**.

### 🟢 Plan A — "Supertrend Trend-Rider" (simplest, most robust)
- **Entry:** daily Supertrend green **+** weekly Supertrend green **+** price > 50 EMA **+**
  volume > 1.5× avg **+** ADX > 20.
- **Exit:** trail on the daily Supertrend line; hard SL at Supertrend.
- **Character:** ~4–8 trades/month, holds 2–6 weeks. Fewest rules, least whipsaw; needs a trending
  tape to reach 3%. *(Design documented; not the one built.)*

### 🔵 Plan B — "Sector-Momentum Breakout" ⭐ (balanced — IMPLEMENTED)
Fits every requirement at once. Long-only. Entry requires **all** of:
| Filter | Rule |
|---|---|
| Trend core | Daily **Supertrend green** (ATR 10, mult 3) |
| Weekly filter | Weekly Supertrend green *(strict mode only; dropped in relaxed)* |
| Structure | Close > EMA20 **and** EMA20 > EMA50 |
| Momentum | RSI(14) between **50 and 70** (strict) / > 50 (relaxed) |
| Trending | ADX(14) > **25** (strict) / > 20 (relaxed) |
| **Volume rising** | Volume ≥ **1.5× VOLMA20** |
| **Accumulation** | **OBV rising** (OBV > OBV n bars ago) |
| Breakout | Close breaks above the recent N-day high |
- **Levels:** entry = close +0.3%; **stop = the Supertrend line** (capped at 1.5×ATR) — this IS the
  "trail on Supertrend"; targets 2R / 3R / 4R.
- **SL / Target / Trailing:** the strategy ships Supertrend-anchored stops + R-multiple targets; the
  Backtest Lab's existing **% SL/Target**, **Reward:Risk**, and **₹/% trailing-stop** controls all
  flow through automatically.
- **Sector rotation + fundamentals:** the shared Layer 3/4 gates (QoQ growth rank + point-in-time
  RRG "sector Leading/Improving") do this — no extra code.
- **Character:** ~8–12 signals/month pre-gate; the sector + growth gates trim to the best. This
  "rising stock, in a rising sector, breaking out on rising volume, in an uptrend" template is the
  one most NSE swing-competition winners run.

### 🟠 Plan C — "Momentum Pyramid" (aggressive)
- Plan B **plus** pyramiding (add a 2nd tranche only to winners on a fresh higher-high breakout),
  tighter Supertrend (10, 2) for earlier entries, RSI>60 breakouts, concentration in top-RS names.
- Highest upside in strong bull phases, higher drawdown in chop. **Use only after Plan B is proven.**
  *(Design documented; not the one built.)*

## What was built for Plan B (modular — other strategies untouched)
- **`nsewing/indicators.py`** — added `supertrend(df, period=10, multiplier=3)` (returns line +
  direction ±1) and `obv(close, volume)`. Wired new columns `SUPERT`, `ST_DIR`, `OBV` into
  `enrich()`, so any strategy can use them; existing strategies unaffected.
- **`nsewing/strategies.py`** — new `SupertrendSectorMomentum` class (long). It resamples daily→weekly
  inside `prepare()` to add a point-in-time **weekly Supertrend** column, then applies the Plan B
  entry conjunction. Registered as **"Supertrend Sector Momentum"** in `STRATEGIES`. `strict` requires
  the weekly filter + ADX>25 + RSI 50–70; `relaxed` drops the weekly filter and loosens ADX/RSI.
- **`nsewing/config.py`** — `STRATEGY_ENABLED` now enables **Supertrend Sector Momentum** and keeps
  **Williams %R** on; the other three stay off.

## How to run Plan B
Backtest Lab → pick **"Supertrend Sector Momentum"** → Portfolio mode → choose universe (Top-25 /
Midcap 50) & timeframe `1d` → keep the sector + growth gates on → Build funnel → set Max hold 45–60,
pick your exit bracket (or enable the trailing stop) → Run. Compare against the index buy-&-hold chart.
