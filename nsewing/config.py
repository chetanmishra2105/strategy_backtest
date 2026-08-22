"""Central configuration: universes, sector maps, cost/risk settings.

All tickers use the yfinance convention (``.NS`` suffix for NSE cash stocks,
``^`` prefix for indices).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
CACHE_DIR = os.path.join(PROJECT_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Network / SSL
# --------------------------------------------------------------------------
# Many corporate networks intercept TLS (re-sign certs with an internal CA that
# curl_cffi's bundled trust store does not know), which makes yfinance fail with
# "unable to get local issuer certificate". Since this fetches only public,
# read-only market data on a machine whose traffic is already inspected by IT,
# we relax verification by default. Set env NSEWING_SSL_VERIFY=1 to re-enable,
# or point NSEWING_CA_BUNDLE at your corporate root-CA .pem for a secure fix.
SSL_VERIFY = os.environ.get("NSEWING_SSL_VERIFY", "0") == "1"
CA_BUNDLE = os.environ.get("NSEWING_CA_BUNDLE")  # optional path to a CA .pem
IMPERSONATE = "chrome"  # curl_cffi browser fingerprint for Yahoo

# --------------------------------------------------------------------------
# Indices
# --------------------------------------------------------------------------
INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "INDIA VIX": "^INDIAVIX",
}
VIX_TICKER = "^INDIAVIX"
BENCHMARK = "^NSEI"

# Broad-market index tickers that fetch reliably on yfinance (verified), used
# for the index-return comparison chart.
COMPARE_INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY Midcap 150": "NIFTYMIDCAP150.NS",
    "NIFTY Smallcap 250": "NIFTYSMLCAP250.NS",
}

# --------------------------------------------------------------------------
# Sector -> constituent stocks (yfinance tickers).
# Representative liquid names per NSE sector; extend freely.
# --------------------------------------------------------------------------
SECTORS: dict[str, list[str]] = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
             "AXISBANK.NS", "INDUSINDBK.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
             "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
               "APOLLOHOSP.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
             "TATACONSUM.NS"],
    "Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS",
              "VEDL.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS"],
    "Financials": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",
                   "SBILIFE.NS", "SHRIRAMFIN.NS"],
    "Infra/Cement": ["LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS"],
    "Consumer/Other": ["ASIANPAINT.NS", "TITAN.NS", "TRENT.NS", "BHARTIARTL.NS"],
}

# Flat NIFTY-50-ish universe (top liquid names).
NIFTY50 = sorted({t for stocks in SECTORS.values() for t in stocks})

# Map yfinance's GICS-style sector strings -> our RRG composite buckets, so
# ANY stock (incl. midcaps/smallcaps not in SECTORS) can be assigned a sector
# and get a rotation quadrant. Used as a fallback in the pipeline/explainer.
YF_SECTOR_MAP = {
    "Technology": "IT",
    "Communication Services": "IT",
    "Financial Services": "Financials",
    "Financial": "Financials",
    "Healthcare": "Pharma",
    "Consumer Cyclical": "Auto",
    "Consumer Defensive": "FMCG",
    "Basic Materials": "Metal",
    "Energy": "Energy",
    "Utilities": "Energy",
    "Industrials": "Infra/Cement",
    "Real Estate": "Infra/Cement",
}

# The doc's Bull Trap validation used "top 25 Nifty 50 stocks by liquidity".
TOP25_LIQUID = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "M&M.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS",
]

# --------------------------------------------------------------------------
# NIFTY Midcap 50 (representative constituents, yfinance tickers)
# --------------------------------------------------------------------------
MIDCAP50 = [
    "ASHOKLEY.NS", "AUROPHARMA.NS", "BALKRISIND.NS", "BHARATFORG.NS", "COFORGE.NS",
    "CONCOR.NS", "CUMMINSIND.NS", "DALBHARAT.NS", "DIXON.NS", "FEDERALBNK.NS",
    "GODREJPROP.NS", "GMRAIRPORT.NS", "HDFCAMC.NS", "IDFCFIRSTB.NS", "INDHOTEL.NS",
    "INDUSTOWER.NS", "JUBLFOOD.NS", "LUPIN.NS", "MRF.NS", "MPHASIS.NS",
    "MUTHOOTFIN.NS", "OBEROIRLTY.NS", "PAGEIND.NS", "PERSISTENT.NS", "PIIND.NS",
    "POLYCAB.NS", "PFC.NS", "RECLTD.NS", "SAIL.NS", "SUNTV.NS",
    "TATACOMM.NS", "TATAELXSI.NS", "TVSMOTOR.NS", "UPL.NS", "VOLTAS.NS",
    "ABCAPITAL.NS", "ASTRAL.NS", "AUBANK.NS", "BANDHANBNK.NS", "BIOCON.NS",
    "COLPAL.NS", "GUJGASLTD.NS", "LICHSGFIN.NS", "MFSL.NS", "NMDC.NS",
    "OFSS.NS", "PETRONET.NS", "SRF.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
]

# --------------------------------------------------------------------------
# NIFTY Smallcap 50 (representative constituents, yfinance tickers)
# --------------------------------------------------------------------------
SMALLCAP50 = [
    "AARTIIND.NS", "ABFRL.NS", "AMBER.NS", "ANGELONE.NS", "APLLTD.NS",
    "BSE.NS", "BSOFT.NS", "CDSL.NS", "CESC.NS", "CHAMBLFERT.NS",
    "CROMPTON.NS", "CYIENT.NS", "DELHIVERY.NS", "FSL.NS", "GNFC.NS",
    "GRAPHITE.NS", "HFCL.NS", "IEX.NS", "IIFL.NS", "IRB.NS",
    "JBCHEPHARM.NS", "KEI.NS", "KPITTECH.NS", "LAURUSLABS.NS", "MANAPPURAM.NS",
    "MCX.NS", "NBCC.NS", "NCC.NS", "NH.NS", "PNBHOUSING.NS",
    "RADICO.NS", "RBLBANK.NS", "REDINGTON.NS", "RITES.NS", "SONACOMS.NS",
    "SUNDRMFAST.NS", "SUPREMEIND.NS", "SWANENERGY.NS", "TANLA.NS", "TRIDENT.NS",
    "UJJIVANSFB.NS", "USHAMART.NS", "UTIAMC.NS", "VGUARD.NS", "WELCORP.NS",
    "ZENSARTECH.NS", "ATGL.NS", "CASTROLIND.NS", "FINCABLES.NS", "GSFC.NS",
    "SWSOLAR.NS",
]

# --------------------------------------------------------------------------
# Extra midcap/smallcap momentum names — combined with MIDCAP50 + SMALLCAP50
# to form the ~150-name universe. Liquid NSE names spread across sectors so the
# strategy has more concurrent signals to fill open slots (higher deployment).
# --------------------------------------------------------------------------
MIDSMALL_EXTRA = [
    "ABBOTINDIA.NS", "ACC.NS", "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKA.NS",
    "BALRAMCHIN.NS", "BATAINDIA.NS", "BEL.NS", "BHEL.NS", "CANFINHOME.NS",
    "CHOLAFIN.NS", "COROMANDEL.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DHANI.NS",
    "ESCORTS.NS", "EXIDEIND.NS", "GLENMARK.NS", "GODREJIND.NS", "GRANULES.NS",
    "GUJALKALI.NS", "HINDCOPPER.NS", "IDBI.NS", "IDEA.NS",
    "IGL.NS", "INDIACEM.NS", "INDIAMART.NS", "IPCALAB.NS", "JINDALSTEL.NS",
    "JKCEMENT.NS", "JSWENERGY.NS", "KAJARIACER.NS", "LALPATHLAB.NS",
    "LTTS.NS", "M&MFIN.NS", "MGL.NS", "MOTHERSON.NS", "NATIONALUM.NS",
    "NAVINFLUOR.NS", "OIL.NS", "PVRINOX.NS", "RAMCOCEM.NS",
    "SHREECEM.NS", "SUNDARMFIN.NS", "SYNGENE.NS", "TATACHEM.NS", "THERMAX.NS",
    "TIINDIA.NS", "TRENT.NS", "UBL.NS", "VBL.NS", "ZEEL.NS",
]

# ~150-name momentum universe (deduped union). Sorted for stable display.
MIDCAP150 = sorted(set(MIDCAP50 + SMALLCAP50 + MIDSMALL_EXTRA))

UNIVERSES = {
    "Top 25 Liquid": TOP25_LIQUID,
    "NIFTY 50 (sample)": NIFTY50,
    "NIFTY Midcap 50": MIDCAP50,
    "NIFTY Smallcap 50": SMALLCAP50,
    "Midcap/Smallcap 150": MIDCAP150,
}

# --------------------------------------------------------------------------
# Strategy enable/disable switches.
# Set a strategy to False to hide it from the whole app (Screener, Backtest
# Lab, Strategy Performance). Keys MUST match strategies.STRATEGIES names.
# Currently: only Williams %R is enabled; the rest are disabled.
# --------------------------------------------------------------------------
STRATEGY_ENABLED = {
    "Bull Trap Reversal": False,
    "Accumulation Breakout": False,
    "Volume Climax Reversal": False,
    "Williams %R Oversold": True,
    "Supertrend Sector Momentum": True,   # Plan B (UPDATE 13)
}

# --------------------------------------------------------------------------
# Trading costs (per the doc: 0.1% slippage; add typical NSE charges).
# Expressed as a fraction of trade value, applied on entry AND exit.
# --------------------------------------------------------------------------
SLIPPAGE_PCT = 0.001          # 0.1% per trade side (doc §2.5)
BROKERAGE_PCT = 0.0003        # ~0.03% discount-broker style
STT_PCT = 0.001               # securities transaction tax (approx, on sell)
COST_PER_SIDE = SLIPPAGE_PCT + BROKERAGE_PCT   # applied both sides
COST_SELL_EXTRA = STT_PCT     # extra on the exit side

# --------------------------------------------------------------------------
# Risk settings (doc §5)
# --------------------------------------------------------------------------
DEFAULT_CAPITAL = 1_000_000.0     # ₹10 lakh
DEFAULT_RISK_PCT = 0.02           # 2% per trade (doc "Full 2-3%")

# Universal position cap: never commit more than this fraction of total capital
# to a single stock at entry, regardless of what risk-based sizing suggests.
# This bounds single-name concentration (a hard rule the user set).
MAX_POSITION_PCT = 0.10           # 10% of capital per position, max

# VIX regime -> (label, risk_pct, note)
VIX_REGIMES = [
    (0, 15, "Complacency", 0.01, "Focus on breakouts"),
    (15, 20, "Normal", 0.025, "All strategies okay"),
    (20, 25, "High", 0.015, "Avoid reversals"),
    (25, 999, "Extreme", 0.005, "Sidelines or tight SL"),
]

# Drawdown triggers (doc §5.3) — fraction of capital.
DRAWDOWN_TRIGGERS = {
    "daily": 0.015,
    "weekly": 0.03,
    "monthly": 0.08,
    "quarterly": 0.15,
}

# Portfolio-level max risk caps (doc §5.1)
MAX_RISK = {"daily": 0.01, "weekly": 0.03, "monthly": 0.08}

# --------------------------------------------------------------------------
# Fundamental gate defaults (doc §2.2)
# --------------------------------------------------------------------------
FUND_GATE = {
    "div_yield_min": 0.01,
    "div_yield_max": 0.03,
    "max_debt_to_equity": 1.5,
    "min_market_cap": 5_000e7,     # ₹5,000 Cr in absolute rupees
    "promoter_min": 0.20,
    "promoter_max": 0.40,
}
