"""
Donchian Channel Breakout Strategy - XLF (Financial Select Sector SPDR Fund)
=============================================================================
Author: Linghang Xue
Asset:  XLF

Strategy overview:
    Enter LONG when price breaks above the 10-day Donchian high with volume
    confirmation (volume >= 1.0x its 10-day average).
    Enter SHORT when price breaks below the 10-day Donchian low with the
    same volume filter.
    Exits: profit target at 2x ATR(14), stop-loss at 1.5x ATR(14),
           or time-based exit after 10 trading days.

Walk-forward design:
    Training window ~252 days, out-of-sample window ~63 days (one quarter),
    rolled forward across the full dataset.

Data retrieval:
    Uses ShinyBroker (sb) to fetch DAILY historical data from Interactive
    Brokers via the SMART exchange. Stops and targets are checked against
    daily high/low prices (standard practice for daily-bar strategies).
"""

import numpy as np
import pandas as pd
import shinybroker as sb
import json
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS - all tunables in one place
# ═══════════════════════════════════════════════════════════════════════════════
TICKER = "XLF"
DATA_DURATION = "3 Y"                # ShinyBroker durationStr
DONCHIAN_LOOKBACK = 10               # rolling window for channel (days)
VOLUME_MULT = 1.0                    # volume must exceed this x 10d avg
ATR_PERIOD = 14                      # ATR lookback for stops / targets
PROFIT_TARGET_ATR = 2.0              # take-profit = entry +/- 2x ATR
STOP_LOSS_ATR = 1.5                  # stop-loss   = entry -/+ 1.5x ATR
TIMEOUT_DAYS = 10                    # close position after n trading days
POSITION_SIZE = 100                  # shares per trade
INITIAL_CAPITAL = 1_000_000.0
RISK_FREE_RATE = 0.0375              # annualised, for Sharpe / Sortino
TRAIN_WINDOW = 252                   # ~1 year training
OOS_WINDOW = 63                      # ~1 quarter out-of-sample

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA RETRIEVAL - ShinyBroker (daily bars only)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"  Fetching {DATA_DURATION} of daily data for {TICKER} via ShinyBroker")
print("=" * 70)

# Define the XLF ETF contract
asset = sb.Contract({
    "symbol": TICKER,
    "secType": "STK",
    "exchange": "SMART",
    "currency": "USD"
})

# ── Fetch daily bars (breakout signals, entries, exits) ──────────────────────
historical_data_daily = sb.fetch_historical_data(
    asset,
    durationStr=DATA_DURATION,
    barSizeSetting="1 day"
)

# ShinyBroker sometimes returns a dict wrapper - unwrap if needed
if isinstance(historical_data_daily, dict):
    for key in ["hst_dta", "bars", "data"]:
        if key in historical_data_daily:
            historical_data_daily = historical_data_daily[key]
            break

# ── Prepare daily DataFrame ─────────────────────────────────────────────────
df = historical_data_daily.copy()

# Ensure timestamp index is datetime
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
elif not pd.api.types.is_datetime64_any_dtype(df.index):
    df.index = pd.to_datetime(df.index)

df.index.name = "date"

# Ensure numeric columns
# ShinyBroker returns: open, high, low, close, volume, wap, barCount
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only the columns we need
df = df[["open", "high", "low", "close", "volume"]].dropna()

# Compute number of years for annualisation
n_calendar_days = (df.index[-1] - df.index[0]).days
DATA_PERIOD_YEARS = max(n_calendar_days / 365.25, 1.0)

print(f"  Daily data: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} bars)")
print(f"  Effective period: {DATA_PERIOD_YEARS:.2f} years")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add Donchian channel, ATR, and volume-average columns to *data*.

    Donchian channel:
        upper = highest high over the past DONCHIAN_LOOKBACK days
        lower = lowest low   over the past DONCHIAN_LOOKBACK days
        Both shifted by 1 day to prevent look-ahead bias.

    ATR (Average True Range):
        Exponential moving average of true range over ATR_PERIOD days.

    Volume average:
        Simple DONCHIAN_LOOKBACK-day moving average of volume.
    """
    d = data.copy()

    # Donchian channel (shifted by 1 to avoid look-ahead)
    d["don_high"] = d["high"].rolling(DONCHIAN_LOOKBACK).max().shift(1)
    d["don_low"]  = d["low"].rolling(DONCHIAN_LOOKBACK).min().shift(1)

    # True Range & ATR
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"]  - d["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # Volume average
    d["vol_avg"] = d["volume"].rolling(DONCHIAN_LOOKBACK).mean()

    return d


df = compute_indicators(df)
df = df.dropna()  # first DONCHIAN_LOOKBACK rows have NaN from rolling windows
print(f"  After indicator warm-up: {len(df)} usable bars\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_breakout(row: pd.Series) -> str:
    """
    Determine whether today's bar constitutes a breakout.

    A LONG breakout occurs when:
        1. The closing price exceeds the previous Donchian high
           (the highest high of the prior DONCHIAN_LOOKBACK days), AND
        2. Today's volume is at least VOLUME_MULT x the DONCHIAN_LOOKBACK-day
           average volume (conviction filter to avoid false breakouts on thin
           volume).

    A SHORT breakout occurs when:
        1. The closing price falls below the previous Donchian low
           (the lowest low of the prior DONCHIAN_LOOKBACK days), AND
        2. The same volume condition holds.

    Parameters
    ----------
    row : pd.Series
        A single daily bar with columns: close, don_high, don_low, volume,
        vol_avg.

    Returns
    -------
    str
        "LONG", "SHORT", or "NONE".
    """
    vol_ok = row["volume"] >= VOLUME_MULT * row["vol_avg"]
    if row["close"] > row["don_high"] and vol_ok:
        return "LONG"
    if row["close"] < row["don_low"] and vol_ok:
        return "SHORT"
    return "NONE"


df["signal"] = df.apply(detect_breakout, axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BACKTESTING ENGINE (walk-forward, out-of-sample only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward backtest using daily bars.

    For each rolling window:
        - Training set   = preceding TRAIN_WINDOW bars (indicator warm-up;
          parameters are fixed, so no optimisation loop).
        - Out-of-sample  = next OOS_WINDOW bars - signals here are traded.

    Trade management (checked on each subsequent daily bar):
        - LONG:  profit target = entry + PROFIT_TARGET_ATR x ATR
                 stop loss     = entry - STOP_LOSS_ATR x ATR
                 On a given day, if bar.low <= stop, exit at stop (conservative
                 assumption that stop is hit first when both levels are
                 breached on the same bar).
        - SHORT: profit target = entry - PROFIT_TARGET_ATR x ATR
                 stop loss     = entry + STOP_LOSS_ATR x ATR
        - Timeout: close at market after TIMEOUT_DAYS trading days.

    Only one position at a time (no overlapping trades).

    Returns a DataFrame - the trade blotter.
    """
    trades = []
    i = TRAIN_WINDOW  # start of first OOS window

    while i < len(data):
        oos_end = min(i + OOS_WINDOW, len(data))
        oos_slice = data.iloc[i:oos_end]

        j = 0
        while j < len(oos_slice):
            row = oos_slice.iloc[j]
            sig = row["signal"]
            if sig == "NONE":
                j += 1
                continue

            # --- ENTRY ---
            entry_date   = oos_slice.index[j]
            entry_price  = row["close"]
            atr_at_entry = row["atr"]
            direction    = sig  # "LONG" or "SHORT"

            if direction == "LONG":
                tp = entry_price + PROFIT_TARGET_ATR * atr_at_entry
                sl = entry_price - STOP_LOSS_ATR * atr_at_entry
            else:
                tp = entry_price - PROFIT_TARGET_ATR * atr_at_entry
                sl = entry_price + STOP_LOSS_ATR * atr_at_entry

            # --- HOLD & EXIT ---
            exit_price = None
            exit_date  = None
            outcome    = None
            holding    = 0

            # Scan forward day-by-day from the bar after entry
            search_start = data.index.get_loc(entry_date) + 1

            for k in range(search_start, min(search_start + TIMEOUT_DAYS, len(data))):
                day_date = data.index[k]
                holding += 1
                bar = data.iloc[k]

                # Stop checked before target (conservative assumption when
                # both levels are breached on the same daily bar)
                if direction == "LONG":
                    if bar["low"] <= sl:
                        exit_price, exit_date, outcome = sl, day_date, "Stop-loss"
                    elif bar["high"] >= tp:
                        exit_price, exit_date, outcome = tp, day_date, "Successful"
                else:  # SHORT
                    if bar["high"] >= sl:
                        exit_price, exit_date, outcome = sl, day_date, "Stop-loss"
                    elif bar["low"] <= tp:
                        exit_price, exit_date, outcome = tp, day_date, "Successful"

                if exit_price is not None:
                    break

            # Timeout: close at market if neither target nor stop was hit
            if exit_price is None:
                timeout_idx = min(search_start + TIMEOUT_DAYS - 1, len(data) - 1)
                exit_price = data.iloc[timeout_idx]["close"]
                exit_date  = data.index[timeout_idx]
                outcome    = "Timed-out"

            # PnL calculation
            if direction == "LONG":
                pnl = (exit_price - entry_price) * POSITION_SIZE
                ret = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) * POSITION_SIZE
                ret = (entry_price - exit_price) / entry_price

            trades.append({
                "entry_date":   entry_date,
                "exit_date":    exit_date,
                "direction":    direction,
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(exit_price, 4),
                "qty":          POSITION_SIZE,
                "pnl":          round(pnl, 2),
                "return":       round(ret, 6),
                "outcome":      outcome,
                "holding_days": holding,
            })

            # Skip ahead past exit so we don't overlap trades
            if exit_date is not None:
                next_j_abs = data.index.get_loc(exit_date) + 1
                j = next_j_abs - data.index.get_loc(oos_slice.index[0])
            else:
                j += 1

        i = oos_end  # roll to next OOS window

    return pd.DataFrame(trades)


print("Running walk-forward backtest ...")
blotter = run_backtest(df)
print(f"  Total trades: {len(blotter)}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(blotter_df: pd.DataFrame) -> dict:
    """
    Compute strategy-level performance metrics.

    Returns a dict with: total trades, average return per trade, annualised
    return and volatility, Sharpe ratio (rf = 3.75%), Sortino ratio,
    max drawdown, win rate, profit factor, and expectancy.
    """
    if len(blotter_df) == 0:
        return {}

    returns  = blotter_df["return"].values
    n_trades = len(returns)
    avg_ret  = returns.mean()
    std_ret  = returns.std(ddof=1) if n_trades > 1 else 0.0

    # Annualise based on actual trade frequency over the data period
    trades_per_year = n_trades / DATA_PERIOD_YEARS
    if trades_per_year == 0:
        trades_per_year = 1

    ann_ret = avg_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year)

    # Sharpe ratio (annualised excess return / annualised volatility)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    # Sortino ratio (penalises only downside volatility)
    neg = returns[returns < 0]
    dd  = np.sqrt((neg ** 2).mean()) * np.sqrt(trades_per_year) if len(neg) > 0 else 0.0
    sortino = (ann_ret - RISK_FREE_RATE) / dd if dd > 0 else np.nan

    # Max drawdown on cumulative PnL curve
    cum_pnl  = blotter_df["pnl"].cumsum()
    peak     = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd   = drawdown.min()

    # Win rate
    wins     = (returns > 0).sum()
    win_rate = wins / n_trades

    # Profit factor = gross profit / gross loss
    gross_profit = blotter_df.loc[blotter_df["pnl"] > 0, "pnl"].sum()
    gross_loss   = abs(blotter_df.loc[blotter_df["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Expectancy = P(win)*avg_win + P(loss)*avg_loss
    avg_win  = returns[returns > 0].mean() if wins > 0 else 0
    avg_loss = returns[returns <= 0].mean() if (n_trades - wins) > 0 else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "Total Trades":              n_trades,
        "Average Return per Trade":  round(avg_ret * 100, 4),
        "Annualised Return (%)":     round(ann_ret * 100, 2),
        "Annualised Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio":              round(sharpe, 4),
        "Sortino Ratio":             round(sortino, 4),
        "Max Drawdown ($)":          round(max_dd, 2),
        "Win Rate (%)":              round(win_rate * 100, 2),
        "Profit Factor":             round(profit_factor, 4),
        "Expectancy (%)":            round(expectancy * 100, 4),
    }


metrics = compute_metrics(blotter)

print("=" * 70)
print("  PERFORMANCE METRICS")
print("=" * 70)
for k, v in metrics.items():
    print(f"  {k:30s}: {v}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRADE OUTCOME SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
outcome_counts = blotter["outcome"].value_counts().to_dict()
print("Trade outcome distribution:")
for o, c in outcome_counts.items():
    print(f"  {o:15s}: {c}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXPORTS - blotter CSV, metrics JSON, chart data CSV
# ═══════════════════════════════════════════════════════════════════════════════
blotter.to_csv("trade_blotter.csv", index=False)
print("  Blotter saved -> trade_blotter.csv")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Metrics saved -> metrics.json")

with open("outcomes.json", "w") as f:
    json.dump(outcome_counts, f, indent=2)
print("  Outcomes saved -> outcomes.json")

# Export daily price + indicators for the webpage chart
chart_df = df[["close", "don_high", "don_low", "volume", "vol_avg", "atr"]].copy()
chart_df.index = chart_df.index.strftime("%Y-%m-%d")
chart_df.to_csv("chart_data.csv")
print("  Chart data saved -> chart_data.csv")

print("\n  Backtest complete.")
