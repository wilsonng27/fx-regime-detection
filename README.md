# FX Mean-Reversion Regime Detection

This project provides a quantitative framework for detecting mean-reverting regimes in foreign exchange (FX) markets. It utilizes a Rolling Augmented Dickey-Fuller (ADF) test on daily OHLC data to determine whether a currency pair is in a trending or mean-reverting state.

Because raw $p$-values from rolling ADF tests can be highly volatile and lead to "whipsawing" (rapidly switching between regimes), this repository implements three distinct quantitative filtering methods to stabilize the signals.

## The Three Methods

1. **Hold-Lock (`_rolling_adf_hold_lock.py`):** When the rolling ADF $p$-value drops below the target threshold (e.g., 0.10), the regime is locked as "mean-reverting" for a fixed duration (e.g., 5 days). This prevents premature exits due to daily noise but may slightly lag behind rapid market structure changes.

2. **Hysteresis (`_rolling_adf_hysteresis.py`):** Utilizes a dual-threshold approach. The regime enters a mean-reverting state when the $p$-value drops below a strict entry threshold (0.05) and only exits when it rises above a looser exit threshold (0.15). This effectively eliminates micro-fluctuations around a single threshold.

3. **Smoothed EMA (`_rolling_adf_smoothed.py`):** Applies an Exponential Moving Average (EMA) to the raw rolling $p$-values before evaluating them against the threshold. This smooths out erratic spikes in the test statistic while maintaining responsiveness to genuine structural breaks.

## Data Requirements

**Note:** Raw market data is NOT included in this repository to comply with data provider licensing agreements. 

To run these scripts, you must provide your own daily OHLC data (e.g., Bloomberg BFIX). The scripts expect `.xlsx` files formatted with the first 6 rows as headers/metadata. The data must start on row 7 with the following exact column names:
`Date`, `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST`

Update the `PAIR_CONFIGS` array in each script to match your local filenames.

## How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt