from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Once the rolling adf p-values drops below 0.05, we will treat the regime as mean reverting
# until the rolling adf rebounce to 0.15 or above.

@dataclass(frozen=True)
class PairConfig:
    pair: str
    filename: str


PAIR_CONFIGS = [
    PairConfig("EURUSD", "eurusdohlcba.xlsx"),
    PairConfig("GBPUSD", "gbpusdohlcba.xlsx"),
    PairConfig("USDCAD", "usdcadphlcba.xlsx"),
    PairConfig("USDJPY", "usdjpyohlcba.xlsx"),
]

DATA_COLUMNS = ["Date", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"

ADF_WINDOW = 90
ENTRY_P_THRESHOLD = 0.05
EXIT_P_THRESHOLD = 0.15


def load_bfix_ohlc(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath, skiprows=6)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.loc[:, DATA_COLUMNS].copy()

    for column in DATA_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["PX_LAST"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    return df.sort_values("Date").reset_index(drop=True)


def rolling_adf_pvalues(series: pd.Series, window: int) -> pd.Series:
    p_values = [np.nan] * len(series)

    for end_idx in range(window, len(series) + 1):
        window_series = series.iloc[end_idx - window:end_idx]
        try:
            p_values[end_idx - 1] = adfuller(
                window_series,
                regression="c",
                autolag="AIC",
            )[1]
        except ValueError:
            p_values[end_idx - 1] = np.nan

    return pd.Series(p_values, index=series.index, dtype="float64")


def apply_hysteresis(p_values: pd.Series) -> pd.Series:
    regime_states: list[object] = []
    state = False

    for p_value in p_values:
        if pd.isna(p_value):
            regime_states.append(np.nan)
            continue

        if not state and p_value < ENTRY_P_THRESHOLD:
            state = True
        elif state and p_value > EXIT_P_THRESHOLD:
            state = False

        regime_states.append(state)

    return pd.Series(regime_states, index=p_values.index, dtype="object")


def build_pair_analysis(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    analysis = df[["Date", "PX_LAST"]].copy()
    analysis["Pair"] = pair
    analysis["ADF_PValue"] = rolling_adf_pvalues(analysis["PX_LAST"], ADF_WINDOW)
    analysis["MeanRev_Regime"] = apply_hysteresis(analysis["ADF_PValue"])
    analysis["Regime"] = np.where(
        analysis["MeanRev_Regime"].eq(True),
        "Mean-Reverting",
        np.where(
            analysis["MeanRev_Regime"].eq(False),
            "Trending",
            "Insufficient Data",
        ),
    )
    analysis["Method"] = "Hysteresis"
    analysis["Entry Threshold"] = ENTRY_P_THRESHOLD
    analysis["Exit Threshold"] = EXIT_P_THRESHOLD
    return analysis


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []

    for pair, pair_df in results.groupby("Pair", sort=True):
        valid = pair_df[pair_df["MeanRev_Regime"].isin([True, False])].copy()
        if valid.empty:
            summaries.append(
                {
                    "Pair": pair,
                    "Method": "Hysteresis",
                    "Observations": 0,
                    "Mean-Reverting Days": 0,
                    "Trending Days": 0,
                    "Mean-Reverting %": np.nan,
                    "Average P-Value": np.nan,
                    "Regime Switches": 0,
                    "Latest Date": None,
                    "Latest P-Value": np.nan,
                    "Latest Regime": "Insufficient Data",
                }
            )
            continue

        regime_series = valid["MeanRev_Regime"].astype(bool)
        switches = int(regime_series.ne(regime_series.shift()).sum() - 1)
        latest = valid.iloc[-1]

        summaries.append(
            {
                "Pair": pair,
                "Method": "Hysteresis",
                "Observations": int(len(valid)),
                "Mean-Reverting Days": int(regime_series.sum()),
                "Trending Days": int((~regime_series).sum()),
                "Mean-Reverting %": round(100 * regime_series.mean(), 2),
                "Average P-Value": round(float(valid["ADF_PValue"].mean()), 4),
                "Regime Switches": max(switches, 0),
                "Latest Date": latest["Date"].date().isoformat(),
                "Latest P-Value": round(float(latest["ADF_PValue"]), 4),
                "Latest Regime": latest["Regime"],
            }
        )

    return pd.DataFrame(summaries)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "adf_hysteresis_outputs"
    output_dir.mkdir(exist_ok=True)

    all_results: list[pd.DataFrame] = []

    for config in PAIR_CONFIGS:
        df = load_bfix_ohlc(base_dir / config.filename)
        pair_results = build_pair_analysis(df, config.pair)
        all_results.append(pair_results)
        pair_results.to_csv(
            output_dir / f"{config.pair.lower()}_rolling_adf_hysteresis.csv",
            index=False,
        )

    combined = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(combined)

    combined.to_csv(output_dir / "combined_rolling_adf_hysteresis.csv", index=False)
    summary.to_csv(output_dir / "summary_rolling_adf_hysteresis.csv", index=False)

    print("\n=== Rolling ADF Hysteresis Summary ===")
    print(summary.to_string(index=False))
    print(f"\nDetailed outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
