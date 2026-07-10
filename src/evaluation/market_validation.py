"""External market validation (Evaluation Arm 3).

Tests whether cluster-level sentiment carries economically substantive
signal by relating it to meeting-day changes in market-implied policy
expectations, and compares its explanatory power against the
document-level baseline (the signal-washout test from Ch3 s3.5).

Expectation proxies, both free to access:
    * 2-year Treasury constant-maturity yield (FRED: DGS2), the
      standard short-rate expectations proxy in the event-study
      literature.
    * 30-day fed funds futures (front month, Yahoo Finance: ZQ=F),
      where the implied rate is 100 minus the price.  Coverage on
      Yahoo is patchy for early years, so DGS2 is the primary series
      and futures are reported where available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ── Market data ───────────────────────────────────────────────────────────────


def fetch_fred_series(series: str, start: str, end: str) -> pd.Series:
    """Fetch a daily series from FRED.

    Args:
        series: FRED series identifier, e.g. "DGS2".
        start: ISO start date.
        end: ISO end date.

    Returns:
        Series indexed by date, forward-filled over holidays.
    """
    import pandas_datareader.data as web

    frame = web.DataReader(series, "fred", start, end)
    values = frame[series].ffill()
    logger.info("Fetched %d observations of %s (%s to %s)",
                len(values), series, values.index.min().date(),
                values.index.max().date())
    return values


def fetch_futures_implied_rate(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Fetch fed funds futures and convert to the implied rate.

    Args:
        ticker: Yahoo Finance ticker, e.g. "ZQ=F".
        start: ISO start date.
        end: ISO end date.

    Returns:
        Implied-rate series (100 - price), or None if unavailable.
    """
    try:
        import yfinance as yf

        history = yf.Ticker(ticker).history(start=start, end=end)
        if history.empty:
            logger.warning("No futures history returned for %s", ticker)
            return None
        implied = (100.0 - history["Close"]).rename("implied_rate")
        implied.index = implied.index.tz_localize(None)
        logger.info("Fetched %d futures observations (%s to %s)",
                    len(implied), implied.index.min().date(),
                    implied.index.max().date())
        return implied
    except Exception as exc:  # Network/AP flakiness must not kill the arm.
        logger.warning("Futures fetch failed for %s: %s", ticker, exc)
        return None


def event_window_change(
    series: pd.Series, dates: pd.Series, window_days: int = 1
) -> pd.Series:
    """Change in a market series around each event date.

    Computes value(first trading day >= date + window) minus
    value(last trading day < date), capturing the market response to
    the meeting and press conference.

    Args:
        series: Daily market series indexed by date.
        dates: Event (meeting) dates.
        window_days: Days after the event to measure the close.

    Returns:
        Series of changes indexed by event date.
    """
    idx = series.index
    changes = {}
    for date in pd.to_datetime(dates):
        before = idx[idx < date]
        after = idx[idx >= date + pd.Timedelta(days=window_days)]
        if len(before) == 0 or len(after) == 0:
            continue
        changes[date] = float(series.loc[after[0]] - series.loc[before[-1]])
    return pd.Series(changes, name=f"d_{series.name}")


# ── Regression analysis ───────────────────────────────────────────────────────


def _ols_row(y: pd.Series, X: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Fit OLS with HC1 robust errors and summarise one predictor set."""
    data = pd.concat([y, X], axis=1).dropna()
    fitted = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit(
        cov_type="HC1"
    )
    main = data.columns[1]
    return {
        "model": label,
        "n": int(fitted.nobs),
        "coef": float(fitted.params.get(main, np.nan)),
        "se": float(fitted.bse.get(main, np.nan)),
        "p_value": float(fitted.pvalues.get(main, np.nan)),
        "r_squared": float(fitted.rsquared),
        "adj_r_squared": float(fitted.rsquared_adj),
    }


def run_validation(config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Run the external validation regressions (Arm 3).

    Args:
        config: Dissertation configuration.
        project_root: Repository root for resolving paths.

    Returns:
        Regression summary table (also written to the tables dir).
    """
    results_dir = project_root / config["output"]["results_dir"]
    tables_dir = project_root / config["output"]["tables_dir"]
    market_cfg = config["evaluation"]["market"]

    document = pd.read_csv(results_dir / "document_sentiment_series.csv",
                           parse_dates=["date"])
    cluster = pd.read_csv(results_dir / "cluster_sentiment_series.csv",
                          parse_dates=["date"])

    start = (document["date"].min() - pd.Timedelta(days=10)).date().isoformat()
    end = (document["date"].max() + pd.Timedelta(days=10)).date().isoformat()

    dgs2 = fetch_fred_series(market_cfg["fred_series"], start, end)
    d_yield = event_window_change(dgs2, document["date"],
                                  market_cfg["event_window_days"])
    d_yield.to_csv(results_dir / "market_event_changes.csv")

    futures = fetch_futures_implied_rate(market_cfg["futures_ticker"],
                                         start, end)
    d_futures = (
        event_window_change(futures, document["date"],
                            market_cfg["event_window_days"])
        if futures is not None else None
    )

    # Meeting x topic sentiment matrix; keep persistent topics only so
    # regressions are not dominated by sparse cells.
    matrix = cluster.pivot(index="date", columns="topic",
                           values="cluster_compound")
    persistence = matrix.notna().mean()
    persistent_topics = persistence[persistence >= 0.6].index.tolist()
    logger.info("%d of %d topics appear in >=60%% of meetings",
                len(persistent_topics), matrix.shape[1])

    doc_series = document.set_index("date")["document_compound"]

    rows: List[Dict[str, Any]] = []
    targets = {"d_dgs2": d_yield}
    if d_futures is not None and len(d_futures) >= 30:
        targets["d_ffutures"] = d_futures

    for target_name, target in targets.items():
        # Document-level baseline (the washout comparator).
        row = _ols_row(target.rename(target_name),
                       doc_series.to_frame("document_compound"),
                       "document_baseline")
        row["target"] = target_name
        rows.append(row)

        # Per-topic univariate regressions.
        for topic in persistent_topics:
            row = _ols_row(target.rename(target_name),
                           matrix[[topic]].rename(
                               columns={topic: f"topic_{topic}"}),
                           f"cluster_topic_{topic}")
            row["target"] = target_name
            rows.append(row)

        # Multivariate: the ten most persistent topics jointly.  A
        # missing meeting-topic cell means the theme was not discussed,
        # which carries zero theme signal, so absent cells are filled
        # with 0 rather than dropping the meeting.
        top_persistent = persistence.sort_values(ascending=False).head(10)
        joint = matrix[top_persistent.index].fillna(0.0)
        joint.columns = [f"topic_{t}" for t in top_persistent.index]
        data = pd.concat([target.rename(target_name), joint],
                         axis=1).dropna(subset=[target_name])
        if len(data) > joint.shape[1] + 5:
            fitted = sm.OLS(data[target_name],
                            sm.add_constant(data.drop(columns=target_name))
                            ).fit(cov_type="HC1")
            rows.append({
                "model": "cluster_multivariate",
                "target": target_name,
                "n": int(fitted.nobs),
                "coef": np.nan,
                "se": np.nan,
                "p_value": float(fitted.f_pvalue),
                "r_squared": float(fitted.rsquared),
                "adj_r_squared": float(fitted.rsquared_adj),
            })

    table = pd.DataFrame(rows)[
        ["target", "model", "n", "coef", "se", "p_value",
         "r_squared", "adj_r_squared"]
    ]
    # Benjamini-Hochberg correction across the per-topic tests within
    # each target, so cluster-level significance survives an honest
    # multiple-testing treatment.
    table["p_bh"] = np.nan
    for target_name in table["target"].unique():
        mask = (table["target"] == target_name) & table["model"].str.startswith(
            "cluster_topic_")
        if mask.sum():
            from statsmodels.stats.multitest import multipletests

            table.loc[mask, "p_bh"] = multipletests(
                table.loc[mask, "p_value"], method="fdr_bh")[1]
    table.to_csv(tables_dir / "market_validation.csv", index=False)
    logger.info("Market validation:\n%s", table.to_string(index=False))
    return table


def main() -> None:
    """CLI entry point."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    with open(project_root / "config/dissertation.yaml") as handle:
        config = yaml.safe_load(handle)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    run_validation(config, project_root)


if __name__ == "__main__":
    main()
