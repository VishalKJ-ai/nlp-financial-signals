"""Financial signal extraction module.

Combines topic modelling outputs and sentiment scores to produce
composite financial signals indicating hawkish/dovish monetary policy
shifts.  Signals can be correlated with market movements to assess
their informational value.

The three signal components are:
1. **Topic shift** — changes in the distribution of discussion topics
2. **Sentiment level** — absolute sentiment polarity of documents
3. **Sentiment momentum** — rate of change in sentiment over time
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalExtractor:
    """Extracts composite financial signals from NLP analysis outputs.

    Combines topic distribution changes and sentiment scores into a
    single composite signal for each central bank over time.  The
    signal is designed to capture shifts in monetary policy stance
    that may precede market movements.

    Attributes:
        weights: Component weights for the composite signal.
        smoothing_window: Rolling average window for signal smoothing.
        hawkish_threshold: Threshold for hawkish classification.
        dovish_threshold: Threshold for dovish classification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the signal extractor from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The ``signals``
                sub-key is used.
        """
        sig_cfg = config.get("signals", {})
        weights = sig_cfg.get("composite_weights", {})
        self.weight_topic_shift: float = weights.get("topic_shift", 0.3)
        self.weight_sentiment_level: float = weights.get("sentiment_level", 0.4)
        self.weight_sentiment_change: float = weights.get("sentiment_change", 0.3)
        self.smoothing_window: int = sig_cfg.get("smoothing_window", 3)
        self.hawkish_threshold: float = sig_cfg.get("hawkish_threshold", 0.3)
        self.dovish_threshold: float = sig_cfg.get("dovish_threshold", -0.3)
        self.market_tickers: List[str] = sig_cfg.get("market_tickers", [])

    def compute_hawkish_dovish_index(
        self,
        df: pd.DataFrame,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute a monthly hawkish-dovish index from scored documents.

        The index ranges from -1 (strongly dovish) to +1 (strongly
        hawkish), based on document-level stance scores aggregated
        to monthly frequency.

        Args:
            df: DataFrame with columns: date, source, stance_score.
            source: Optional filter for a specific central bank.

        Returns:
            Monthly DataFrame with hawkish_dovish_index column.
        """
        df = df.copy()
        if source:
            df = df[df["source"] == source]

        if "date" not in df.columns:
            logger.warning("No date column found, returning empty DataFrame")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        monthly = df.groupby("year_month").agg(
            hawkish_dovish_index=("stance_score", "mean"),
            document_count=("stance_score", "count"),
            stance_std=("stance_score", "std"),
        ).reset_index()

        monthly["stance_std"] = monthly["stance_std"].fillna(0)

        logger.info("Computed hawkish-dovish index: %d months, mean=%.3f",
                     len(monthly), monthly["hawkish_dovish_index"].mean())
        return monthly

    def compute_topic_shift(
        self,
        topic_probs: np.ndarray,
        dates: pd.Series,
    ) -> pd.DataFrame:
        """Compute topic distribution shift over time.

        Measures the Jensen-Shannon divergence between consecutive
        monthly topic distributions to detect shifts in the themes
        central banks are discussing.

        Args:
            topic_probs: Array of shape (n_docs, n_topics) with
                topic probability distributions.
            dates: Series of document dates.

        Returns:
            Monthly DataFrame with topic_shift column.
        """
        from scipy.spatial.distance import jensenshannon

        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
        })
        # Add topic probability columns
        for i in range(topic_probs.shape[1]):
            df[f"topic_{i}"] = topic_probs[:, i]

        df["year_month"] = df["date"].dt.to_period("M")
        topic_cols = [c for c in df.columns if c.startswith("topic_")]

        # Monthly average topic distribution
        monthly = df.groupby("year_month")[topic_cols].mean().reset_index()
        monthly = monthly.sort_values("year_month")

        # Compute JS divergence between consecutive months
        shifts = [0.0]  # First month has no shift
        for i in range(1, len(monthly)):
            prev_dist = monthly.iloc[i - 1][topic_cols].values.astype(float)
            curr_dist = monthly.iloc[i][topic_cols].values.astype(float)

            # Ensure valid probability distributions
            prev_dist = np.clip(prev_dist, 1e-10, None)
            curr_dist = np.clip(curr_dist, 1e-10, None)
            prev_dist = prev_dist / prev_dist.sum()
            curr_dist = curr_dist / curr_dist.sum()

            jsd = jensenshannon(prev_dist, curr_dist)
            shifts.append(float(jsd))

        monthly["topic_shift"] = shifts
        logger.info("Topic shift: mean=%.4f, max=%.4f",
                     np.mean(shifts), np.max(shifts))
        return monthly[["year_month", "topic_shift"]]

    def compute_sentiment_momentum(
        self,
        df: pd.DataFrame,
        window: int = 3,
    ) -> pd.DataFrame:
        """Compute sentiment momentum (rate of change over time).

        Args:
            df: DataFrame with date and sentiment_compound columns.
            window: Rolling window for momentum calculation (months).

        Returns:
            Monthly DataFrame with sentiment_momentum column.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        monthly = df.groupby("year_month").agg(
            sentiment_level=("sentiment_compound", "mean"),
        ).reset_index()
        monthly = monthly.sort_values("year_month")

        # Momentum = change from window months ago
        monthly["sentiment_momentum"] = (
            monthly["sentiment_level"]
            - monthly["sentiment_level"].shift(window)
        ).fillna(0)

        logger.info("Sentiment momentum: mean=%.4f",
                     monthly["sentiment_momentum"].mean())
        return monthly

    def compute_composite_signal(
        self,
        hdi: pd.DataFrame,
        topic_shift: pd.DataFrame,
        sentiment_momentum: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the composite financial signal.

        Combines three components with configurable weights:
        1. Hawkish-dovish index (sentiment level)
        2. Topic distribution shift
        3. Sentiment momentum

        Args:
            hdi: Monthly hawkish-dovish index DataFrame.
            topic_shift: Monthly topic shift DataFrame.
            sentiment_momentum: Monthly sentiment momentum DataFrame.

        Returns:
            Monthly DataFrame with composite_signal and signal_label.
        """
        logger.info("Computing composite signal with weights: "
                     "topic_shift=%.2f, sentiment_level=%.2f, sentiment_change=%.2f",
                     self.weight_topic_shift, self.weight_sentiment_level,
                     self.weight_sentiment_change)

        # Merge all components on year_month
        signal = hdi[["year_month", "hawkish_dovish_index"]].copy()
        signal = signal.merge(
            topic_shift[["year_month", "topic_shift"]],
            on="year_month", how="left",
        )
        signal = signal.merge(
            sentiment_momentum[["year_month", "sentiment_momentum"]],
            on="year_month", how="left",
        )

        # Fill missing values
        signal = signal.fillna(0)

        # Normalise each component to [-1, 1] range
        for col in ["hawkish_dovish_index", "topic_shift", "sentiment_momentum"]:
            col_max = signal[col].abs().max()
            if col_max > 0:
                signal[f"{col}_norm"] = signal[col] / col_max
            else:
                signal[f"{col}_norm"] = 0.0

        # Weighted combination
        signal["composite_signal"] = (
            self.weight_sentiment_level * signal["hawkish_dovish_index_norm"]
            + self.weight_topic_shift * signal["topic_shift_norm"]
            + self.weight_sentiment_change * signal["sentiment_momentum_norm"]
        )

        # Classify signal
        signal["signal_label"] = "neutral"
        signal.loc[
            signal["composite_signal"] > self.hawkish_threshold, "signal_label"
        ] = "hawkish"
        signal.loc[
            signal["composite_signal"] < self.dovish_threshold, "signal_label"
        ] = "dovish"

        logger.info(
            "Composite signal: %d hawkish, %d dovish, %d neutral months",
            (signal["signal_label"] == "hawkish").sum(),
            (signal["signal_label"] == "dovish").sum(),
            (signal["signal_label"] == "neutral").sum(),
        )
        return signal

    def smooth_signals(self, df: pd.DataFrame, column: str = "composite_signal") -> pd.DataFrame:
        """Apply rolling average smoothing to a signal.

        Args:
            df: DataFrame with the signal column.
            column: Name of the column to smooth.

        Returns:
            DataFrame with smoothed signal column added.
        """
        df = df.copy()
        df[f"{column}_smooth"] = (
            df[column]
            .rolling(window=self.smoothing_window, min_periods=1, center=False)
            .mean()
        )
        return df

    def correlate_with_market(
        self,
        signal_df: pd.DataFrame,
        market_df: pd.DataFrame,
        signal_col: str = "composite_signal",
        market_col: str = "returns",
        lags: List[int] = None,
    ) -> pd.DataFrame:
        """Compute cross-correlation between signal and market returns.

        Args:
            signal_df: DataFrame with year_month and signal column.
            market_df: DataFrame with year_month and market returns.
            signal_col: Name of the signal column.
            market_col: Name of the market returns column.
            lags: List of lag values to compute (default: [0, 1, 3, 6]).

        Returns:
            DataFrame with correlation coefficients at each lag.
        """
        if lags is None:
            lags = [0, 1, 3, 6]

        merged = signal_df.merge(market_df, on="year_month", how="inner")
        if len(merged) < 10:
            logger.warning("Insufficient data for correlation (%d rows)", len(merged))
            return pd.DataFrame({"lag": lags, "correlation": [np.nan] * len(lags)})

        correlations = []
        for lag in lags:
            if lag == 0:
                corr = merged[signal_col].corr(merged[market_col])
            else:
                shifted = merged[market_col].shift(-lag)
                corr = merged[signal_col].corr(shifted)
            correlations.append({
                "lag": lag,
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
            })

        result = pd.DataFrame(correlations)
        logger.info("Signal-market correlation at lags %s: %s",
                     lags, [f"{c['correlation']:.3f}" for c in correlations])
        return result

    def save_signals(
        self,
        df: pd.DataFrame,
        output_path: str,
        source: str = "all",
    ) -> None:
        """Save extracted signals to CSV.

        Args:
            df: Signal DataFrame.
            output_path: Directory to save signals.
            source: Central bank identifier for the filename.
        """
        from pathlib import Path

        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        # Convert Period to string for CSV
        df_out = df.copy()
        for col in df_out.columns:
            if hasattr(df_out[col], "dt") and hasattr(df_out[col].dt, "to_timestamp"):
                try:
                    df_out[col] = df_out[col].astype(str)
                except Exception:
                    pass

        filepath = path / f"signals_{source}.csv"
        df_out.to_csv(filepath, index=False)
        logger.info("Saved signals to %s", filepath)
