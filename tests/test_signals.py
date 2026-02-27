"""Tests for the financial signal extraction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.extractor import SignalExtractor


class TestSignalExtractor:
    """Tests for the SignalExtractor class."""

    @pytest.fixture
    def extractor(self, config: dict) -> SignalExtractor:
        """Create a signal extractor from test config."""
        return SignalExtractor(config)

    @pytest.fixture
    def scored_df(self, sample_speeches_df: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame with stance scores."""
        df = sample_speeches_df.copy()
        np.random.seed(42)
        df["stance_score"] = np.random.uniform(-0.5, 0.5, len(df))
        df["sentiment_compound"] = np.random.uniform(-0.3, 0.3, len(df))
        return df

    def test_hawkish_dovish_index_range(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """HDI should be bounded within [-1, 1]."""
        result = extractor.compute_hawkish_dovish_index(scored_df)
        assert "hawkish_dovish_index" in result.columns
        assert (result["hawkish_dovish_index"] >= -1.0).all()
        assert (result["hawkish_dovish_index"] <= 1.0).all()

    def test_hawkish_dovish_index_has_monthly_rows(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """Should produce one row per month with data."""
        result = extractor.compute_hawkish_dovish_index(scored_df)
        assert len(result) > 0
        assert "year_month" in result.columns
        assert "document_count" in result.columns

    def test_topic_shift_nonnegative(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """Jensen-Shannon divergence should be non-negative."""
        np.random.seed(42)
        n = len(scored_df)
        topic_probs = np.random.dirichlet(np.ones(6), size=n)
        result = extractor.compute_topic_shift(topic_probs, scored_df["date"])
        assert "topic_shift" in result.columns
        assert (result["topic_shift"] >= 0).all()

    def test_sentiment_momentum_computed(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """Sentiment momentum should be computable."""
        result = extractor.compute_sentiment_momentum(scored_df, window=2)
        assert "sentiment_momentum" in result.columns
        assert len(result) > 0

    def test_composite_signal_range(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """Composite signal should be bounded."""
        hdi = extractor.compute_hawkish_dovish_index(scored_df)

        np.random.seed(42)
        topic_probs = np.random.dirichlet(np.ones(6), size=len(scored_df))
        topic_shift = extractor.compute_topic_shift(topic_probs, scored_df["date"])
        sentiment_mom = extractor.compute_sentiment_momentum(scored_df)

        result = extractor.compute_composite_signal(hdi, topic_shift, sentiment_mom)
        assert "composite_signal" in result.columns
        assert "signal_label" in result.columns
        assert (result["composite_signal"] >= -1.5).all()
        assert (result["composite_signal"] <= 1.5).all()

    def test_signal_labels_valid(
        self, extractor: SignalExtractor, scored_df: pd.DataFrame
    ) -> None:
        """Signal labels should be hawkish, dovish, or neutral."""
        hdi = extractor.compute_hawkish_dovish_index(scored_df)
        np.random.seed(42)
        topic_probs = np.random.dirichlet(np.ones(6), size=len(scored_df))
        topic_shift = extractor.compute_topic_shift(topic_probs, scored_df["date"])
        sentiment_mom = extractor.compute_sentiment_momentum(scored_df)

        result = extractor.compute_composite_signal(hdi, topic_shift, sentiment_mom)
        valid_labels = {"hawkish", "dovish", "neutral"}
        assert set(result["signal_label"].unique()).issubset(valid_labels)

    def test_smooth_signals(
        self, extractor: SignalExtractor
    ) -> None:
        """Smoothing should produce a less volatile series."""
        df = pd.DataFrame({
            "year_month": pd.period_range("2020-01", periods=12, freq="M"),
            "composite_signal": np.random.randn(12),
        })
        result = extractor.smooth_signals(df)
        assert "composite_signal_smooth" in result.columns
        # Smoothed series should have lower variance
        assert result["composite_signal_smooth"].std() <= df["composite_signal"].std() + 0.01

    def test_correlate_with_market(self, extractor: SignalExtractor) -> None:
        """Market correlation should return results for each lag."""
        signal_df = pd.DataFrame({
            "year_month": pd.period_range("2020-01", periods=24, freq="M"),
            "composite_signal": np.random.randn(24),
        })
        market_df = pd.DataFrame({
            "year_month": pd.period_range("2020-01", periods=24, freq="M"),
            "returns": np.random.randn(24) * 0.02,
        })
        result = extractor.correlate_with_market(
            signal_df, market_df, lags=[0, 1, 3]
        )
        assert len(result) == 3
        assert "lag" in result.columns
        assert "correlation" in result.columns
