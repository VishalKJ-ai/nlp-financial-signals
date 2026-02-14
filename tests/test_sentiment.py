"""Tests for the sentiment analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sentiment.finbert_scorer import FinBERTScorer


class TestFinBERTScorer:
    """Tests for the FinBERTScorer class (rule-based / sample mode)."""

    @pytest.fixture
    def scorer(self, config: dict) -> FinBERTScorer:
        """Create a scorer in precomputed (rule-based) mode."""
        return FinBERTScorer(config, use_precomputed=True)

    def test_score_documents_adds_columns(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Scoring should add all expected sentiment columns."""
        result = scorer.score_documents(sample_speeches_df)
        expected_cols = [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_compound", "sentiment_label",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_probabilities_sum_to_one(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Positive + negative + neutral should sum to ~1.0."""
        result = scorer.score_documents(sample_speeches_df)
        totals = (
            result["sentiment_positive"]
            + result["sentiment_negative"]
            + result["sentiment_neutral"]
        )
        np.testing.assert_allclose(totals, 1.0, atol=0.01)

    def test_compound_score_range(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Compound score should be in [-1, 1]."""
        result = scorer.score_documents(sample_speeches_df)
        assert (result["sentiment_compound"] >= -1.0).all()
        assert (result["sentiment_compound"] <= 1.0).all()

    def test_sentiment_label_valid(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Labels should be one of positive, negative, neutral."""
        result = scorer.score_documents(sample_speeches_df)
        valid_labels = {"positive", "negative", "neutral"}
        assert set(result["sentiment_label"].unique()).issubset(valid_labels)

    def test_rule_based_hawkish_text(self, scorer: FinBERTScorer) -> None:
        """Hawkish text should score positively (inflation up = hawkish)."""
        text = (
            "We must tighten monetary policy further. Inflation remains "
            "persistent and well above our target. Wage pressures are "
            "overheating the economy. Higher for longer is the approach."
        )
        score = scorer._rule_based_score(text)
        assert score["positive"] > score["negative"]

    def test_rule_based_dovish_text(self, scorer: FinBERTScorer) -> None:
        """Dovish text should score negatively."""
        text = (
            "The economy shows clear signs of slowdown and weakness. "
            "We should consider cutting rates to provide accommodative "
            "support. Disinflation is well underway and subdued demand "
            "poses downside risks to growth."
        )
        score = scorer._rule_based_score(text)
        assert score["negative"] > score["positive"]


class TestStanceClassification:
    """Tests for monetary policy stance classification."""

    @pytest.fixture
    def scorer(self, config: dict) -> FinBERTScorer:
        return FinBERTScorer(config, use_precomputed=True)

    def test_classify_stance_adds_columns(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Stance classification should add stance and stance_score."""
        scored = scorer.score_documents(sample_speeches_df)
        result = scorer.classify_stance(scored)
        assert "stance" in result.columns
        assert "stance_score" in result.columns

    def test_stance_values(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Stance should be hawkish, dovish, or neutral."""
        scored = scorer.score_documents(sample_speeches_df)
        result = scorer.classify_stance(scored)
        valid_stances = {"hawkish", "dovish", "neutral"}
        assert set(result["stance"].unique()).issubset(valid_stances)


class TestParagraphScoring:
    """Tests for paragraph-level sentiment analysis."""

    @pytest.fixture
    def scorer(self, config: dict) -> FinBERTScorer:
        return FinBERTScorer(config, use_precomputed=True)

    def test_paragraph_scoring_produces_more_rows(
        self, scorer: FinBERTScorer, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Paragraph scoring should produce more rows than documents."""
        # Use documents with multiple paragraphs
        df = sample_speeches_df.copy()
        df["text"] = df["text"].apply(
            lambda t: t[:len(t)//2] + "\n\n" + t[len(t)//2:]
        )
        result = scorer.score_paragraphs(df)
        assert len(result) >= len(df)
        assert "paragraph_id" in result.columns
