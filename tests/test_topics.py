"""Tests for the topic modelling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.topics.topic_model import TopicModeler


class TestTopicModeler:
    """Tests for the TopicModeler class (sample mode)."""

    @pytest.fixture
    def modeler(self, config: dict) -> TopicModeler:
        """Create a topic modeler from test config."""
        return TopicModeler(config)

    @pytest.fixture
    def sample_docs(self, sample_speeches_df: pd.DataFrame) -> list:
        """Extract document texts from sample data."""
        return sample_speeches_df["text"].tolist()

    def test_fit_sample_returns_topics(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Sample mode fitting should return topic assignments."""
        topics, probs = modeler.fit_sample(sample_docs, n_topics=4)
        assert len(topics) == len(sample_docs)
        assert all(isinstance(t, (int, np.integer)) for t in topics)

    def test_fit_sample_returns_probabilities(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Probabilities should sum to 1 for each document."""
        topics, probs = modeler.fit_sample(sample_docs, n_topics=4)
        assert probs.shape == (len(sample_docs), 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_topic_count_bounded(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Number of discovered topics should be <= n_topics."""
        n_topics = 4
        topics, _ = modeler.fit_sample(sample_docs, n_topics=n_topics)
        unique_topics = set(topics)
        assert len(unique_topics) <= n_topics
        assert len(unique_topics) >= 2

    def test_get_topic_info_after_fit(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Topic info should be available after fitting."""
        modeler.fit_sample(sample_docs, n_topics=4)
        info = modeler.get_topic_info()
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 4
        assert "Topic" in info.columns
        assert "Count" in info.columns

    def test_get_topic_words(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Should return word-weight pairs for a given topic."""
        modeler.fit_sample(sample_docs, n_topics=4)
        words = modeler.get_topic_words(0)
        assert len(words) > 0
        assert all(isinstance(w, str) and isinstance(s, float) for w, s in words)

    def test_transform_after_fit(
        self, modeler: TopicModeler, sample_docs: list
    ) -> None:
        """Transform should work on new documents after fitting."""
        modeler.fit_sample(sample_docs, n_topics=4)
        new_docs = ["Inflation continues to rise above target levels."]
        topics, probs = modeler.transform(new_docs)
        assert len(topics) == 1
        assert probs.shape[0] == 1

    def test_transform_without_fit_raises(self, modeler: TopicModeler) -> None:
        """Transform without prior fit should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            modeler.transform(["Some text"])

    def test_topics_over_time(
        self, modeler: TopicModeler, sample_docs: list,
        sample_speeches_df: pd.DataFrame
    ) -> None:
        """Topic evolution over time should return a DataFrame."""
        topics, _ = modeler.fit_sample(sample_docs, n_topics=4)
        timestamps = sample_speeches_df["date"].dt.strftime("%Y-%m-%d").tolist()
        result = modeler.get_topics_over_time(
            sample_docs, timestamps, topics, nr_bins=5
        )
        assert isinstance(result, pd.DataFrame)
        assert "topic" in result.columns
        assert "frequency" in result.columns


class TestTopicPersistence:
    """Tests for model save/load."""

    @pytest.fixture
    def modeler(self, config: dict) -> TopicModeler:
        return TopicModeler(config)

    def test_save_and_load_sample_model(
        self, modeler: TopicModeler, sample_speeches_df: pd.DataFrame, tmp_path
    ) -> None:
        """Sample model should be saveable and loadable."""
        docs = sample_speeches_df["text"].tolist()
        topics_orig, _ = modeler.fit_sample(docs, n_topics=4)

        modeler.save(str(tmp_path))

        new_modeler = TopicModeler({"topics": {}})
        new_modeler.load(str(tmp_path / "sample_topic_model.pkl"))

        topics_loaded, _ = new_modeler.transform(docs)
        assert topics_loaded == topics_orig
