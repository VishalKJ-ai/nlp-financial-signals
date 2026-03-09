"""Tests for pipeline configuration and integration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


class TestConfigLoading:
    """Tests for pipeline configuration file."""

    def test_config_loads_successfully(self, config: dict) -> None:
        """Configuration YAML should parse without error."""
        assert isinstance(config, dict)

    def test_config_has_required_sections(self, config: dict) -> None:
        """Config should contain all required top-level sections."""
        required = [
            "data", "preprocessing", "topics", "sentiment",
            "signals", "evaluation", "output", "logging",
        ]
        for section in required:
            assert section in config, f"Missing config section: {section}"

    def test_config_data_sources(self, config: dict) -> None:
        """Data section should define all three central bank sources."""
        sources = config["data"]["sources"]
        assert "boe" in sources
        assert "fed" in sources
        assert "ecb" in sources

    def test_config_topics_has_umap_params(self, config: dict) -> None:
        """Topics config should include UMAP parameters."""
        umap = config["topics"]["umap"]
        assert "n_neighbors" in umap
        assert "n_components" in umap
        assert "random_state" in umap

    def test_config_topics_has_hdbscan_params(self, config: dict) -> None:
        """Topics config should include HDBSCAN parameters."""
        hdbscan = config["topics"]["hdbscan"]
        assert "min_cluster_size" in hdbscan
        assert "min_samples" in hdbscan

    def test_config_sentiment_model(self, config: dict) -> None:
        """Sentiment config should specify a model name."""
        assert "model_name" in config["sentiment"]
        assert "finbert" in config["sentiment"]["model_name"].lower()

    def test_config_signal_weights_sum_to_one(self, config: dict) -> None:
        """Composite signal weights should approximately sum to 1.0."""
        weights = config["signals"]["composite_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_config_evaluation_coherence_measures(self, config: dict) -> None:
        """Evaluation should list valid coherence measures."""
        measures = config["evaluation"]["coherence_measures"]
        assert isinstance(measures, list)
        assert len(measures) > 0
        valid = {"c_v", "c_npmi", "c_uci", "u_mass"}
        assert all(m in valid for m in measures)


class TestSamplePipeline:
    """Integration tests for the sample mode pipeline."""

    def test_sample_data_exists(self, project_root: Path) -> None:
        """Sample speeches CSV should exist."""
        path = project_root / "data" / "sample" / "sample_speeches.csv"
        assert path.exists(), f"Sample data not found at {path}"

    def test_sample_data_loadable(self, project_root: Path) -> None:
        """Sample speeches should load as a valid DataFrame."""
        path = project_root / "data" / "sample" / "sample_speeches.csv"
        df = pd.read_csv(path)
        assert len(df) > 0
        assert "text" in df.columns
        assert "date" in df.columns
        assert "source" in df.columns

    def test_sample_data_has_all_sources(self, project_root: Path) -> None:
        """Sample data should include documents from all three banks."""
        path = project_root / "data" / "sample" / "sample_speeches.csv"
        df = pd.read_csv(path)
        sources = set(df["source"].unique())
        assert sources == {"boe", "fed", "ecb"}

    def test_sample_topics_exist(self, project_root: Path) -> None:
        """Pre-computed topic assignments should exist."""
        path = project_root / "data" / "sample" / "sample_topics.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert "topic_id" in df.columns

    def test_sample_sentiment_exists(self, project_root: Path) -> None:
        """Pre-computed sentiment scores should exist."""
        path = project_root / "data" / "sample" / "sample_sentiment.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert "sentiment_compound" in df.columns
