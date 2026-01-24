"""Tests for data collection and preprocessing modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for the TextPreprocessor class."""

    @pytest.fixture
    def preprocessor(self, config: dict) -> TextPreprocessor:
        """Create a preprocessor instance from test config."""
        return TextPreprocessor(config)

    def test_clean_removes_boilerplate(
        self, preprocessor: TextPreprocessor, sample_text: str
    ) -> None:
        """Boilerplate phrases should be removed from cleaned text."""
        cleaned = preprocessor.clean(sample_text)
        assert "All rights reserved" not in cleaned
        assert "press office" not in cleaned
        assert "Page 1 of 3" not in cleaned

    def test_clean_removes_urls(
        self, preprocessor: TextPreprocessor, sample_text: str
    ) -> None:
        """URLs should be stripped from cleaned text."""
        cleaned = preprocessor.clean(sample_text)
        assert "https://" not in cleaned

    def test_clean_removes_footnote_markers(
        self, preprocessor: TextPreprocessor, sample_text: str
    ) -> None:
        """Footnote markers like [1] should be removed."""
        cleaned = preprocessor.clean(sample_text)
        assert "[1]" not in cleaned

    def test_clean_preserves_content(
        self, preprocessor: TextPreprocessor, sample_text: str
    ) -> None:
        """Core content should be preserved after cleaning."""
        cleaned = preprocessor.clean(sample_text)
        assert "GDP growth" in cleaned
        assert "inflation" in cleaned
        assert "monetary policy" in cleaned

    def test_clean_empty_input(self, preprocessor: TextPreprocessor) -> None:
        """Empty or None inputs should return empty string."""
        assert preprocessor.clean("") == ""
        assert preprocessor.clean(None) == ""

    def test_process_filters_short_documents(
        self, preprocessor: TextPreprocessor, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Documents below minimum length should be removed."""
        # Add a very short document
        short_doc = pd.DataFrame([{
            "date": "2024-01-01",
            "title": "Short",
            "speaker": "Test",
            "source": "boe",
            "doc_type": "speech",
            "text": "Hi.",
            "url": "https://example.com",
        }])
        df = pd.concat([sample_speeches_df, short_doc], ignore_index=True)

        result = preprocessor.process(df)
        assert len(result) == len(sample_speeches_df)

    def test_process_adds_word_count(
        self, preprocessor: TextPreprocessor, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Processing should add a word_count column."""
        result = preprocessor.process(sample_speeches_df)
        assert "word_count" in result.columns
        assert "clean_text" in result.columns
        assert (result["word_count"] > 0).all()


class TestChunking:
    """Tests for document chunking."""

    @pytest.fixture
    def preprocessor(self, config: dict) -> TextPreprocessor:
        """Create a preprocessor with small chunk size for testing."""
        config = config.copy()
        config["preprocessing"] = {
            "chunk_size": 20,
            "chunk_overlap": 5,
            "min_doc_length": 10,
        }
        return TextPreprocessor(config)

    def test_short_document_single_chunk(
        self, preprocessor: TextPreprocessor
    ) -> None:
        """Documents shorter than chunk_size should produce one chunk."""
        df = pd.DataFrame([{
            "clean_text": "This is a short document with few words.",
            "date": "2024-01-01",
        }])
        result = preprocessor.chunk_documents(df)
        assert len(result) == 1
        assert result.iloc[0]["chunk_id"] == 0

    def test_long_document_multiple_chunks(
        self, preprocessor: TextPreprocessor
    ) -> None:
        """Long documents should produce multiple overlapping chunks."""
        long_text = " ".join(["word"] * 100)
        df = pd.DataFrame([{
            "clean_text": long_text,
            "date": "2024-01-01",
        }])
        result = preprocessor.chunk_documents(df)
        assert len(result) > 1
        assert result["chunk_id"].max() >= 1

    def test_chunk_metadata_preserved(
        self, preprocessor: TextPreprocessor
    ) -> None:
        """Metadata columns should be preserved in each chunk."""
        long_text = " ".join(["word"] * 100)
        df = pd.DataFrame([{
            "clean_text": long_text,
            "date": "2024-01-01",
            "source": "boe",
            "speaker": "Andrew Bailey",
        }])
        result = preprocessor.chunk_documents(df)
        assert (result["source"] == "boe").all()
        assert (result["speaker"] == "Andrew Bailey").all()


class TestMetadata:
    """Tests for metadata extraction."""

    @pytest.fixture
    def preprocessor(self, config: dict) -> TextPreprocessor:
        return TextPreprocessor(config)

    def test_extract_metadata_adds_year_month(
        self, preprocessor: TextPreprocessor, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Metadata extraction should add year and month columns."""
        result = preprocessor.extract_metadata(sample_speeches_df)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "source_full" in result.columns

    def test_source_full_names(
        self, preprocessor: TextPreprocessor, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Source codes should be expanded to full names."""
        result = preprocessor.extract_metadata(sample_speeches_df)
        expected = {"Bank of England", "Federal Reserve", "European Central Bank"}
        assert set(result["source_full"].unique()) == expected


class TestMergeAndPersistence:
    """Tests for source merging and file I/O."""

    def test_merge_sources(self, sample_speeches_df: pd.DataFrame) -> None:
        """Merging three source DataFrames should concatenate and sort."""
        boe = sample_speeches_df[sample_speeches_df["source"] == "boe"]
        fed = sample_speeches_df[sample_speeches_df["source"] == "fed"]
        ecb = sample_speeches_df[sample_speeches_df["source"] == "ecb"]

        merged = TextPreprocessor.merge_sources(boe, fed, ecb)
        assert len(merged) == len(sample_speeches_df)
        # Should be sorted by date
        dates = merged["date"].tolist()
        assert dates == sorted(dates)

    def test_save_and_load_roundtrip(
        self, config: dict, sample_speeches_df: pd.DataFrame
    ) -> None:
        """Saving and loading should preserve the data."""
        preprocessor = TextPreprocessor(config)
        processed = preprocessor.process(sample_speeches_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_docs.parquet")
            preprocessor.save(processed, path)

            loaded = TextPreprocessor.load(path)
            assert len(loaded) == len(processed)
            assert list(loaded.columns) == list(processed.columns)
