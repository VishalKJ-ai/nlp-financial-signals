"""Text preprocessing module for central bank communications.

Handles cleaning, tokenisation, chunking, and document preparation
for downstream topic modelling and sentiment analysis.  Designed to
handle the specific characteristics of central bank text: formal
language, legal disclaimers, embedded tables, and footnotes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocesses raw central bank documents for NLP analysis.

    Applies a multi-step cleaning pipeline to remove boilerplate,
    normalise text, chunk long documents into manageable segments,
    and extract metadata for downstream use.

    Attributes:
        min_doc_length: Minimum characters for a document to be kept.
        max_doc_length: Maximum characters per document.
        chunk_size: Target chunk size in approximate tokens.
        chunk_overlap: Overlap between consecutive chunks.
        remove_boilerplate: Whether to strip headers/footers/disclaimers.
    """

    # Common boilerplate patterns in central bank publications
    _BOILERPLATE_PATTERNS = [
        r"(?i)all rights reserved\.?",
        r"(?i)this (?:speech|publication|document) is available on [\w\s]+ website",
        r"(?i)views expressed (?:here|in this|are those of).*?(?:do not necessarily|and not)",
        r"(?i)copyright ©?\s*\d{4}",
        r"(?i)for (?:further|more) information,? (?:please )?(?:contact|visit|see)",
        r"(?i)press (?:office|enquiries|contact)",
        r"(?i)(?:^|\n)page \d+ of \d+",
        r"(?i)(?:^|\n)\d+\s*$",  # Standalone page numbers
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the preprocessor from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The
                ``preprocessing`` sub-key is used.
        """
        prep_cfg = config.get("preprocessing", {})
        self.min_doc_length: int = prep_cfg.get("min_doc_length", 100)
        self.max_doc_length: int = prep_cfg.get("max_doc_length", 50000)
        self.chunk_size: int = prep_cfg.get("chunk_size", 512)
        self.chunk_overlap: int = prep_cfg.get("chunk_overlap", 64)
        self.remove_boilerplate: bool = prep_cfg.get("remove_boilerplate", True)
        self.lowercase: bool = prep_cfg.get("lowercase", False)

        self._compiled_patterns = [
            re.compile(p) for p in self._BOILERPLATE_PATTERNS
        ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full preprocessing pipeline to a documents DataFrame.

        Args:
            df: DataFrame with at least a 'text' column.

        Returns:
            Cleaned DataFrame with additional 'clean_text' and
            'word_count' columns.  Documents below the minimum
            length threshold are removed.
        """
        logger.info("Preprocessing %d documents", len(df))

        df = df.copy()
        df["clean_text"] = df["text"].apply(self.clean)
        df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))

        # Filter by document length
        initial_count = len(df)
        df = df[df["clean_text"].str.len() >= self.min_doc_length].reset_index(drop=True)
        removed = initial_count - len(df)
        if removed > 0:
            logger.info("Removed %d documents below minimum length (%d chars)",
                        removed, self.min_doc_length)

        logger.info("Preprocessing complete: %d documents, median %d words",
                     len(df), int(df["word_count"].median()) if len(df) > 0 else 0)
        return df

    def clean(self, text: str) -> str:
        """Clean a single document text.

        Applies the following steps in order:
        1. Unicode normalisation (NFKC)
        2. Boilerplate removal (if enabled)
        3. Footnote and reference removal
        4. Table artefact cleaning
        5. Whitespace normalisation
        6. Optional lowercasing

        Args:
            text: Raw document text.

        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Unicode normalisation
        text = unicodedata.normalize("NFKC", text)

        # Replace non-breaking spaces and special whitespace
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")  # Zero-width space

        # Remove boilerplate
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)

        # Remove footnote markers [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Clean table artefacts (pipes, excessive dashes)
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"-{3,}", "", text)
        text = re.sub(r"={3,}", "", text)

        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        # Truncate excessively long documents
        if len(text) > self.max_doc_length:
            text = text[: self.max_doc_length]
            logger.debug("Truncated document to %d characters", self.max_doc_length)

        if self.lowercase:
            text = text.lower()

        return text.strip()

    def _remove_boilerplate(self, text: str) -> str:
        """Remove known boilerplate patterns from text.

        Args:
            text: Document text.

        Returns:
            Text with boilerplate patterns removed.
        """
        for pattern in self._compiled_patterns:
            text = pattern.sub("", text)
        return text

    def chunk_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split long documents into overlapping chunks.

        Each chunk inherits the metadata (date, source, speaker, etc.)
        from its parent document.  A ``chunk_id`` column is added to
        track the chunk index within each document.

        Args:
            df: DataFrame with 'clean_text' column and metadata.

        Returns:
            DataFrame with one row per chunk.
        """
        logger.info("Chunking %d documents (chunk_size=%d, overlap=%d)",
                     len(df), self.chunk_size, self.chunk_overlap)

        chunked_rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            text = row.get("clean_text", row.get("text", ""))
            chunks = self._split_into_chunks(text)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_row = row.to_dict()
                chunk_row["clean_text"] = chunk
                chunk_row["chunk_id"] = chunk_idx
                chunk_row["word_count"] = len(chunk.split())
                chunked_rows.append(chunk_row)

        result = pd.DataFrame(chunked_rows)
        logger.info("Chunking produced %d chunks from %d documents",
                     len(result), len(df))
        return result

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks by approximate token count.

        Uses a simple word-based tokenisation for splitting.  Chunks
        are created at paragraph boundaries where possible to maintain
        semantic coherence.

        Args:
            text: Document text to split.

        Returns:
            List of text chunks.
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break
            start += step

        return chunks

    def prepare_for_topics(self, df: pd.DataFrame) -> List[str]:
        """Extract clean text documents for topic modelling.

        Args:
            df: Preprocessed DataFrame with 'clean_text' column.

        Returns:
            List of document strings suitable for BERTopic.
        """
        docs = df["clean_text"].tolist()
        docs = [str(d) for d in docs if d and len(str(d).split()) >= 10]
        logger.info("Prepared %d documents for topic modelling", len(docs))
        return docs

    def extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and standardise document metadata.

        Ensures consistent date parsing, source labelling, and
        speaker name normalisation across all three central banks.

        Args:
            df: DataFrame with raw metadata columns.

        Returns:
            DataFrame with standardised metadata.
        """
        df = df.copy()

        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Standardise source names
        source_map = {
            "boe": "Bank of England",
            "fed": "Federal Reserve",
            "ecb": "European Central Bank",
        }
        if "source" in df.columns:
            df["source_full"] = df["source"].map(source_map).fillna(df["source"])

        # Extract year and month for temporal analysis
        if "date" in df.columns:
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["year_month"] = df["date"].dt.to_period("M")

        return df

    @staticmethod
    def merge_sources(
        boe_df: pd.DataFrame,
        fed_df: pd.DataFrame,
        ecb_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge documents from all three central banks into one DataFrame.

        Args:
            boe_df: Bank of England documents.
            fed_df: Federal Reserve documents.
            ecb_df: ECB documents.

        Returns:
            Combined DataFrame sorted by date.
        """
        combined = pd.concat([boe_df, fed_df, ecb_df], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        logger.info("Merged %d documents from 3 central banks", len(combined))
        return combined

    def save(self, df: pd.DataFrame, output_path: str) -> Path:
        """Save preprocessed documents to Parquet format.

        Args:
            df: Preprocessed DataFrame.
            output_path: Path to save the file.

        Returns:
            Path to the saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Period columns to string for Parquet compatibility
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col]):
                df[col] = df[col].astype(str)

        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d preprocessed documents to %s", len(df), path)
        return path

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load preprocessed documents from Parquet.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame of preprocessed documents.
        """
        df = pd.read_parquet(path, engine="pyarrow")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
