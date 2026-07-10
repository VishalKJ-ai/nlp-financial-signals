"""FinBERT-based sentiment analysis module for financial text.

Scores central bank communications using the ProsusAI/finbert model,
producing document- and paragraph-level sentiment scores.  Also
classifies the monetary policy stance as hawkish, dovish, or neutral.

In sample mode, uses pre-computed sentiment scores or a rule-based
fallback to avoid requiring model downloads.

References:
    FinBERT: https://huggingface.co/ProsusAI/finbert
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinBERTScorer:
    """Sentiment scorer using the FinBERT financial language model.

    Supports both full model inference and a lightweight rule-based
    fallback for sample/testing mode.  Produces three outputs per text
    segment: positive, negative, and neutral probabilities, plus a
    compound score (positive - negative).

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum token length for the tokenizer.
        batch_size: Batch size for model inference.
        device: Device for inference ('auto', 'cpu', or 'cuda').
        aggregation: Level of analysis ('document' or 'paragraph').
        use_precomputed: Whether to use pre-computed scores.
    """

    # Keywords used for rule-based sentiment in sample mode
    _HAWKISH_KEYWORDS = [
        "tightening", "restrictive", "raise", "increase", "hike",
        "inflation above target", "persistent inflation", "wage pressure",
        "overheating", "higher for longer", "vigilant", "upside risks",
        "price pressures", "elevated", "accelerated", "sticky",
    ]
    _DOVISH_KEYWORDS = [
        "easing", "accommodative", "cut", "reduce", "lower",
        "disinflation", "below target", "slowdown", "recession",
        "weakness", "downside risks", "supportive", "gradual",
        "moderation", "falling", "subdued",
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        use_precomputed: bool = False,
    ) -> None:
        """Initialise the FinBERT scorer.

        Args:
            config: Full pipeline configuration dict.  The ``sentiment``
                sub-key is used.
            use_precomputed: If True, use pre-computed scores or
                rule-based fallback instead of model inference.
        """
        sent_cfg = config.get("sentiment", {})
        self.model_name: str = sent_cfg.get("model_name", "ProsusAI/finbert")
        self.max_length: int = sent_cfg.get("max_length", 512)
        self.batch_size: int = sent_cfg.get("batch_size", 32)
        self.device: str = sent_cfg.get("device", "auto")
        self.aggregation: str = sent_cfg.get("aggregation", "paragraph")
        self.use_precomputed = use_precomputed

        self._model = None
        self._tokenizer = None

    def score_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all documents in the DataFrame with sentiment.

        Adds columns: sentiment_positive, sentiment_negative,
        sentiment_neutral, sentiment_compound, sentiment_label.

        Args:
            df: DataFrame with 'clean_text' or 'text' column.

        Returns:
            DataFrame with sentiment score columns added.
        """
        logger.info("Scoring %d documents for sentiment", len(df))
        df = df.copy()

        text_col = "clean_text" if "clean_text" in df.columns else "text"

        if self.use_precomputed:
            scores = df[text_col].apply(self._rule_based_score)
        else:
            self._load_model()
            texts = df[text_col].tolist()
            scores = pd.Series(self._batch_score(texts))

        df["sentiment_positive"] = scores.apply(lambda s: s["positive"])
        df["sentiment_negative"] = scores.apply(lambda s: s["negative"])
        df["sentiment_neutral"] = scores.apply(lambda s: s["neutral"])
        df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
        df["sentiment_label"] = scores.apply(lambda s: s["label"])

        logger.info(
            "Sentiment distribution: %.1f%% positive, %.1f%% negative, %.1f%% neutral",
            (df["sentiment_label"] == "positive").mean() * 100,
            (df["sentiment_label"] == "negative").mean() * 100,
            (df["sentiment_label"] == "neutral").mean() * 100,
        )
        return df

    def score_paragraphs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score each paragraph within each document separately.

        Splits documents into paragraphs and scores each one, then
        returns a long-format DataFrame with one row per paragraph.

        Args:
            df: DataFrame with 'clean_text' or 'text' column.

        Returns:
            Long-format DataFrame with paragraph-level scores.
        """
        logger.info("Scoring paragraphs for %d documents", len(df))
        text_col = "clean_text" if "clean_text" in df.columns else "text"

        paragraph_records: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            paragraphs = self._split_paragraphs(row[text_col])
            for para_idx, para in enumerate(paragraphs):
                if len(para.split()) < 5:
                    continue

                if self.use_precomputed:
                    scores = self._rule_based_score(para)
                else:
                    scores = self._score_single(para)

                record = row.to_dict()
                record["paragraph_id"] = para_idx
                record["paragraph_text"] = para
                record["sentiment_positive"] = scores["positive"]
                record["sentiment_negative"] = scores["negative"]
                record["sentiment_neutral"] = scores["neutral"]
                record["sentiment_compound"] = scores["compound"]
                record["sentiment_label"] = scores["label"]
                paragraph_records.append(record)

        result = pd.DataFrame(paragraph_records)
        logger.info("Scored %d paragraphs", len(result))
        return result

    def classify_stance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify monetary policy stance as hawkish, dovish, or neutral.

        Uses a combination of sentiment scores and keyword matching
        to determine the policy stance of each document.

        Args:
            df: DataFrame with sentiment score columns.

        Returns:
            DataFrame with 'stance' and 'stance_score' columns.
        """
        logger.info("Classifying monetary policy stance for %d documents", len(df))
        df = df.copy()

        text_col = "clean_text" if "clean_text" in df.columns else "text"

        stances = []
        stance_scores = []

        for _, row in df.iterrows():
            text = str(row[text_col]).lower()
            compound = row.get("sentiment_compound", 0.0)

            # Count hawkish and dovish keywords
            hawkish_count = sum(1 for kw in self._HAWKISH_KEYWORDS if kw in text)
            dovish_count = sum(1 for kw in self._DOVISH_KEYWORDS if kw in text)

            # Combine keyword signal with sentiment
            keyword_score = (hawkish_count - dovish_count) / max(
                hawkish_count + dovish_count, 1
            )

            # Weighted combination: 60% keywords, 40% sentiment
            stance_score = 0.6 * keyword_score + 0.4 * compound
            stance_scores.append(stance_score)

            if stance_score > 0.15:
                stances.append("hawkish")
            elif stance_score < -0.15:
                stances.append("dovish")
            else:
                stances.append("neutral")

        df["stance"] = stances
        df["stance_score"] = stance_scores

        logger.info(
            "Stance distribution: %d hawkish, %d dovish, %d neutral",
            stances.count("hawkish"),
            stances.count("dovish"),
            stances.count("neutral"),
        )
        return df

    def _load_model(self) -> None:
        """Lazy-load the FinBERT model and tokenizer."""
        if self._model is not None:
            return

        logger.info("Loading FinBERT model: %s", self.model_name)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # yiyanghkust/finbert-tone predates the Auto* config format
            # (no model_type key, vocab.txt-only tokenizer), so fall back
            # to the explicit BERT classes when auto-detection fails.
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except ValueError:
                from transformers import BertTokenizerFast

                self._tokenizer = BertTokenizerFast.from_pretrained(
                    self.model_name
                )
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
            except ValueError:
                from transformers import BertForSequenceClassification

                self._model = BertForSequenceClassification.from_pretrained(
                    self.model_name
                )

            # Device selection
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self.device

            self._model.to(self._device)
            self._model.eval()
            logger.info("FinBERT loaded on %s", self._device)

        except ImportError as e:
            logger.error("transformers/torch not installed: %s", e)
            raise

    def _batch_score(self, texts: List[str]) -> List[Dict[str, float]]:
        """Score a batch of texts using the loaded FinBERT model.

        Args:
            texts: List of text strings.

        Returns:
            List of score dictionaries.
        """
        import torch

        results: List[Dict[str, float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            # Truncate long texts
            batch = [t[: self.max_length * 4] for t in batch]  # Rough char limit

            encodings = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            # Label order differs between FinBERT variants (ProsusAI:
            # positive/negative/neutral; yiyanghkust/finbert-tone:
            # Neutral/Positive/Negative), so map via the model config.
            id2label = {
                i: label.lower()
                for i, label in self._model.config.id2label.items()
            }
            for prob in probs:
                by_label = {id2label[i]: float(p) for i, p in enumerate(prob)}
                positive = by_label.get("positive", 0.0)
                negative = by_label.get("negative", 0.0)
                neutral = by_label.get("neutral", 0.0)
                compound = positive - negative
                label = id2label[int(prob.argmax())]
                results.append({
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "compound": compound,
                    "label": label,
                })

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.debug("Scored %d / %d texts", i + self.batch_size, len(texts))

        return results

    def _score_single(self, text: str) -> Dict[str, float]:
        """Score a single text string.

        Args:
            text: Text to score.

        Returns:
            Score dictionary.
        """
        results = self._batch_score([text])
        return results[0]

    def _rule_based_score(self, text: str) -> Dict[str, float]:
        """Compute sentiment using keyword-based rules (sample mode).

        Args:
            text: Text to score.

        Returns:
            Score dictionary with positive, negative, neutral,
            compound, and label keys.
        """
        text_lower = str(text).lower()

        pos_count = sum(1 for kw in self._HAWKISH_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in self._DOVISH_KEYWORDS if kw in text_lower)
        total = max(pos_count + neg_count, 1)

        # Add noise for variety (deterministic based on text length)
        np.random.seed(len(text_lower) % 10000)
        noise = np.random.normal(0, 0.05)

        positive = np.clip(pos_count / total + noise, 0, 1)
        negative = np.clip(neg_count / total + noise * 0.5, 0, 1)
        neutral = np.clip(1.0 - positive - negative, 0, 1)

        # Renormalise
        total_prob = positive + negative + neutral
        positive /= total_prob
        negative /= total_prob
        neutral /= total_prob

        compound = float(positive - negative)
        if positive > negative and positive > neutral:
            label = "positive"
        elif negative > positive and negative > neutral:
            label = "negative"
        else:
            label = "neutral"

        return {
            "positive": float(positive),
            "negative": float(negative),
            "neutral": float(neutral),
            "compound": compound,
            "label": label,
        }

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Document text.

        Returns:
            List of paragraph strings.
        """
        if not text:
            return []
        paragraphs = re.split(r"\n\n+", str(text))
        return [p.strip() for p in paragraphs if p.strip()]

    def aggregate_sentiment(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = None,
    ) -> pd.DataFrame:
        """Aggregate sentiment scores by specified grouping columns.

        Args:
            df: DataFrame with sentiment columns.
            group_cols: Columns to group by (default: ['source', 'year_month']).

        Returns:
            Aggregated DataFrame with mean, std sentiment scores.
        """
        if group_cols is None:
            group_cols = ["source"]
            if "year_month" in df.columns:
                group_cols.append("year_month")
            elif "date" in df.columns:
                df = df.copy()
                df["year_month"] = df["date"].dt.to_period("M")
                group_cols.append("year_month")

        sent_cols = [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_compound",
        ]
        existing_cols = [c for c in sent_cols if c in df.columns]

        agg_dict = {col: ["mean", "std", "count"] for col in existing_cols}
        result = df.groupby(group_cols).agg(agg_dict)
        result.columns = ["_".join(col).strip() for col in result.columns]
        result = result.reset_index()

        logger.info("Aggregated sentiment into %d groups", len(result))
        return result
