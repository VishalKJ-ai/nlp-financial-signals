"""Preprocessing for FOMC press conference transcripts.

Implements the data preparation protocol from Chapter 3 of the
dissertation.  Takes the speaker-turn corpus produced by
``fomc_presser_scraper`` and yields the analysis units for the
pipeline: cleaned sentences drawn from the Chair's Q&A answers.

Cleaning is deliberately conservative: spoken-language disfluencies
(false starts, immediate word repetitions, leading fillers) are
removed, but hedged phrasing is retained because hedging itself
carries policy signal in central bank communication.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class PresserPreprocessor:
    """Converts Chair speaker turns into cleaned sentence-level units.

    Attributes:
        min_sentence_words: Sentences shorter than this are dropped
            (fragments, acknowledgements such as "Thank you.").
        include_opening_statement: Whether to keep the prepared opening
            statement alongside Q&A answers.  Chapter 3 restricts the
            corpus to answers; the flag exists for robustness checks.
    """

    # Abbreviations that end with a period but do not end a sentence.
    _ABBREVIATIONS = (
        "Mr.", "Ms.", "Mrs.", "Dr.", "St.", "vs.", "etc.", "e.g.", "i.e.",
        "U.S.", "U.K.", "p.m.", "a.m.", "No.", "Gov.", "Sen.", "Rep.",
    )

    # Immediate word repetition, e.g. "the the economy" -> "the economy".
    _WORD_REPEAT_RE = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)

    # False starts rendered with dashes, e.g. "I— I would say".
    _FALSE_START_RE = re.compile(r"\b\w+[—–-]\s+(?=\w)")

    # Leading spoken fillers at the start of a sentence.
    _LEADING_FILLER_RE = re.compile(
        r"^(?:So|Well|You know|I mean|Look|Again|Right)[,.]\s+", re.IGNORECASE
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise from the dissertation configuration.

        Args:
            config: Configuration dict; the ``corpus`` sub-key is used.
        """
        corpus_cfg = config.get("corpus", {})
        self.min_sentence_words: int = int(corpus_cfg.get("min_sentence_words", 6))
        self.include_opening_statement: bool = bool(
            corpus_cfg.get("include_opening_statement", False)
        )

    # ------------------------------------------------------------------
    # Sentence segmentation
    # ------------------------------------------------------------------

    def split_sentences(self, text: str) -> List[str]:
        """Split a turn into sentences, respecting common abbreviations.

        Args:
            text: Cleaned single-spaced turn text.

        Returns:
            List of sentence strings.
        """
        if not text:
            return []
        # Protect abbreviation periods with a placeholder.
        protected = text
        for abbr in self._ABBREVIATIONS:
            protected = protected.replace(abbr, abbr.replace(".", "․"))
        # Protect decimal points (e.g. "2.5 percent").
        protected = re.sub(r"(\d)\.(\d)", r"\1․\2", protected)

        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", protected)
        return [p.replace("․", ".").strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Disfluency cleaning
    # ------------------------------------------------------------------

    def clean_sentence(self, sentence: str) -> str:
        """Remove spoken-language disfluencies from one sentence.

        Args:
            sentence: Raw sentence text.

        Returns:
            Cleaned sentence text.
        """
        cleaned = self._FALSE_START_RE.sub("", sentence)
        cleaned = self._WORD_REPEAT_RE.sub(r"\1", cleaned)
        cleaned = self._LEADING_FILLER_RE.sub("", cleaned)
        cleaned = " ".join(cleaned.split())
        # Restore sentence-initial capitalisation after filler removal.
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def build_units(self, turns: pd.DataFrame) -> pd.DataFrame:
        """Build the sentence-level analysis corpus from speaker turns.

        Args:
            turns: Speaker-turn DataFrame from the scraper with columns
                date, speaker, role, segment_type, turn_index, text.

        Returns:
            DataFrame with one row per retained sentence: date, chair,
            segment_type, turn_index, sentence_index, text, n_words.
        """
        segment_types = ["qa_answer"]
        if self.include_opening_statement:
            segment_types.append("opening_statement")

        chair_turns = turns[
            (turns["role"] == "chair")
            & (turns["segment_type"].isin(segment_types))
        ]
        logger.info(
            "Building sentence units from %d Chair turns across %d meetings",
            len(chair_turns), chair_turns["date"].nunique(),
        )

        rows: List[Dict[str, Any]] = []
        for _, turn in chair_turns.iterrows():
            for sent_idx, sentence in enumerate(self.split_sentences(turn["text"])):
                cleaned = self.clean_sentence(sentence)
                n_words = len(cleaned.split())
                if n_words < self.min_sentence_words:
                    continue
                rows.append({
                    "date": turn["date"],
                    "chair": turn["speaker"],
                    "segment_type": turn["segment_type"],
                    "turn_index": turn["turn_index"],
                    "sentence_index": sent_idx,
                    "text": cleaned,
                    "n_words": n_words,
                })

        units = pd.DataFrame(rows)
        logger.info(
            "Corpus: %d sentences, %d words, %d meetings (%s to %s)",
            len(units), units["n_words"].sum(), units["date"].nunique(),
            units["date"].min().date(), units["date"].max().date(),
        )
        return units

    def corpus_statistics(self, turns: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
        """Produce the per-year corpus statistics table for Chapter 4.

        Args:
            turns: Full speaker-turn DataFrame.
            units: Sentence-level corpus from :meth:`build_units`.

        Returns:
            Per-year DataFrame of meetings, Chair turns, retained
            sentences, and word counts.
        """
        turns = turns.assign(year=turns["date"].dt.year)
        units = units.assign(year=units["date"].dt.year)

        stats = pd.DataFrame({
            "meetings": turns.groupby("year")["date"].nunique(),
            "chair_qa_turns": turns[
                (turns["role"] == "chair")
                & (turns["segment_type"] == "qa_answer")
            ].groupby("year").size(),
            "sentences_retained": units.groupby("year").size(),
            "words_retained": units.groupby("year")["n_words"].sum(),
        }).fillna(0).astype(int)
        stats.loc["Total"] = stats.sum()
        return stats.reset_index(names="year")
