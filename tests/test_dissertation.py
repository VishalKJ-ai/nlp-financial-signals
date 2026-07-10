"""Tests for the dissertation-specific modules.

Covers press conference parsing/preprocessing, LM dictionary scoring,
and market event-window computation.  Pipeline stages that require
model downloads are exercised separately via the CLI.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.presser_preprocessor import PresserPreprocessor
from src.evaluation.market_validation import event_window_change


@pytest.fixture
def preprocessor():
    return PresserPreprocessor({"corpus": {"min_sentence_words": 4}})


class TestSentenceSplitting:
    def test_basic_split(self, preprocessor):
        text = ("Inflation remains elevated. We are strongly committed to "
                "returning inflation to our 2 percent objective. Thank you.")
        sentences = preprocessor.split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Inflation remains elevated."

    def test_abbreviations_not_split(self, preprocessor):
        text = "Mr. Powell noted that U.S. growth slowed. Markets reacted."
        sentences = preprocessor.split_sentences(text)
        assert len(sentences) == 2

    def test_decimals_not_split(self, preprocessor):
        text = "Inflation is running at 2.5 percent. That is above target."
        sentences = preprocessor.split_sentences(text)
        assert len(sentences) == 2
        assert "2.5 percent" in sentences[0]


class TestDisfluencyCleaning:
    def test_word_repetition_collapsed(self, preprocessor):
        assert preprocessor.clean_sentence(
            "We we are committed to the the mandate"
        ) == "We are committed to the mandate"

    def test_leading_filler_removed(self, preprocessor):
        cleaned = preprocessor.clean_sentence(
            "So, we will continue to monitor incoming data")
        assert cleaned == "We will continue to monitor incoming data"

    def test_hedging_preserved(self, preprocessor):
        sentence = "We think it may be appropriate to slow the pace of increases"
        assert preprocessor.clean_sentence(sentence) == sentence


class TestBuildUnits:
    def test_restricts_to_chair_answers(self, preprocessor):
        turns = pd.DataFrame({
            "date": pd.to_datetime(["2022-06-15"] * 3),
            "speaker": ["CHAIR POWELL", "STEVE LIESMAN", "CHAIR POWELL"],
            "role": ["chair", "journalist", "chair"],
            "segment_type": ["opening_statement", "question", "qa_answer"],
            "turn_index": [0, 1, 2],
            "text": [
                "Good afternoon. Inflation remains far too high today.",
                "What will you do about inflation over the next year?",
                "We will raise rates until inflation is clearly moving down.",
            ],
        })
        units = preprocessor.build_units(turns)
        assert set(units["segment_type"]) == {"qa_answer"}
        assert len(units) == 1
        assert units.iloc[0]["text"].startswith("We will raise rates")


class TestPageNoiseRemoval:
    def test_inline_running_header_excised(self):
        from src.data.fomc_presser_scraper import FomcPressConfScraper

        scraper = FomcPressConfScraper.__new__(FomcPressConfScraper)
        raw = ("It March 21, 2018 Chairman Powell’s Press Conference "
               "FINAL Page 7 of 22 would take significant increases in "
               "productivity.")
        cleaned = " ".join(scraper._clean_page_noise(raw).split())
        assert cleaned == "It would take significant increases in productivity."

    def test_header_without_space_excised(self):
        from src.data.fomc_presser_scraper import FomcPressConfScraper

        scraper = FomcPressConfScraper.__new__(FomcPressConfScraper)
        raw = ("we are committed September 20, 2017 Chair Yellen’s "
               "PressConference FINAL to taking the actions")
        cleaned = " ".join(scraper._clean_page_noise(raw).split())
        assert cleaned == "we are committed to taking the actions"

    def test_genuine_mention_preserved(self):
        from src.data.fomc_presser_scraper import FomcPressConfScraper

        scraper = FomcPressConfScraper.__new__(FomcPressConfScraper)
        raw = "We discussed this at the last press conference in June."
        assert scraper._clean_page_noise(raw) == raw


class TestEventWindowChange:
    def test_change_spans_event(self):
        index = pd.bdate_range("2022-06-10", periods=10)
        series = pd.Series(np.arange(10, dtype=float), index=index,
                           name="dgs2")
        events = pd.Series([pd.Timestamp("2022-06-15")])
        changes = event_window_change(series, events, window_days=1)
        assert len(changes) == 1
        # Last close before 15 Jun is 14 Jun; first close on/after 16 Jun.
        expected = series.loc["2022-06-16"] - series.loc["2022-06-14"]
        assert changes.iloc[0] == expected

    def test_event_outside_series_skipped(self):
        index = pd.bdate_range("2022-06-10", periods=5)
        series = pd.Series(np.ones(5), index=index, name="dgs2")
        events = pd.Series([pd.Timestamp("2030-01-01")])
        changes = event_window_change(series, events)
        assert changes.empty
