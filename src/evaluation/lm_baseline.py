"""Loughran-McDonald dictionary benchmark (Evaluation Arm 2).

Scores the sentence corpus with the finance-specific LM word lists
(Loughran and McDonald, 2011) and compares the resulting series with
the FinBERT scores at both document and cluster level.  Chapter 3
defines success as a statistically significant positive correlation:
a zero or negative correlation would falsify the pipeline's baseline
validity.

The master dictionary CSV must contain ``Word``, ``Positive`` and
``Negative`` columns (non-zero values flag list membership), which is
the format distributed at https://sraf.nd.edu.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


class LMScorer:
    """Sentence scorer using the Loughran-McDonald word lists.

    Attributes:
        positive: Set of LM positive words (uppercase).
        negative: Set of LM negative words (uppercase).
    """

    def __init__(self, dictionary_path: Path) -> None:
        """Load the LM master dictionary.

        Args:
            dictionary_path: Path to the master dictionary CSV.
        """
        table = pd.read_csv(dictionary_path)
        table.columns = [c.strip().lower() for c in table.columns]
        self.positive: Set[str] = set(
            table.loc[table["positive"] != 0, "word"].str.upper()
        )
        self.negative: Set[str] = set(
            table.loc[table["negative"] != 0, "word"].str.upper()
        )
        logger.info("LM dictionary: %d positive, %d negative words",
                    len(self.positive), len(self.negative))

    def score(self, text: str) -> Tuple[float, float, float]:
        """Score one text unit with the LM lists.

        Args:
            text: Sentence or document text.

        Returns:
            Tuple of (positive share, negative share, compound), where
            shares are proportions of total tokens and compound is
            (pos - neg) / (pos + neg), zero when no sentiment words.
        """
        tokens = [t.upper() for t in _TOKEN_RE.findall(text)]
        if not tokens:
            return 0.0, 0.0, 0.0
        pos = sum(token in self.positive for token in tokens)
        neg = sum(token in self.negative for token in tokens)
        polar = pos + neg
        compound = (pos - neg) / polar if polar else 0.0
        return pos / len(tokens), neg / len(tokens), compound

    def score_frame(self, frame: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Score every row of a DataFrame, adding lm_* columns."""
        scores = frame[text_col].apply(self.score)
        frame = frame.copy()
        frame["lm_positive"] = scores.str[0]
        frame["lm_negative"] = scores.str[1]
        frame["lm_compound"] = scores.str[2]
        return frame


def _correlate(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Pearson and Spearman correlations plus directional agreement."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    agreement = float((np.sign(x) == np.sign(y)).mean())
    return {
        "n": int(mask.sum()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "directional_agreement": agreement,
    }


def run_benchmark(config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Run the full LM benchmark comparison (Arm 2).

    Args:
        config: Dissertation configuration.
        project_root: Repository root for resolving paths.

    Returns:
        Comparison table with one row per aggregation level.
    """
    results_dir = project_root / config["output"]["results_dir"]
    tables_dir = project_root / config["output"]["tables_dir"]
    dictionary_path = project_root / config["evaluation"]["lm_dictionary_path"]

    merged = pd.read_parquet(results_dir / "sentences_scored_assigned.parquet")
    scorer = LMScorer(dictionary_path)
    merged = scorer.score_frame(merged)
    merged.to_parquet(results_dir / "sentences_lm_scored.parquet", index=False)

    rows = []

    # Sentence level.
    row = _correlate(merged["sentiment_compound"], merged["lm_compound"])
    row["level"] = "sentence"
    rows.append(row)

    # Document (meeting) level.
    doc = merged.groupby("date")[["sentiment_compound", "lm_compound"]].mean()
    row = _correlate(doc["sentiment_compound"], doc["lm_compound"])
    row["level"] = "document"
    rows.append(row)

    # Cluster (meeting x topic) level, excluding outliers.
    clustered = merged[merged["topic"] != -1]
    cluster = clustered.groupby(["date", "topic"])[
        ["sentiment_compound", "lm_compound"]
    ].mean()
    row = _correlate(cluster["sentiment_compound"], cluster["lm_compound"])
    row["level"] = "cluster"
    rows.append(row)

    table = pd.DataFrame(rows)[
        ["level", "n", "pearson_r", "pearson_p", "spearman_r",
         "spearman_p", "directional_agreement"]
    ]
    table.to_csv(tables_dir / "lm_benchmark.csv", index=False)
    logger.info("LM benchmark:\n%s", table.to_string(index=False))
    return table


def main() -> None:
    """CLI entry point."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    with open(project_root / "config/dissertation.yaml") as handle:
        config = yaml.safe_load(handle)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    run_benchmark(config, project_root)


if __name__ == "__main__":
    main()
