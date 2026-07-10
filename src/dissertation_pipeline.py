"""Dissertation pipeline: cluster-level sentiment from FOMC press conferences.

Implements the experimental design in Chapter 3 of the dissertation,
distinct from the legacy multi-bank pipeline in ``src.pipeline``.  The
pipeline is staged so that each expensive artefact (corpus, embeddings,
topic model, sentiment scores) is computed once and persisted:

    python -m src.dissertation_pipeline --stage prepare
    python -m src.dissertation_pipeline --stage topics
    python -m src.dissertation_pipeline --stage grid
    python -m src.dissertation_pipeline --stage sentiment
    python -m src.dissertation_pipeline --stage aggregate
    python -m src.dissertation_pipeline --stage all

Outputs land in ``outputs/dissertation/`` (tables, series) and
``models/dissertation/`` (embeddings, fitted topic model).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.presser_preprocessor import PresserPreprocessor
from src.sentiment.finbert_scorer import FinBERTScorer

logger = logging.getLogger(__name__)


# ── Configuration and reproducibility ─────────────────────────────────────────


def load_config(path: str = "config/dissertation.yaml") -> Dict[str, Any]:
    """Load the dissertation configuration file."""
    with open(PROJECT_ROOT / path) as handle:
        return yaml.safe_load(handle)


def set_seeds(seed: int) -> None:
    """Fix random seeds for reproducibility (Ch3 s3.3)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _resolve(cfg_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(cfg_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _outdirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create and return the output directories."""
    out = {key: _resolve(value) for key, value in config["output"].items()}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


# ── Stage: prepare ─────────────────────────────────────────────────────────────


def stage_prepare(config: Dict[str, Any]) -> pd.DataFrame:
    """Build the sentence-level corpus and corpus statistics table."""
    logger.info("--- Stage: prepare ---")
    corpus_cfg = config["corpus"]
    turns = pd.read_parquet(_resolve(corpus_cfg["turns_path"]))

    preprocessor = PresserPreprocessor(config)
    units = preprocessor.build_units(turns)
    units_path = _resolve(corpus_cfg["units_path"])
    units_path.parent.mkdir(parents=True, exist_ok=True)
    units.to_parquet(units_path, index=False)

    stats = preprocessor.corpus_statistics(turns, units)
    out = _outdirs(config)
    stats_path = out["tables_dir"] / "corpus_statistics.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("Wrote %d sentence units and corpus statistics to %s",
                len(units), stats_path)
    return units


def load_units(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the persisted sentence-level corpus."""
    return pd.read_parquet(_resolve(config["corpus"]["units_path"]))


# ── Embeddings (computed once, reused by topics and grid) ─────────────────────


def get_embeddings(config: Dict[str, Any], texts: List[str]) -> np.ndarray:
    """Compute or load cached sentence embeddings.

    The cache is invalidated if the corpus size or embedding model
    changes.

    Args:
        config: Dissertation configuration.
        texts: Sentence texts in corpus order.

    Returns:
        Array of shape (n_sentences, embedding_dim).
    """
    out = _outdirs(config)
    model_name = config["topics"]["embedding_model"]
    cache = out["models_dir"] / "embeddings.npy"
    meta_path = out["models_dir"] / "embeddings_meta.json"

    if cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n") == len(texts) and meta.get("model") == model_name:
            logger.info("Loading cached embeddings from %s", cache)
            return np.load(cache)

    logger.info("Computing embeddings for %d sentences with %s",
                len(texts), model_name)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(texts, show_progress_bar=True,
                                batch_size=128)
    np.save(cache, embeddings)
    meta_path.write_text(json.dumps({"n": len(texts), "model": model_name}))
    return embeddings


# ── Topic modelling ────────────────────────────────────────────────────────────


def build_topic_model(
    config: Dict[str, Any],
    n_neighbors: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Construct a BERTopic model with explicit UMAP/HDBSCAN parameters.

    Any parameter left as None falls back to the configured default,
    which lets the grid search vary one dimension at a time.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    topics_cfg = config["topics"]
    umap_cfg = topics_cfg["umap"]
    hdbscan_cfg = topics_cfg["hdbscan"]

    umap_model = UMAP(
        n_neighbors=n_neighbors or umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=seed if seed is not None else umap_cfg["random_state"],
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size or hdbscan_cfg["min_cluster_size"],
        min_samples=min_samples or hdbscan_cfg["min_samples"],
        metric=hdbscan_cfg["metric"],
        cluster_selection_method=hdbscan_cfg["cluster_selection_method"],
        prediction_data=False,
    )
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=5)
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=topics_cfg["top_n_words"],
        calculate_probabilities=False,
        verbose=False,
    )


def _topic_top_words(model, top_n: int = 10) -> Dict[int, List[str]]:
    """Extract the top words per topic, excluding the outlier topic."""
    return {
        topic_id: [word for word, _ in model.get_topic(topic_id)[:top_n]]
        for topic_id in model.get_topic_info()["Topic"]
        if topic_id != -1
    }


def compute_cv_coherence(
    topic_words: Dict[int, List[str]], texts: List[str]
) -> float:
    """Compute mean Cv coherence for a topic solution (Ch3 Arm 1).

    Args:
        topic_words: Mapping of topic id to its top words.
        texts: Corpus sentences used as the reference corpus.

    Returns:
        Mean Cv coherence across topics.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    tokenised = [re.findall(r"[a-z]{3,}", text.lower()) for text in texts]
    dictionary = Dictionary(tokenised)
    vocabulary = set(dictionary.token2id)
    # c_v requires every topic word to exist in the reference vocabulary;
    # bigrams from the CountVectorizer are split into unigrams first.
    topics = []
    for words in topic_words.values():
        flat = [token for word in words for token in word.split()]
        kept = [token for token in flat if token in vocabulary]
        if len(kept) >= 3:
            topics.append(kept[:10])
    model = CoherenceModel(topics=topics, texts=tokenised,
                           dictionary=dictionary, coherence="c_v",
                           processes=1)
    return float(model.get_coherence())


def stage_topics(config: Dict[str, Any]) -> None:
    """Fit the primary topic model and persist assignments and tables."""
    logger.info("--- Stage: topics ---")
    set_seeds(config["random_seed"])
    units = load_units(config)
    texts = units["text"].tolist()
    embeddings = get_embeddings(config, texts)

    model = build_topic_model(config)
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    units["topic"] = topics

    out = _outdirs(config)
    units.to_parquet(out["results_dir"] / "topic_assignments.parquet",
                     index=False)

    info = model.get_topic_info()
    top_words = _topic_top_words(model)
    info["top_words"] = info["Topic"].map(
        lambda t: ", ".join(top_words.get(t, []))
    )
    info.to_csv(out["tables_dir"] / "topic_overview.csv", index=False)

    coherence = compute_cv_coherence(top_words, texts)
    outlier_share = float((units["topic"] == -1).mean())
    summary = {
        "n_topics": int(len(top_words)),
        "outlier_share": outlier_share,
        "cv_coherence": coherence,
        "seed": config["random_seed"],
        "umap_n_neighbors": config["topics"]["umap"]["n_neighbors"],
        "hdbscan_min_cluster_size": config["topics"]["hdbscan"]["min_cluster_size"],
        "hdbscan_min_samples": config["topics"]["hdbscan"]["min_samples"],
    }
    (out["results_dir"] / "topic_model_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    model.save(str(out["models_dir"] / "bertopic_model"),
               serialization="safetensors", save_embedding_model=False)
    logger.info("Primary topic model: %d topics, %.1f%% outliers, Cv=%.3f",
                summary["n_topics"], outlier_share * 100, coherence)


def stage_grid(config: Dict[str, Any]) -> None:
    """Run the hyperparameter sensitivity grid (Ch3 robustness)."""
    logger.info("--- Stage: grid ---")
    set_seeds(config["random_seed"])
    units = load_units(config)
    texts = units["text"].tolist()
    embeddings = get_embeddings(config, texts)
    grid_cfg = config["topics"]["grid"]

    rows = []
    for n_neighbors in grid_cfg["umap_n_neighbors"]:
        for min_cluster_size in grid_cfg["hdbscan_min_cluster_size"]:
            for min_samples in grid_cfg["hdbscan_min_samples"]:
                set_seeds(config["random_seed"])
                model = build_topic_model(
                    config, n_neighbors=n_neighbors,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                topics, _ = model.fit_transform(texts, embeddings=embeddings)
                top_words = _topic_top_words(model)
                coherence = compute_cv_coherence(top_words, texts)
                row = {
                    "umap_n_neighbors": n_neighbors,
                    "hdbscan_min_cluster_size": min_cluster_size,
                    "hdbscan_min_samples": min_samples,
                    "n_topics": len(top_words),
                    "outlier_share": float(np.mean(np.array(topics) == -1)),
                    "cv_coherence": coherence,
                }
                rows.append(row)
                logger.info("grid %s -> %d topics, %.1f%% outliers, Cv=%.3f",
                            (n_neighbors, min_cluster_size, min_samples),
                            row["n_topics"], row["outlier_share"] * 100,
                            coherence)

    out = _outdirs(config)
    pd.DataFrame(rows).to_csv(out["tables_dir"] / "coherence_grid.csv",
                              index=False)
    logger.info("Grid complete: %d configurations", len(rows))


def stage_stability(config: Dict[str, Any]) -> None:
    """Refit the chosen configuration across seeds (Ch3 robustness)."""
    logger.info("--- Stage: stability ---")
    units = load_units(config)
    texts = units["text"].tolist()
    embeddings = get_embeddings(config, texts)

    rows = []
    for seed in config["evaluation"]["stability_seeds"]:
        set_seeds(seed)
        model = build_topic_model(config, seed=seed)
        topics, _ = model.fit_transform(texts, embeddings=embeddings)
        top_words = _topic_top_words(model)
        rows.append({
            "seed": seed,
            "n_topics": len(top_words),
            "outlier_share": float(np.mean(np.array(topics) == -1)),
            "cv_coherence": compute_cv_coherence(top_words, texts),
        })
        logger.info("seed %d -> %d topics", seed, rows[-1]["n_topics"])

    out = _outdirs(config)
    pd.DataFrame(rows).to_csv(out["tables_dir"] / "seed_stability.csv",
                              index=False)


# ── Sentiment ─────────────────────────────────────────────────────────────────


def stage_sentiment(config: Dict[str, Any]) -> None:
    """Score every sentence with FinBERT (Yang et al. 2020 variant)."""
    logger.info("--- Stage: sentiment ---")
    set_seeds(config["random_seed"])
    units = load_units(config)

    scorer = FinBERTScorer({"sentiment": config["sentiment"]})
    scored = scorer.score_documents(units)

    out = _outdirs(config)
    scored.to_parquet(out["results_dir"] / "sentence_sentiment.parquet",
                      index=False)
    logger.info("Scored %d sentences with %s", len(scored),
                config["sentiment"]["model_name"])


# ── Aggregation: the signal-washout comparison inputs ─────────────────────────


def stage_aggregate(config: Dict[str, Any]) -> None:
    """Build cluster-level and document-level sentiment time series.

    Produces the two series compared in the signal-washout analysis:
    the theme-level (meeting x topic) series and the document-level
    baseline (one score per meeting).
    """
    logger.info("--- Stage: aggregate ---")
    out = _outdirs(config)

    sentiment = pd.read_parquet(out["results_dir"] / "sentence_sentiment.parquet")
    assignments = pd.read_parquet(out["results_dir"] / "topic_assignments.parquet")
    merged = sentiment.merge(
        assignments[["date", "turn_index", "sentence_index", "topic"]],
        on=["date", "turn_index", "sentence_index"], how="inner",
    )
    logger.info("Merged %d scored sentences with topic assignments",
                len(merged))

    # Document-level baseline: mean compound per meeting (the washout
    # comparator from Ch3 s3.5).
    document_series = (
        merged.groupby("date")
        .agg(document_compound=("sentiment_compound", "mean"),
             n_sentences=("sentiment_compound", "size"))
        .reset_index()
    )
    document_series.to_csv(out["results_dir"] / "document_sentiment_series.csv",
                           index=False)

    # Cluster-level series: mean compound per meeting x topic,
    # excluding HDBSCAN outliers.
    clustered = merged[merged["topic"] != -1]
    cluster_series = (
        clustered.groupby(["date", "topic"])
        .agg(cluster_compound=("sentiment_compound", "mean"),
             n_sentences=("sentiment_compound", "size"))
        .reset_index()
    )
    cluster_series.to_csv(out["results_dir"] / "cluster_sentiment_series.csv",
                          index=False)

    merged.to_parquet(out["results_dir"] / "sentences_scored_assigned.parquet",
                      index=False)
    logger.info("Wrote document series (%d meetings) and cluster series "
                "(%d meeting-topic cells)", len(document_series),
                len(cluster_series))


# ── CLI ───────────────────────────────────────────────────────────────────────


STAGES = {
    "prepare": stage_prepare,
    "topics": stage_topics,
    "grid": stage_grid,
    "stability": stage_stability,
    "sentiment": stage_sentiment,
    "aggregate": stage_aggregate,
}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(STAGES) + ["all"],
                        default="all")
    parser.add_argument("--config", default="config/dissertation.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"], force=True,
    )
    for noisy in ("urllib3", "transformers", "sentence_transformers",
                  "gensim", "numba", "umap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.stage == "all":
        for name in ("prepare", "topics", "sentiment", "aggregate"):
            STAGES[name](config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
