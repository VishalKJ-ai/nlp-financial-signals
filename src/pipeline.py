"""
Main pipeline orchestrator for the NLP Financial Signals project.

Entry point for the full analysis pipeline. Supports three modes:
- sample: Run end-to-end with synthetic sample data (no scraping needed)
- full: Scrape real central bank data and run the full pipeline
- analyze: Load processed data and re-run analysis only

Usage:
    python -m src.pipeline --mode sample
    python -m src.pipeline --mode full
    python -m src.pipeline --mode analyze
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import TextPreprocessor
from src.topics.topic_model import TopicModeler
from src.sentiment.finbert_scorer import FinBERTScorer
from src.signals.extractor import SignalExtractor
from src.evaluation.evaluator import PipelineEvaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_file)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config settings.

    Args:
        config: Configuration dictionary with logging settings.
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(level=level, format=fmt, force=True)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("umap").setLevel(logging.WARNING)


# ── Step Functions ────────────────────────────────────────────────────────────


def step_load_sample(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and preprocess sample data for offline testing.

    Args:
        config: Pipeline configuration.

    Returns:
        Preprocessed DataFrame of sample documents.
    """
    logger.info("=" * 60)
    logger.info("--- Step 1: Loading sample data ---")
    logger.info("=" * 60)

    from data.sample.generate_sample_data import SampleDataGenerator

    generator = SampleDataGenerator(
        output_dir=str(PROJECT_ROOT / config["data"]["paths"]["sample"])
    )
    df = generator.load_or_generate()

    preprocessor = TextPreprocessor(config)
    df = preprocessor.process(df)
    df = preprocessor.extract_metadata(df)

    logger.info("Loaded %d sample documents", len(df))
    return df


def step_collect_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Collect real data from central bank websites.

    Args:
        config: Pipeline configuration.

    Returns:
        Preprocessed DataFrame of collected documents.
    """
    logger.info("=" * 60)
    logger.info("--- Step 1: Collecting central bank data ---")
    logger.info("=" * 60)

    from src.data.boe_scraper import BoEScraper
    from src.data.fed_scraper import FedScraper
    from src.data.ecb_scraper import ECBScraper

    raw_dir = str(PROJECT_ROOT / config["data"]["paths"]["raw"])

    # Collect from each source
    boe_scraper = BoEScraper(config, output_dir=raw_dir)
    fed_scraper = FedScraper(config, output_dir=raw_dir)
    ecb_scraper = ECBScraper(config, output_dir=raw_dir)

    try:
        boe_df = boe_scraper.collect()
        boe_scraper.save(boe_df)
    except Exception as e:
        logger.error("BoE collection failed: %s", e)
        boe_df = pd.DataFrame()
    finally:
        boe_scraper.close()

    try:
        fed_df = fed_scraper.collect()
        fed_scraper.save(fed_df)
    except Exception as e:
        logger.error("Fed collection failed: %s", e)
        fed_df = pd.DataFrame()
    finally:
        fed_scraper.close()

    try:
        ecb_df = ecb_scraper.collect()
        ecb_scraper.save(ecb_df)
    except Exception as e:
        logger.error("ECB collection failed: %s", e)
        ecb_df = pd.DataFrame()
    finally:
        ecb_scraper.close()

    # Merge and preprocess
    preprocessor = TextPreprocessor(config)
    combined = preprocessor.merge_sources(boe_df, fed_df, ecb_df)
    combined = preprocessor.process(combined)
    combined = preprocessor.extract_metadata(combined)

    # Save processed data
    processed_path = str(
        PROJECT_ROOT / config["data"]["paths"]["processed"] / "documents.parquet"
    )
    preprocessor.save(combined, processed_path)

    logger.info("Collected and processed %d documents", len(combined))
    return combined


def step_topic_modelling(
    config: Dict[str, Any],
    df: pd.DataFrame,
    sample_mode: bool = True,
) -> Tuple[TopicModeler, List[int], np.ndarray]:
    """Run topic modelling on preprocessed documents.

    Args:
        config: Pipeline configuration.
        df: Preprocessed documents DataFrame.
        sample_mode: Whether to use the lightweight sample mode.

    Returns:
        Tuple of (fitted TopicModeler, topic assignments, probabilities).
    """
    logger.info("=" * 60)
    logger.info("--- Step 2: Topic Modelling ---")
    logger.info("=" * 60)

    preprocessor = TextPreprocessor(config)
    docs = preprocessor.prepare_for_topics(df)

    modeler = TopicModeler(config)
    eval_cfg = config.get("evaluation", {})
    n_topics = eval_cfg.get("topic_count_range", [5, 30])[0]

    if sample_mode:
        topics, probs = modeler.fit_sample(docs, n_topics=n_topics)
    else:
        topics, probs = modeler.fit(docs)

    # Save model
    models_dir = str(PROJECT_ROOT / config["output"]["models_dir"])
    modeler.save(models_dir)

    logger.info("Topic modelling complete: %d topics assigned", len(set(topics)))
    return modeler, topics, probs


def step_sentiment_analysis(
    config: Dict[str, Any],
    df: pd.DataFrame,
    sample_mode: bool = True,
) -> pd.DataFrame:
    """Run sentiment analysis on preprocessed documents.

    Args:
        config: Pipeline configuration.
        df: Preprocessed documents DataFrame.
        sample_mode: Whether to use rule-based fallback.

    Returns:
        DataFrame with sentiment scores and stance classification.
    """
    logger.info("=" * 60)
    logger.info("--- Step 3: Sentiment Analysis ---")
    logger.info("=" * 60)

    scorer = FinBERTScorer(config, use_precomputed=sample_mode)
    df = scorer.score_documents(df)
    df = scorer.classify_stance(df)

    logger.info("Sentiment analysis complete")
    return df


def step_signal_extraction(
    config: Dict[str, Any],
    df: pd.DataFrame,
    topics: List[int],
    topic_probs: np.ndarray,
) -> pd.DataFrame:
    """Extract composite financial signals.

    Args:
        config: Pipeline configuration.
        df: Sentiment-scored documents DataFrame.
        topics: Topic assignments.
        topic_probs: Topic probability distributions.

    Returns:
        DataFrame with composite signal scores.
    """
    logger.info("=" * 60)
    logger.info("--- Step 4: Signal Extraction ---")
    logger.info("=" * 60)

    extractor = SignalExtractor(config)

    # Compute components
    hdi = extractor.compute_hawkish_dovish_index(df)
    topic_shift = extractor.compute_topic_shift(topic_probs, df["date"])
    sentiment_momentum = extractor.compute_sentiment_momentum(df)

    # Compute composite signal
    signal = extractor.compute_composite_signal(hdi, topic_shift, sentiment_momentum)
    signal = extractor.smooth_signals(signal)

    # Save signals
    signals_dir = str(PROJECT_ROOT / config["output"]["signals_dir"])
    extractor.save_signals(signal, signals_dir)

    logger.info("Signal extraction complete: %d monthly signals", len(signal))
    return signal


def step_evaluation(
    config: Dict[str, Any],
    df: pd.DataFrame,
    modeler: TopicModeler,
    topics: List[int],
    topic_probs: np.ndarray,
    signal: pd.DataFrame,
) -> None:
    """Generate all evaluation plots and summary tables.

    Args:
        config: Pipeline configuration.
        df: Sentiment-scored documents DataFrame.
        modeler: Fitted topic model.
        topics: Topic assignments.
        topic_probs: Topic probability distributions.
        signal: Composite signal DataFrame.
    """
    logger.info("=" * 60)
    logger.info("--- Step 5: Evaluation ---")
    logger.info("=" * 60)

    evaluator = PipelineEvaluator(
        figures_dir=str(PROJECT_ROOT / config["output"]["figures_dir"]),
        signals_dir=str(PROJECT_ROOT / config["output"]["signals_dir"]),
    )

    # Evaluate topics
    topic_info = modeler.get_topic_info()
    preprocessor = TextPreprocessor(config)
    docs = preprocessor.prepare_for_topics(df)

    coherence = modeler.get_coherence_scores(docs, topics)
    evaluator.evaluate_topics(topic_info, coherence, topics, docs)

    # Topic evolution
    timestamps = df["date"].dt.strftime("%Y-%m-%d").tolist()
    topics_over_time = modeler.get_topics_over_time(docs, timestamps, topics)

    # Generate all plots
    evaluator.generate_all_plots(
        topic_info=topic_info,
        topics=topics,
        topics_over_time=topics_over_time,
        scored_df=df,
        signal_df=signal,
    )

    # Summary table
    summary = evaluator.generate_summary_table()
    summary_path = PROJECT_ROOT / config["output"]["signals_dir"] / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved evaluation summary to %s", summary_path)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


# ── Pipeline Modes ────────────────────────────────────────────────────────────


def run_sample_pipeline(config: Dict[str, Any]) -> None:
    """Run the full pipeline with sample data.

    Args:
        config: Pipeline configuration.
    """
    logger.info("Running pipeline in SAMPLE mode")

    df = step_load_sample(config)
    modeler, topics, probs = step_topic_modelling(config, df, sample_mode=True)
    df = step_sentiment_analysis(config, df, sample_mode=True)
    signal = step_signal_extraction(config, df, topics, probs)
    step_evaluation(config, df, modeler, topics, probs, signal)


def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Run the full pipeline with real data collection.

    Args:
        config: Pipeline configuration.
    """
    logger.info("Running pipeline in FULL mode")

    df = step_collect_data(config)
    modeler, topics, probs = step_topic_modelling(config, df, sample_mode=False)
    df = step_sentiment_analysis(config, df, sample_mode=False)
    signal = step_signal_extraction(config, df, topics, probs)
    step_evaluation(config, df, modeler, topics, probs, signal)


def run_analyze_pipeline(config: Dict[str, Any]) -> None:
    """Re-run analysis on previously processed data.

    Args:
        config: Pipeline configuration.
    """
    logger.info("Running pipeline in ANALYZE mode")

    processed_path = (
        PROJECT_ROOT / config["data"]["paths"]["processed"] / "documents.parquet"
    )
    if not processed_path.exists():
        logger.error("Processed data not found at %s. Run 'full' mode first.", processed_path)
        sys.exit(1)

    df = TextPreprocessor.load(str(processed_path))
    df = TextPreprocessor(config).extract_metadata(df)

    modeler, topics, probs = step_topic_modelling(config, df, sample_mode=False)
    df = step_sentiment_analysis(config, df, sample_mode=False)
    signal = step_signal_extraction(config, df, topics, probs)
    step_evaluation(config, df, modeler, topics, probs, signal)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="NLP Financial Signals Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline --mode sample   # Run with sample data\n"
            "  python -m src.pipeline --mode full      # Scrape and analyse\n"
            "  python -m src.pipeline --mode analyze   # Re-analyse processed data\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "full", "analyze"],
        default="sample",
        help="Pipeline execution mode (default: sample)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the pipeline."""
    args = parse_args()

    config = load_config(args.config)
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("NLP Financial Signals Pipeline")
    logger.info("Mode: %s", args.mode)
    logger.info("Config: %s", args.config)
    logger.info("=" * 60)

    mode_dispatch = {
        "sample": run_sample_pipeline,
        "full": run_full_pipeline,
        "analyze": run_analyze_pipeline,
    }

    try:
        mode_dispatch[args.mode](config)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
