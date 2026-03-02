"""Pipeline evaluation and visualisation module.

Generates comprehensive evaluation metrics and visualisations for
topic modelling quality, sentiment analysis distribution, signal
characteristics, and temporal evolution of all components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]
CB_COLORS = {"boe": "#0072C6", "fed": "#003366", "ecb": "#003399"}


class PipelineEvaluator:
    """Evaluator for the NLP financial signals pipeline.

    Generates evaluation plots and summary tables for topic quality,
    sentiment distribution, signal characteristics, and cross-bank
    comparisons.

    Attributes:
        figures_dir: Directory to save evaluation plots.
        signals_dir: Directory to save signal output files.
        results: Collected evaluation results.
    """

    def __init__(
        self,
        figures_dir: str = "outputs/figures",
        signals_dir: str = "outputs/signals",
    ) -> None:
        """Initialise the evaluator.

        Args:
            figures_dir: Directory path to save generated figures.
            signals_dir: Directory path to save signal outputs.
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir = Path(signals_dir)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def evaluate_topics(
        self,
        topic_info: pd.DataFrame,
        coherence_scores: Dict[str, float],
        topics: List[int],
        documents: List[str],
    ) -> Dict[str, Any]:
        """Evaluate topic model quality.

        Args:
            topic_info: BERTopic topic info DataFrame.
            coherence_scores: Dict of coherence measure → score.
            topics: Topic assignments for each document.
            documents: List of document strings.

        Returns:
            Dictionary of evaluation metrics.
        """
        n_topics = len(set(t for t in topics if t >= 0))
        n_outliers = topics.count(-1) if isinstance(topics, list) else int((np.array(topics) == -1).sum())
        outlier_pct = n_outliers / len(topics) * 100

        metrics = {
            "n_topics": n_topics,
            "n_documents": len(documents),
            "n_outliers": n_outliers,
            "outlier_pct": outlier_pct,
            **coherence_scores,
        }

        self.results["topics"] = metrics
        logger.info("Topic evaluation: %d topics, %.1f%% outliers, c_v=%.4f",
                     n_topics, outlier_pct, coherence_scores.get("c_v", 0))
        return metrics

    def plot_topic_distribution(
        self, topic_info: pd.DataFrame, topics: List[int]
    ) -> Path:
        """Plot the distribution of documents across topics.

        Args:
            topic_info: Topic information DataFrame.
            topics: Topic assignments.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        topic_counts = pd.Series(topics).value_counts().sort_index()
        topic_counts = topic_counts[topic_counts.index >= 0]

        bars = ax.bar(
            range(len(topic_counts)),
            topic_counts.values,
            color=COLORS[0],
            alpha=0.8,
            edgecolor="white",
        )

        ax.set_xlabel("Topic ID", fontsize=12)
        ax.set_ylabel("Number of Documents", fontsize=12)
        ax.set_title("Document Distribution Across Topics", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(topic_counts)))
        ax.set_xticklabels(topic_counts.index, fontsize=10)

        # Add count labels
        for bar, count in zip(bars, topic_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = self.figures_dir / "topic_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved topic distribution plot to %s", path)
        return path

    def plot_topic_evolution(
        self,
        topics_over_time: pd.DataFrame,
        topic_names: Optional[Dict[int, str]] = None,
    ) -> Path:
        """Plot topic frequency evolution over time as a heatmap.

        Args:
            topics_over_time: DataFrame with period, topic, frequency.
            topic_names: Optional mapping of topic ID to name.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        if "frequency" in topics_over_time.columns:
            pivot = topics_over_time.pivot_table(
                index="topic", columns="period", values="frequency", fill_value=0
            )
        else:
            pivot = topics_over_time.pivot_table(
                index="topic", columns="period", values="count", fill_value=0
            )

        if topic_names:
            pivot.index = [topic_names.get(t, f"Topic {t}") for t in pivot.index]

        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", linewidths=0.5,
            cbar_kws={"label": "Frequency"},
        )

        ax.set_title("Topic Evolution Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Period", fontsize=12)
        ax.set_ylabel("Topic", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)

        plt.tight_layout()
        path = self.figures_dir / "topic_evolution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved topic evolution plot to %s", path)
        return path

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> Path:
        """Plot sentiment score distribution across documents.

        Args:
            df: DataFrame with sentiment columns.

        Returns:
            Path to the saved figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Compound score histogram
        axes[0].hist(
            df["sentiment_compound"], bins=30, color=COLORS[0],
            alpha=0.7, edgecolor="white",
        )
        axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Compound Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Compound Sentiment Distribution")

        # Sentiment label pie chart
        label_counts = df["sentiment_label"].value_counts()
        colors_pie = [COLORS[2], COLORS[1], COLORS[5]]
        axes[1].pie(
            label_counts.values, labels=label_counts.index,
            colors=colors_pie[:len(label_counts)],
            autopct="%1.1f%%", startangle=90,
        )
        axes[1].set_title("Sentiment Label Distribution")

        # Sentiment by source
        if "source" in df.columns:
            sources = df["source"].unique()
            for i, source in enumerate(sorted(sources)):
                subset = df[df["source"] == source]["sentiment_compound"]
                axes[2].hist(
                    subset, bins=20, alpha=0.5,
                    color=CB_COLORS.get(source, COLORS[i % len(COLORS)]),
                    label=source.upper(), edgecolor="white",
                )
            axes[2].legend()
            axes[2].set_xlabel("Compound Score")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Sentiment by Central Bank")

        plt.tight_layout()
        path = self.figures_dir / "sentiment_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sentiment distribution plot to %s", path)
        return path

    def plot_sentiment_trends(self, df: pd.DataFrame) -> Path:
        """Plot sentiment trends over time by central bank.

        Args:
            df: DataFrame with date, source, and sentiment columns.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        sources = sorted(df["source"].unique())
        for source in sources:
            subset = df[df["source"] == source]
            monthly = subset.groupby("year_month")["sentiment_compound"].mean()
            # Rolling average for smoother trend
            smoothed = monthly.rolling(3, min_periods=1, center=True).mean()
            ax.plot(
                range(len(smoothed)), smoothed.values,
                color=CB_COLORS.get(source, COLORS[0]),
                label=source.upper(), linewidth=2, alpha=0.8,
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Sentiment (3-month rolling mean)", fontsize=12)
        ax.set_title("Sentiment Trends by Central Bank", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)

        plt.tight_layout()
        path = self.figures_dir / "sentiment_trends.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sentiment trends plot to %s", path)
        return path

    def plot_signal_dashboard(
        self,
        signal_df: pd.DataFrame,
        source: str = "all",
    ) -> Path:
        """Plot a comprehensive signal dashboard.

        Shows the composite signal, its components, and the signal
        classification over time.

        Args:
            signal_df: DataFrame with composite signal columns.
            source: Central bank identifier for the title.

        Returns:
            Path to the saved figure.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        x = range(len(signal_df))

        # Panel 1: Composite signal
        axes[0].fill_between(
            x, signal_df["composite_signal"], 0,
            where=signal_df["composite_signal"] >= 0,
            color=COLORS[1], alpha=0.3, label="Hawkish",
        )
        axes[0].fill_between(
            x, signal_df["composite_signal"], 0,
            where=signal_df["composite_signal"] < 0,
            color=COLORS[2], alpha=0.3, label="Dovish",
        )
        if "composite_signal_smooth" in signal_df.columns:
            axes[0].plot(x, signal_df["composite_signal_smooth"],
                         color="black", linewidth=2, label="Smoothed")
        axes[0].axhline(0, color="gray", linestyle="-", alpha=0.3)
        axes[0].axhline(0.3, color="red", linestyle="--", alpha=0.3, label="Threshold")
        axes[0].axhline(-0.3, color="green", linestyle="--", alpha=0.3)
        axes[0].set_ylabel("Composite Signal")
        axes[0].set_title(f"Financial Signal Dashboard — {source.upper()}", fontsize=14, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=9)

        # Panel 2: Components
        if "hawkish_dovish_index_norm" in signal_df.columns:
            axes[1].plot(x, signal_df["hawkish_dovish_index_norm"],
                         label="Sentiment Level", color=COLORS[0], alpha=0.7)
        if "topic_shift_norm" in signal_df.columns:
            axes[1].plot(x, signal_df["topic_shift_norm"],
                         label="Topic Shift", color=COLORS[3], alpha=0.7)
        if "sentiment_momentum_norm" in signal_df.columns:
            axes[1].plot(x, signal_df["sentiment_momentum_norm"],
                         label="Sentiment Momentum", color=COLORS[4], alpha=0.7)
        axes[1].axhline(0, color="gray", linestyle="-", alpha=0.3)
        axes[1].set_ylabel("Normalised Component")
        axes[1].set_title("Signal Components", fontsize=12)
        axes[1].legend(loc="upper right", fontsize=9)

        # Panel 3: Signal classification
        signal_colors = {
            "hawkish": COLORS[1],
            "dovish": COLORS[2],
            "neutral": COLORS[5],
        }
        for i, (_, row) in enumerate(signal_df.iterrows()):
            label = row.get("signal_label", "neutral")
            axes[2].bar(i, 1, color=signal_colors.get(label, "gray"), alpha=0.7)
        axes[2].set_ylabel("Classification")
        axes[2].set_yticks([])
        axes[2].set_title("Monthly Signal Classification", fontsize=12)

        # Custom legend for panel 3
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS[1], alpha=0.7, label="Hawkish"),
            Patch(facecolor=COLORS[2], alpha=0.7, label="Dovish"),
            Patch(facecolor=COLORS[5], alpha=0.7, label="Neutral"),
        ]
        axes[2].legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.tight_layout()
        path = self.figures_dir / f"signal_dashboard_{source}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved signal dashboard to %s", path)
        return path

    def plot_stance_comparison(self, df: pd.DataFrame) -> Path:
        """Plot stance distribution comparison across central banks.

        Args:
            df: DataFrame with source and stance columns.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if "stance" not in df.columns or "source" not in df.columns:
            logger.warning("Missing stance or source columns")
            plt.close(fig)
            return self.figures_dir / "stance_comparison.png"

        ct = pd.crosstab(df["source"], df["stance"], normalize="index") * 100
        ct = ct.reindex(columns=["hawkish", "neutral", "dovish"], fill_value=0)

        ct.plot(
            kind="bar", ax=ax, stacked=True,
            color=[COLORS[1], COLORS[5], COLORS[2]],
            alpha=0.8, edgecolor="white",
        )

        ax.set_xlabel("Central Bank", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title("Monetary Policy Stance by Central Bank", fontsize=14, fontweight="bold")
        ax.legend(title="Stance", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()
        path = self.figures_dir / "stance_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved stance comparison plot to %s", path)
        return path

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of all evaluation metrics.

        Returns:
            DataFrame with evaluation metrics.
        """
        rows: List[Dict[str, Any]] = []

        if "topics" in self.results:
            t = self.results["topics"]
            rows.append({"metric": "Topics Discovered", "value": t.get("n_topics", 0)})
            rows.append({"metric": "Documents Analysed", "value": t.get("n_documents", 0)})
            rows.append({"metric": "Outlier %", "value": f"{t.get('outlier_pct', 0):.1f}%"})
            rows.append({"metric": "Coherence (c_v)", "value": f"{t.get('c_v', 0):.4f}"})
            rows.append({"metric": "Coherence (c_npmi)", "value": f"{t.get('c_npmi', 0):.4f}"})

        summary = pd.DataFrame(rows)
        logger.info("Generated summary table with %d rows", len(summary))
        return summary

    def generate_all_plots(
        self,
        topic_info: pd.DataFrame,
        topics: List[int],
        topics_over_time: pd.DataFrame,
        scored_df: pd.DataFrame,
        signal_df: pd.DataFrame,
    ) -> List[Path]:
        """Generate all evaluation plots.

        Args:
            topic_info: Topic information DataFrame.
            topics: Topic assignments.
            topics_over_time: Topic evolution DataFrame.
            scored_df: Sentiment-scored documents.
            signal_df: Composite signal DataFrame.

        Returns:
            List of paths to saved figures.
        """
        logger.info("=" * 60)
        logger.info("Generating all evaluation plots")
        logger.info("=" * 60)

        paths: List[Path] = []
        paths.append(self.plot_topic_distribution(topic_info, topics))
        paths.append(self.plot_topic_evolution(topics_over_time))
        paths.append(self.plot_sentiment_distribution(scored_df))
        paths.append(self.plot_sentiment_trends(scored_df))
        paths.append(self.plot_signal_dashboard(signal_df))
        paths.append(self.plot_stance_comparison(scored_df))

        logger.info("Generated %d plots", len(paths))
        return paths
