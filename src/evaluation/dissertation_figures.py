"""Chapter 4 figure generation.

Every figure in the results chapter is produced by this module from
the persisted pipeline artefacts, so the full set can be regenerated
with one command:

    python -m src.evaluation.dissertation_figures

Design rules: Okabe-Ito colourblind-safe palette in fixed assignment
order for categorical series; single hue for magnitudes; a two-hue
diverging scale with neutral midpoint for sentiment polarity; one axis
per chart; recessive grids.  Figures are saved as 300 dpi PNG and PDF.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)

# Okabe-Ito palette, fixed assignment order (colourblind-safe).
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442"]
SINGLE_HUE = "#0072B2"
NEUTRAL = "#666666"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


class FigureFactory:
    """Generates all Chapter 4 figures from pipeline artefacts."""

    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        self.root = project_root
        self.results_dir = project_root / config["output"]["results_dir"]
        self.tables_dir = project_root / config["output"]["tables_dir"]
        self.figures_dir = project_root / config["output"]["figures_dir"]
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(self.figures_dir / f"{name}.{ext}",
                        bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure %s", name)

    def _topic_labels(self, top_n_words: int = 3) -> Dict[int, str]:
        """Short topic labels: id plus leading c-TF-IDF terms."""
        overview = pd.read_csv(self.tables_dir / "topic_overview.csv")
        labels = {}
        for _, row in overview.iterrows():
            if row["Topic"] == -1:
                continue
            words = str(row["top_words"]).split(", ")[:top_n_words]
            labels[int(row["Topic"])] = f"T{row['Topic']}: {', '.join(words)}"
        return labels

    def _top_topics(self, k: int = 6) -> List[int]:
        """The k largest topics by sentence count."""
        assignments = pd.read_parquet(
            self.results_dir / "topic_assignments.parquet")
        sizes = assignments[assignments["topic"] != -1]["topic"].value_counts()
        return sizes.head(k).index.tolist()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def fig_corpus_overview(self) -> None:
        """Sentences retained per meeting across the sample window."""
        units = pd.read_parquet(
            self.results_dir / "topic_assignments.parquet")
        per_meeting = units.groupby("date").size()

        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.bar(per_meeting.index, per_meeting.values, width=25,
               color=SINGLE_HUE, linewidth=0)
        ax.axvline(pd.Timestamp("2019-01-01"), color=NEUTRAL, lw=0.8,
                   linestyle="--")
        ax.annotate("Press conference after\nevery meeting (2019–)",
                    xy=(pd.Timestamp("2019-03-01"),
                        per_meeting.max() * 0.92),
                    fontsize=8, color=NEUTRAL)
        ax.set_ylabel("Retained sentences")
        ax.set_xlabel("Meeting date")
        self._save(fig, "fig_corpus_overview")

    def fig_topic_sizes(self, k: int = 12) -> None:
        """Largest topics with their c-TF-IDF terms."""
        assignments = pd.read_parquet(
            self.results_dir / "topic_assignments.parquet")
        sizes = (assignments[assignments["topic"] != -1]["topic"]
                 .value_counts().head(k).sort_values())
        labels = self._topic_labels()

        fig, ax = plt.subplots(figsize=(7.5, 0.38 * k + 1))
        ax.barh([labels.get(t, str(t)) for t in sizes.index],
                sizes.values, color=SINGLE_HUE, height=0.62)
        ax.set_xlabel("Sentences")
        ax.grid(axis="y", alpha=0)
        self._save(fig, "fig_topic_sizes")

    def fig_topics_over_time(self, k: int = 6) -> None:
        """Prevalence of the largest topics per meeting over time."""
        assignments = pd.read_parquet(
            self.results_dir / "topic_assignments.parquet")
        top = self._top_topics(k)
        labels = self._topic_labels()

        share = (
            assignments[assignments["topic"].isin(top)]
            .groupby(["date", "topic"]).size()
            .div(assignments.groupby("date").size(), level="date")
            .unstack(fill_value=0.0)
        )
        smoothed = share.rolling(4, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, topic in enumerate(top):
            ax.plot(smoothed.index, smoothed[topic],
                    color=OKABE_ITO[i % len(OKABE_ITO)], lw=2,
                    label=labels.get(topic, str(topic)))
        ax.set_ylabel("Share of meeting sentences (4-meeting MA)")
        ax.set_xlabel("Meeting date")
        ax.legend(fontsize=7.5, ncol=2, loc="upper left")
        self._save(fig, "fig_topics_over_time")

    def fig_cluster_sentiment_heatmap(self, k: int = 10) -> None:
        """Meeting x topic sentiment, diverging scale, gaps = absent."""
        cluster = pd.read_csv(
            self.results_dir / "cluster_sentiment_series.csv",
            parse_dates=["date"])
        top = self._top_topics(k)
        labels = self._topic_labels()

        matrix = (cluster[cluster["topic"].isin(top)]
                  .pivot(index="topic", columns="date",
                         values="cluster_compound")
                  .reindex(top))
        vmax = np.nanmax(np.abs(matrix.values))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(10, 0.42 * k + 1.6))
        mesh = ax.pcolormesh(matrix.columns, np.arange(len(top)),
                             matrix.values, cmap="RdBu", norm=norm,
                             shading="nearest")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels([labels.get(t, str(t)) for t in top], fontsize=8)
        ax.set_xlabel("Meeting date")
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, label="Mean FinBERT compound",
                     fraction=0.025, pad=0.01)
        self._save(fig, "fig_cluster_sentiment_heatmap")

    def fig_signal_washout(self) -> None:
        """Document-level series vs the two most divergent topic series."""
        document = pd.read_csv(
            self.results_dir / "document_sentiment_series.csv",
            parse_dates=["date"]).set_index("date")
        cluster = pd.read_csv(
            self.results_dir / "cluster_sentiment_series.csv",
            parse_dates=["date"])
        labels = self._topic_labels()

        matrix = cluster.pivot(index="date", columns="topic",
                               values="cluster_compound")
        # Restrict to large topics discussed at nearly every meeting,
        # then take the least-correlated pair: the clearest substantive
        # illustration of internally conflicting signals.
        candidates = [t for t in self._top_topics(8)
                      if t in matrix.columns
                      and matrix[t].notna().mean() >= 0.8]
        corr = matrix[candidates].corr()
        pair = corr.stack().idxmin()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.plot(document.index, document["document_compound"],
                color=NEUTRAL, lw=1.6, linestyle="--",
                label="Document-level (baseline)")
        for i, topic in enumerate(pair):
            series = matrix[topic].dropna()
            ax.plot(series.index, series.values, color=OKABE_ITO[i],
                    lw=2, label=labels.get(topic, str(topic)))
        ax.set_ylabel("Mean FinBERT compound")
        ax.set_xlabel("Meeting date")
        ax.legend(fontsize=8, loc="lower left")
        self._save(fig, "fig_signal_washout")

    def fig_lm_benchmark(self) -> None:
        """Document-level FinBERT vs LM dictionary agreement."""
        merged = pd.read_parquet(
            self.results_dir / "sentences_lm_scored.parquet")
        doc = merged.groupby("date")[
            ["sentiment_compound", "lm_compound"]].mean()

        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.axhline(0, color=NEUTRAL, lw=0.6)
        ax.axvline(0, color=NEUTRAL, lw=0.6)
        ax.scatter(doc["lm_compound"], doc["sentiment_compound"],
                   s=22, color=SINGLE_HUE, alpha=0.75, linewidth=0)
        slope, intercept = np.polyfit(doc["lm_compound"],
                                      doc["sentiment_compound"], 1)
        xs = np.linspace(doc["lm_compound"].min(),
                         doc["lm_compound"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#D55E00", lw=1.6)
        ax.set_xlabel("LM dictionary compound (meeting mean)")
        ax.set_ylabel("FinBERT compound (meeting mean)")
        self._save(fig, "fig_lm_benchmark")

    def fig_market_validation(self) -> None:
        """Explained variance: document baseline vs cluster models.

        Shows the baseline, the joint cluster model, and the individual
        topics significant at the 10% level, rather than all ~35
        persistent topics.
        """
        table = pd.read_csv(self.tables_dir / "market_validation.csv")
        table = table[table["target"] == "d_dgs2"].copy()
        keep = (
            table["model"].isin(["document_baseline", "cluster_multivariate"])
            | (table["p_value"] < 0.10)
        )
        table = table[keep]
        table["label"] = table["model"].str.replace("cluster_topic_", "T")
        table = table.sort_values("r_squared")
        colors = [
            NEUTRAL if model == "document_baseline"
            else OKABE_ITO[0] if model == "cluster_multivariate"
            else OKABE_ITO[4]
            for model in table["model"]
        ]

        fig, ax = plt.subplots(figsize=(7, 0.34 * len(table) + 1.2))
        ax.barh(table["label"], table["r_squared"], color=colors,
                height=0.62)
        ax.set_xlabel("$R^2$ (Δ 2-year Treasury yield, "
                      "meeting-day window)")
        ax.grid(axis="y", alpha=0)
        self._save(fig, "fig_market_validation")

    def fig_coherence_grid(self) -> None:
        """Cv coherence across the hyperparameter grid."""
        grid = pd.read_csv(self.tables_dir / "coherence_grid.csv")
        pivots = grid.groupby("hdbscan_min_samples")

        fig, axes = plt.subplots(1, pivots.ngroups,
                                 figsize=(4.2 * pivots.ngroups, 3.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        vmin, vmax = grid["cv_coherence"].min(), grid["cv_coherence"].max()
        for ax, (min_samples, sub) in zip(axes, pivots):
            pivot = sub.pivot(index="hdbscan_min_cluster_size",
                              columns="umap_n_neighbors",
                              values="cv_coherence")
            mesh = ax.pcolormesh(pivot.columns.astype(str),
                                 pivot.index.astype(str), pivot.values,
                                 cmap="Blues", vmin=vmin, vmax=vmax,
                                 shading="nearest")
            for (i, mcs) in enumerate(pivot.index):
                for (j, nn) in enumerate(pivot.columns):
                    ax.text(j, i, f"{pivot.iloc[i, j]:.3f}",
                            ha="center", va="center", fontsize=8)
            ax.set_title(f"min_samples = {min_samples}", fontsize=9)
            ax.set_xlabel("UMAP n_neighbors")
            ax.grid(False)
        axes[0].set_ylabel("HDBSCAN min_cluster_size")
        fig.colorbar(mesh, ax=axes, label="Cv coherence", fraction=0.02)
        self._save(fig, "fig_coherence_grid")

    def fig_seed_stability(self) -> None:
        """Topic count and coherence across random seeds."""
        stability = pd.read_csv(self.tables_dir / "seed_stability.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
        ax1.bar(stability["seed"].astype(str), stability["n_topics"],
                color=SINGLE_HUE, width=0.6)
        ax1.set_xlabel("Seed")
        ax1.set_ylabel("Topics discovered")
        ax2.bar(stability["seed"].astype(str), stability["cv_coherence"],
                color=SINGLE_HUE, width=0.6)
        ax2.set_xlabel("Seed")
        ax2.set_ylabel("Cv coherence")
        for ax in (ax1, ax2):
            ax.grid(axis="x", alpha=0)
        self._save(fig, "fig_seed_stability")

    # ------------------------------------------------------------------

    def generate_all(self) -> None:
        """Generate every figure whose inputs exist."""
        steps = [
            ("fig_corpus_overview", self.fig_corpus_overview),
            ("fig_topic_sizes", self.fig_topic_sizes),
            ("fig_topics_over_time", self.fig_topics_over_time),
            ("fig_cluster_sentiment_heatmap",
             self.fig_cluster_sentiment_heatmap),
            ("fig_signal_washout", self.fig_signal_washout),
            ("fig_lm_benchmark", self.fig_lm_benchmark),
            ("fig_market_validation", self.fig_market_validation),
            ("fig_coherence_grid", self.fig_coherence_grid),
            ("fig_seed_stability", self.fig_seed_stability),
        ]
        for name, step in steps:
            try:
                step()
            except FileNotFoundError as exc:
                logger.warning("Skipping %s (missing input: %s)", name, exc)


def main() -> None:
    """CLI entry point."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent.parent
    with open(project_root / "config/dissertation.yaml") as handle:
        config = yaml.safe_load(handle)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    FigureFactory(config, project_root).generate_all()


if __name__ == "__main__":
    main()
