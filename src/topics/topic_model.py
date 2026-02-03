"""BERTopic-based topic modelling module for central bank communications.

Wraps BERTopic with configuration-driven UMAP and HDBSCAN parameters,
and provides methods for fitting, transforming, topic evolution tracking,
coherence scoring, and model persistence.

In sample mode, uses a lightweight embedding model to avoid large
downloads during testing.  Full mode supports any sentence-transformers
model specified in the config.

References:
    BERTopic: https://maartengr.github.io/BERTopic/
    UMAP: https://umap-learn.readthedocs.io/
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopicModeler:
    """Topic modelling wrapper using BERTopic.

    Configures the full BERTopic pipeline including the embedding model,
    UMAP dimensionality reduction, and HDBSCAN clustering.  Provides
    methods for fitting, transformation, topic evolution over time,
    and coherence evaluation.

    Attributes:
        embedding_model: Sentence-transformers model identifier.
        nr_topics: Target number of topics or 'auto'.
        min_topic_size: Minimum documents per topic.
        umap_params: UMAP hyperparameters.
        hdbscan_params: HDBSCAN hyperparameters.
        top_n_words: Number of words per topic representation.
        model: Fitted BERTopic model instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the topic modeler from pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The ``topics``
                sub-key is used.
        """
        topics_cfg = config.get("topics", {})
        self.embedding_model: str = topics_cfg.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )
        self.nr_topics: Any = topics_cfg.get("nr_topics", "auto")
        self.min_topic_size: int = topics_cfg.get("min_topic_size", 10)
        self.top_n_words: int = topics_cfg.get("top_n_words", 10)
        self.calculate_probabilities: bool = topics_cfg.get(
            "calculate_probabilities", True
        )
        self.verbose: bool = topics_cfg.get("verbose", True)

        self.umap_params: Dict[str, Any] = topics_cfg.get("umap", {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        })
        self.hdbscan_params: Dict[str, Any] = topics_cfg.get("hdbscan", {
            "min_cluster_size": 10,
            "min_samples": 5,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        })

        self.model = None
        self._topic_info: Optional[pd.DataFrame] = None

    def fit(self, documents: List[str]) -> Tuple[List[int], Optional[np.ndarray]]:
        """Fit the BERTopic model on the provided documents.

        Args:
            documents: List of document strings.

        Returns:
            Tuple of (topic_assignments, topic_probabilities).
            Probabilities may be None if calculate_probabilities is False.
        """
        logger.info("Fitting BERTopic model on %d documents", len(documents))
        logger.info("Embedding model: %s", self.embedding_model)
        logger.info("UMAP params: %s", self.umap_params)
        logger.info("HDBSCAN params: %s", self.hdbscan_params)

        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from umap import UMAP
        except ImportError as e:
            logger.error(
                "Required packages not installed: %s. "
                "Install with: pip install bertopic umap-learn hdbscan", e
            )
            raise

        # Configure sub-models
        umap_model = UMAP(**self.umap_params)
        hdbscan_model = HDBSCAN(**self.hdbscan_params, prediction_data=True)

        # Build BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=self.nr_topics if self.nr_topics != "auto" else None,
            top_n_words=self.top_n_words,
            min_topic_size=self.min_topic_size,
            calculate_probabilities=self.calculate_probabilities,
            verbose=self.verbose,
        )

        topics, probs = self.model.fit_transform(documents)
        self._topic_info = self.model.get_topic_info()

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = topics.count(-1)
        logger.info("Discovered %d topics (%d outlier documents)", n_topics, n_outliers)

        return topics, probs

    def fit_sample(
        self, documents: List[str], n_topics: int = 6
    ) -> Tuple[List[int], np.ndarray]:
        """Fit a simplified topic model for sample/testing mode.

        Uses TF-IDF and basic clustering instead of transformer
        embeddings, allowing fast execution without model downloads.

        Args:
            documents: List of document strings.
            n_topics: Number of topics to assign.

        Returns:
            Tuple of (topic_assignments, topic_probabilities).
        """
        logger.info("Fitting sample topic model on %d documents (n_topics=%d)",
                     len(documents), n_topics)

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        # TF-IDF vectorisation
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(documents)

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        topics = kmeans.fit_predict(tfidf_matrix).tolist()

        # Compute soft probabilities via distance to centroids
        distances = kmeans.transform(tfidf_matrix)
        # Convert distances to probabilities (inverse distance, normalised)
        inv_distances = 1.0 / (1.0 + distances)
        probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        # Extract topic representations (top words per cluster)
        feature_names = vectorizer.get_feature_names_out()
        topic_info_records = []
        for topic_id in range(n_topics):
            centroid = kmeans.cluster_centers_[topic_id]
            top_indices = centroid.argsort()[-self.top_n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_info_records.append({
                "Topic": topic_id,
                "Count": topics.count(topic_id),
                "Name": f"Topic_{topic_id}",
                "Representation": top_words,
            })

        self._topic_info = pd.DataFrame(topic_info_records)
        self._sample_vectorizer = vectorizer
        self._sample_kmeans = kmeans

        logger.info("Sample model fitted: %d topics", n_topics)
        return topics, probs

    def transform(self, documents: List[str]) -> Tuple[List[int], Optional[np.ndarray]]:
        """Assign topics to new documents using a fitted model.

        Args:
            documents: List of new document strings.

        Returns:
            Tuple of (topic_assignments, topic_probabilities).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None and not hasattr(self, "_sample_kmeans"):
            raise RuntimeError("Model must be fitted before transform")

        if self.model is not None:
            topics, probs = self.model.transform(documents)
            return topics, probs

        # Sample mode transform
        tfidf = self._sample_vectorizer.transform(documents)
        topics = self._sample_kmeans.predict(tfidf).tolist()
        distances = self._sample_kmeans.transform(tfidf)
        inv_distances = 1.0 / (1.0 + distances)
        probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Return topic information including top words and counts.

        Returns:
            DataFrame with topic details.
        """
        if self.model is not None:
            return self.model.get_topic_info()
        if self._topic_info is not None:
            return self._topic_info
        return pd.DataFrame()

    def get_topic_words(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get the top words for a specific topic.

        Args:
            topic_id: Topic identifier.

        Returns:
            List of (word, weight) tuples.
        """
        if self.model is not None:
            return self.model.get_topic(topic_id)
        if self._topic_info is not None:
            row = self._topic_info[self._topic_info["Topic"] == topic_id]
            if not row.empty:
                words = row.iloc[0]["Representation"]
                return [(w, 1.0 / (i + 1)) for i, w in enumerate(words)]
        return []

    def get_topics_over_time(
        self,
        documents: List[str],
        timestamps: List[str],
        topics: List[int],
        nr_bins: int = 20,
    ) -> pd.DataFrame:
        """Compute topic frequency over time.

        Args:
            documents: List of document strings.
            timestamps: List of date strings corresponding to documents.
            topics: Topic assignments for each document.
            nr_bins: Number of time bins.

        Returns:
            DataFrame with topic frequencies per time bin.
        """
        if self.model is not None:
            return self.model.topics_over_time(
                documents, timestamps, nr_bins=nr_bins
            )

        # Manual computation for sample mode
        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps),
            "topic": topics,
        })
        df["period"] = pd.cut(df["date"], bins=nr_bins)

        # Count topics per period
        topic_time = df.groupby(["period", "topic"]).size().reset_index(name="count")
        total_per_period = df.groupby("period").size().reset_index(name="total")
        topic_time = topic_time.merge(total_per_period, on="period")
        topic_time["frequency"] = topic_time["count"] / topic_time["total"]

        return topic_time

    def get_coherence_scores(
        self,
        documents: List[str],
        topics: List[int],
        measures: List[str] = None,
    ) -> Dict[str, float]:
        """Compute topic coherence scores.

        Uses the Gensim coherence model to evaluate topic quality
        across multiple coherence measures.

        Args:
            documents: List of document strings.
            topics: Topic assignments.
            measures: List of coherence measures (default: ['c_v', 'c_npmi']).

        Returns:
            Dictionary mapping measure names to scores.
        """
        if measures is None:
            measures = ["c_v", "c_npmi"]

        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
        except ImportError:
            logger.warning("Gensim not available, returning dummy coherence scores")
            return {m: 0.0 for m in measures}

        # Tokenise documents
        tokenized = [doc.lower().split() for doc in documents]
        dictionary = Dictionary(tokenized)

        # Get topic word lists
        topic_ids = sorted(set(t for t in topics if t >= 0))
        topic_words = []
        for tid in topic_ids:
            words = self.get_topic_words(tid)
            topic_words.append([w for w, _ in words[:self.top_n_words]])

        if not topic_words:
            return {m: 0.0 for m in measures}

        scores: Dict[str, float] = {}
        for measure in measures:
            try:
                cm = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized,
                    dictionary=dictionary,
                    coherence=measure,
                )
                scores[measure] = cm.get_coherence()
                logger.info("Coherence (%s): %.4f", measure, scores[measure])
            except Exception as e:
                logger.warning("Failed to compute %s coherence: %s", measure, e)
                scores[measure] = 0.0

        return scores

    def merge_similar_topics(self, threshold: float = 0.7) -> None:
        """Merge topics with high similarity.

        Args:
            threshold: Cosine similarity threshold for merging.
        """
        if self.model is None:
            logger.warning("Cannot merge topics in sample mode")
            return

        topics_to_merge = []
        topic_ids = [t for t in self.model.get_topics().keys() if t >= 0]

        for i, t1 in enumerate(topic_ids):
            for t2 in topic_ids[i + 1:]:
                words1 = set(w for w, _ in self.model.get_topic(t1))
                words2 = set(w for w, _ in self.model.get_topic(t2))
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                if overlap >= threshold:
                    topics_to_merge.append([t1, t2])

        if topics_to_merge:
            self.model.merge_topics(
                self.model._get_document_info()["Document"].tolist(),
                topics_to_merge,
            )
            logger.info("Merged %d topic pairs", len(topics_to_merge))

    def save(self, path: str) -> Path:
        """Save the fitted model to disk.

        Args:
            path: Directory path to save the model.

        Returns:
            Path to the saved model.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = save_path / "bertopic_model"
            self.model.save(str(model_path))
            logger.info("Saved BERTopic model to %s", model_path)
            return model_path

        # Save sample mode artefacts
        artefacts = {
            "vectorizer": self._sample_vectorizer,
            "kmeans": self._sample_kmeans,
            "topic_info": self._topic_info,
        }
        pkl_path = save_path / "sample_topic_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artefacts, f)
        logger.info("Saved sample topic model to %s", pkl_path)
        return pkl_path

    def load(self, path: str) -> None:
        """Load a fitted model from disk.

        Args:
            path: Path to the saved model.
        """
        load_path = Path(path)

        if load_path.is_dir() and (load_path / "topic_model.bin").exists():
            from bertopic import BERTopic
            self.model = BERTopic.load(str(load_path))
            logger.info("Loaded BERTopic model from %s", load_path)
            return

        if load_path.suffix == ".pkl" or (load_path / "sample_topic_model.pkl").exists():
            pkl_path = load_path if load_path.suffix == ".pkl" else load_path / "sample_topic_model.pkl"
            with open(pkl_path, "rb") as f:
                artefacts = pickle.load(f)
            self._sample_vectorizer = artefacts["vectorizer"]
            self._sample_kmeans = artefacts["kmeans"]
            self._topic_info = artefacts["topic_info"]
            logger.info("Loaded sample topic model from %s", pkl_path)
            return

        raise FileNotFoundError(f"No model found at {load_path}")
