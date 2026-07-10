# NLP Financial Signals

**Cluster-level sentiment signals from Federal Reserve press conference transcripts, using BERTopic and FinBERT.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-orange)

> MSc Applied AI Dissertation Project — University of Warwick, 2025–2026

---

## Motivation

A single FOMC press conference can be hawkish about inflation and dovish about the labour market at the same time. The dominant approach in financial NLP — scoring sentiment for the whole document — averages those conflicting signals away. This project calls that problem **signal washout**, and resolves it by scoring sentiment at the *topic-cluster* level instead.

The pipeline discovers latent policy themes in Federal Reserve press conference transcripts with **BERTopic** (Sentence-BERT embeddings → UMAP → HDBSCAN → c-TF-IDF), then scores each theme with **FinBERT** (Yang et al., 2020, `yiyanghkust/finbert-tone`), producing a theme-level sentiment time series for every FOMC meeting since April 2011.

## Corpus

| Property | Value |
|----------|-------|
| Document type | FOMC post-meeting press conference transcripts |
| Window | April 2011 (first Bernanke press conference) – present |
| Meetings | 92 |
| Analysis unit | Sentences from the Chair's Q&A answers (~22,500 sentences, ~467k words) |
| Source | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) (public domain), scraped with rate limiting |

Restricting to a single document register — the Chair's spontaneous answers — controls for the stylistic heterogeneity of Fed communications and isolates the authoritative policy voice.

## Method

1. **Collection & parsing** — download all transcript PDFs, segment into speaker turns, retain the Chair's Q&A answers (`src/data/fomc_presser_scraper.py`).
2. **Preprocessing** — conservative disfluency removal (false starts, repeated words, leading fillers) and sentence segmentation; hedging is deliberately preserved because it carries policy signal (`src/data/presser_preprocessor.py`).
3. **Topic discovery** — BERTopic with fixed seeds over MiniLM sentence embeddings (`src/dissertation_pipeline.py --stage topics`).
4. **Sentiment scoring** — FinBERT-tone at sentence level, aggregated per meeting × topic (cluster-level) and per meeting (document-level baseline).
5. **Triangulated evaluation**:
   - **Arm 1 — internal coherence**: Cv coherence plus a structured hyperparameter grid over UMAP/HDBSCAN settings.
   - **Arm 2 — benchmark**: correlation and directional agreement against the Loughran–McDonald dictionary (`src/evaluation/lm_baseline.py`).
   - **Arm 3 — external validation**: cluster-level sentiment regressed against meeting-day changes in market-implied rate expectations (2-year Treasury yield / fed funds futures), compared against the document-level baseline (`src/evaluation/market_validation.py`).
6. **Robustness** — seed-stability runs and hyperparameter sensitivity analysis.

## Quick start

```bash
git clone https://github.com/VishalKJ-ai/nlp-financial-signals.git
cd nlp-financial-signals
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt

# Collect the corpus (~92 PDFs, rate-limited)
python -m src.data.fomc_presser_scraper

# Run the staged pipeline
python -m src.dissertation_pipeline --stage prepare
python -m src.dissertation_pipeline --stage topics
python -m src.dissertation_pipeline --stage sentiment
python -m src.dissertation_pipeline --stage aggregate

# Evaluation arms (LM master dictionary: download from https://sraf.nd.edu
# into data/external/Loughran-McDonald_MasterDictionary.csv first)
python -m src.evaluation.lm_baseline
python -m src.evaluation.market_validation

# Robustness + figures
python -m src.dissertation_pipeline --stage grid
python -m src.dissertation_pipeline --stage stability
python -m src.evaluation.dissertation_figures
```

All parameters live in `config/dissertation.yaml`; every stage persists its artefacts, so figures and tables regenerate without re-running models. Results land in `outputs/dissertation/`.

## Repository layout

```
nlp-financial-signals/
├── config/dissertation.yaml            # All dissertation parameters + seeds
├── src/
│   ├── dissertation_pipeline.py        # Staged orchestrator
│   ├── data/
│   │   ├── fomc_presser_scraper.py     # Transcript download + speaker parsing
│   │   └── presser_preprocessor.py     # Disfluency cleaning + sentence units
│   ├── sentiment/finbert_scorer.py     # FinBERT inference (variant-agnostic labels)
│   └── evaluation/
│       ├── lm_baseline.py              # Loughran-McDonald benchmark (Arm 2)
│       ├── market_validation.py        # Market regressions (Arm 3)
│       └── dissertation_figures.py     # All Chapter 4 figures, scripted
├── tests/                              # pytest suite
└── outputs/dissertation/               # Frozen tables, series, figures
```

The repository also contains an earlier multi-bank prototype (BoE/Fed/ECB speeches with a composite hawkish/dovish indicator, `src/pipeline.py`) retained as an extension direction; the dissertation methodology above supersedes it.

## Tech stack

BERTopic · sentence-transformers · HuggingFace Transformers (FinBERT) · UMAP · HDBSCAN · Gensim (Cv coherence) · statsmodels · pandas · matplotlib · pytest

## Ethics & licensing

FOMC transcripts are public-domain institutional records with no human-subject concerns. FinBERT and Sentence-BERT are Apache-2.0 via Hugging Face. Extracted signals are academic research outputs, not financial advice or trading signals. Code, seeds, and configuration are published for reproducibility.

## Author

**Vishal Joshi** — MSc Applied Artificial Intelligence, University of Warwick (2025–2026)
