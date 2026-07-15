"""Live demo: FinBERT vs the Loughran-McDonald dictionary, then FinBERT's limit.

Run via ./demo — loads the cached model silently and prints:
  Act 1 (always): why a context-reading transformer beats a word-counting dictionary.
  Act 2 (only with `./demo limits`): the tone-vs-stance boundary of the domain model.
"""

import sys
from pathlib import Path

import transformers

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

from src.evaluation.lm_baseline import LMScorer
from src.sentiment.finbert_scorer import FinBERTScorer

print("\nLive scoring: FinBERT transformer vs Loughran-McDonald dictionary")
print("Loading model from local cache...", flush=True)

finbert = FinBERTScorer({"sentiment": {
    "model_name": "yiyanghkust/finbert-tone",
    "max_length": 128, "batch_size": 8, "device": "auto",
}})
finbert._load_model()

_LM_PATH = Path("data/external/Loughran-McDonald_MasterDictionary.csv")
lm = LMScorer(_LM_PATH) if _LM_PATH.exists() else None
if lm is None:
    print("(LM dictionary not found - download from https://sraf.nd.edu for the comparison column)")


def row(text):
    fb = finbert._score_single(text)
    lm_col = f"{lm.score(text)[2]:+.2f}" if lm else "  n/a"
    return f"  {fb['compound']:+.2f} ({fb['label']:>8})   {lm_col}        {text}"


print("\n[1] Why context matters - the dictionary cannot read negation")
print(f"\n  {'FinBERT':<18}{'LM dict':<10}sentence")
print("  " + "-" * 95)
print(row("We are concerned about inflation."))
print(row("We are not concerned about inflation."))
print("\n  Same dictionary score for both - it counts the word 'concerned' and cannot see the 'not'.")
print("  The transformer reads the whole sentence. This is why the pipeline scores with FinBERT")
print("  and keeps the dictionary only as a validation benchmark.")

if len(sys.argv) > 1 and sys.argv[1] == "limits":
    print("\n[2] Where even FinBERT hits its limit")
    print(f"\n  {'FinBERT':<18}{'LM dict':<10}sentence")
    print("  " + "-" * 95)
    print(row("Inflation remains elevated, and we are attentive to the risks it poses to both sides of our mandate."))
    print("\n  Hawkish, bad-news content - scored POSITIVE. FinBERT reads confident institutional")
    print("  language as optimism: it measures TONE, not policy STANCE. That construct gap is a core")
    print("  finding of the dissertation, and why results are validated against market data rather")
    print("  than trusted on faith.")
print()
