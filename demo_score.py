"""Live demo: score Fed-style sentences with the pipeline's FinBERT scorer.

Run once before the interview to warm the model cache, then live on call.
"""

from src.sentiment.finbert_scorer import FinBERTScorer

scorer = FinBERTScorer({"sentiment": {
    "model_name": "yiyanghkust/finbert-tone",
    "max_length": 128, "batch_size": 8, "device": "auto",
}})
scorer._load_model()

SENTENCES = [
    "The labor market has continued to strengthen and economic activity has been rising at a strong rate.",
    "Inflation remains elevated, and we are attentive to the risks it poses to both sides of our mandate.",
    "We will continue to monitor the implications of incoming information for the economic outlook.",
]

print(f"\n{'label':>8}  {'score':>6}  sentence")
print("-" * 100)
for text in SENTENCES:
    result = scorer._score_single(text)
    print(f"{result['label']:>8}  {result['compound']:+.2f}  {text}")
print("\nNote the second sentence: hawkish content, POSITIVE tone score.")
print("FinBERT measures tone, not policy stance - the construct distinction in Chapter 5.")
