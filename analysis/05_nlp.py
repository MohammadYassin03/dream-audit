"""Stage 5: NLP layer on the State of the Union corpus.

- Sentiment scoring per address (transformer-based, FinBERT or DistilBERT)
- Topic modeling with BERTopic (sentence-transformer embeddings + UMAP + HDBSCAN)
- Outputs:
    data/processed/sotu_sentiment.parquet  (year, paragraph_idx, sentiment, score)
    data/processed/sotu_topics.parquet     (year, topic_id, topic_label, prevalence)
    data/processed/sotu_topic_river.parquet  (year × topic share, for streamgraph)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "sotu"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Implementation notes (filled in once corpus is fetched in 01_acquire):
#
# 1. Load each .txt file, split into paragraphs.
# 2. For each paragraph, run sentiment via:
#       sentiment-analysis pipeline (DistilBERT-SST2 baseline)
#    Then re-score the economy-keyword subset with FinBERT for finance-tuned sentiment.
# 3. For topic modeling:
#       - encode paragraphs with all-MiniLM-L6-v2
#       - reduce with UMAP (n_neighbors=15, n_components=5)
#       - cluster with HDBSCAN (min_cluster_size=30)
#       - label topics with BERTopic.get_topic_info()
# 4. Aggregate paragraphs to year-level prevalence per topic; output a long table
#    suitable for a streamgraph (year, topic, share).


def main() -> None:
    print("== Stage 5: NLP (deferred — implements after corpus fetch is wired) ==")


if __name__ == "__main__":
    main()
