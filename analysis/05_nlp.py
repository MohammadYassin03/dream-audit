"""Stage 5: NLP layer on the State of the Union corpus.

Topic modeling pipeline using BERTopic over sentence-transformer embeddings.
Sentiment classification is available behind --with-sentiment but defaults
off because DistilBERT-SST2 over 14k paragraphs is slow on CPU and the
streamgraph chart only needs topic shares.

1. Load all SOTU .txt files from data/corpus/sotu/, parse out year and president.
2. Split each address into paragraphs, drop short stubs and applause cues.
3. Embed paragraphs with all-MiniLM-L6-v2.
4. Cluster with BERTopic (UMAP + HDBSCAN + class-based TF-IDF).
5. Aggregate per-paragraph topic assignments to year-level topic shares.

Outputs:
    data/processed/sotu_paragraphs.parquet  (year, idx, text, topic_id, [sentiment])
    data/processed/sotu_topics.parquet      (topic_id, label, top_words, count)
    data/processed/sotu_topic_river.parquet (year, topic_label, share)

Usage:
    python analysis/05_nlp.py [--with-sentiment]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "corpus" / "sotu"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

MIN_TOKENS = 20          # drop very short paragraphs (applause cues, stubs)
MAX_TOKENS = 400         # split very long paragraphs to keep embedding cost bounded
TOP_N_TOPICS = 7         # how many topics to show in the streamgraph
RANDOM_SEED = 42


def _parse_filename(fp: Path) -> dict | None:
    """Filename pattern: 1965_lyndon_b_johnson_d.txt OR 2024_recent.txt"""
    name = fp.stem
    m = re.match(r"(\d{4})_(.+)_([dr])$", name)
    if m:
        year, prez, party = m.groups()
        return {
            "year": int(year),
            "president": prez.replace("_", " ").title(),
            "party": party.upper(),
        }
    m = re.match(r"(\d{4})_recent$", name)
    if m:
        return {"year": int(m.group(1)), "president": "Recent", "party": "?"}
    return None


def load_corpus_paragraphs() -> pd.DataFrame:
    """One row per non-trivial paragraph across all SOTU addresses 1960-2024."""
    rows = []
    for fp in sorted(CORPUS.glob("*.txt")):
        meta = _parse_filename(fp)
        if not meta or meta["year"] < 1960 or meta["year"] > 2025:
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # Split on blank lines first; if document has none, split on sentence-final
        # punctuation as a fallback.
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) < 5:
            paragraphs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            words = para.split()
            if len(words) < MIN_TOKENS:
                continue
            # Trim very long paragraphs at sentence boundary
            if len(words) > MAX_TOKENS:
                para = " ".join(words[:MAX_TOKENS])
            rows.append({
                "year": meta["year"],
                "president": meta["president"],
                "party": meta["party"],
                "paragraph_idx": idx,
                "text": para,
            })
    return pd.DataFrame(rows)


def run_topics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BERTopic. Returns (per-paragraph topic assignments, topic info)."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    print("  loading sentence-transformer (all-MiniLM-L6-v2)")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  embedding {len(df):,} paragraphs")
    embeddings = embedder.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    print(f"  embeddings: shape {embeddings.shape}")

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=25, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    # Stopword-aware vectorizer for clean topic labels. Without this BERTopic's
    # c-TF-IDF picks up "the, and, of, to" as top words for the largest clusters.
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
    )
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=False,
    )
    print("  fitting BERTopic")
    topics, _ = topic_model.fit_transform(df["text"].tolist(), embeddings=embeddings)

    info = topic_model.get_topic_info()
    info = info.rename(columns={"Topic": "topic_id", "Name": "label", "Count": "count"})
    top_words = []
    for tid in info["topic_id"]:
        words = topic_model.get_topic(tid)
        top_words.append(", ".join(w for w, _ in (words or [])[:5]))
    info["top_words"] = top_words

    df = df.copy()
    df["topic_id"] = topics
    return df, info


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """DistilBERT fine-tuned on SST-2. Slow on CPU; opt-in via --with-sentiment."""
    from transformers import pipeline
    print("  loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)")
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=256,
    )
    print(f"  scoring {len(df):,} paragraphs (this is slow on CPU)")
    out = pipe(df["text"].tolist(), batch_size=32)
    df = df.copy()
    df["sentiment"] = [o["label"] for o in out]
    df["sentiment_score"] = [o["score"] for o in out]
    df["sentiment_signed"] = df.apply(
        lambda r: r["sentiment_score"] if r["sentiment"] == "POSITIVE" else -r["sentiment_score"],
        axis=1,
    )
    return df


# Procedural / boilerplate topics that BERTopic surfaces but aren't story-relevant.
# Skip topics whose top words match any of these phrases.
PROCEDURAL_KEYWORDS = (
    "session, congress",
    "speaker, vice president",
    "state union",
    "honor, men women",       # "I salute these heroes" boilerplate
    "leaders, terror, options",
    "personal, information, records",
)


def topic_river(df: pd.DataFrame, info: pd.DataFrame, top_n: int = TOP_N_TOPICS) -> pd.DataFrame:
    """For the top N content topics by overall paragraph count, compute
    year-level share of paragraphs in each topic. Procedural / boilerplate
    topics (session/legislation, speaker greetings, salutes) are skipped."""

    candidates = info.query("topic_id >= 0").copy()
    is_procedural = candidates["top_words"].apply(
        lambda w: any(p in w for p in PROCEDURAL_KEYWORDS)
    )
    candidates = candidates[~is_procedural]
    top = candidates.nlargest(top_n, "count")
    keep_ids = set(top["topic_id"].tolist())

    yr_topic = (
        df.assign(in_top=df["topic_id"].isin(keep_ids))
          .groupby(["year", "topic_id"]).size().reset_index(name="paragraph_count")
    )
    yr_total = df.groupby("year").size().reset_index(name="year_total")
    yr_topic = yr_topic.merge(yr_total, on="year")
    yr_topic["share"] = yr_topic["paragraph_count"] / yr_topic["year_total"]

    yr_topic = yr_topic.merge(
        top[["topic_id", "label", "top_words"]],
        on="topic_id", how="inner",
    )
    return yr_topic[["year", "topic_id", "label", "top_words", "share"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--with-sentiment", action="store_true",
                   help="Also run DistilBERT sentiment classification (slow on CPU)")
    args = p.parse_args()

    print("== Stage 5: NLP ==")
    df = load_corpus_paragraphs()
    if df.empty:
        print("  (no SOTU files in data/corpus/sotu/, run 01_acquire first)")
        return
    print(f"  paragraphs: {len(df):,} across {df['year'].nunique()} years "
          f"({df['year'].min()} to {df['year'].max()})")

    df, info = run_topics(df)
    info.to_parquet(PROC / "sotu_topics.parquet", index=False)
    print(f"  sotu_topics.parquet: {len(info)} topics (top words shown in chart)")

    if args.with_sentiment:
        df = run_sentiment(df)

    df.to_parquet(PROC / "sotu_paragraphs.parquet", index=False)
    print(f"  sotu_paragraphs.parquet: {len(df):,} rows")

    river = topic_river(df, info)
    river.to_parquet(PROC / "sotu_topic_river.parquet", index=False)
    print(f"  sotu_topic_river.parquet: {len(river):,} year-topic rows")
    print("Done.")


if __name__ == "__main__":
    main()
