"""Stage 5: NLP layer on the State of the Union corpus.

Pipeline:
1. Load all SOTU .txt files from data/corpus/sotu/, parse out year and president.
2. Split each address into paragraphs (skip applause cues and tiny stubs).
3. Sentiment classification per paragraph (DistilBERT fine-tuned on SST-2).
4. Topic modeling across all paragraphs with BERTopic
   (sentence-transformer all-MiniLM-L6-v2, UMAP, HDBSCAN).
5. Aggregate paragraphs to year-level topic prevalence for the streamgraph.

Outputs:
    data/processed/sotu_paragraphs.parquet  (year, idx, text, sentiment, score)
    data/processed/sotu_topics.parquet      (topic_id, label, top_words, count)
    data/processed/sotu_topic_river.parquet (year, topic_label, share)
"""
from __future__ import annotations

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

MIN_TOKENS = 12          # drop very short paragraphs (applause cues etc.)
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
            if len(para.split()) < MIN_TOKENS:
                continue
            rows.append({
                "year": meta["year"],
                "president": meta["president"],
                "party": meta["party"],
                "paragraph_idx": idx,
                "text": para,
            })
    df = pd.DataFrame(rows)
    return df


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """DistilBERT fine-tuned on SST-2. Returns df with sentiment_label + score."""
    from transformers import pipeline
    print(f"  loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)")
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=256,
    )
    print(f"  scoring {len(df):,} paragraphs")
    out = pipe(df["text"].tolist(), batch_size=16)
    df = df.copy()
    df["sentiment"] = [o["label"] for o in out]
    df["sentiment_score"] = [o["score"] for o in out]
    df["sentiment_signed"] = df.apply(
        lambda r: r["sentiment_score"] if r["sentiment"] == "POSITIVE" else -r["sentiment_score"],
        axis=1,
    )
    return df


def run_topics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BERTopic. Returns (per-paragraph topic assignments, topic info)."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    print("  loading sentence-transformer (all-MiniLM-L6-v2)")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  embedding {len(df):,} paragraphs")
    embeddings = embedder.encode(df["text"].tolist(), show_progress_bar=False, batch_size=32)

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=False,
    )
    print("  fitting BERTopic")
    topics, _ = topic_model.fit_transform(df["text"].tolist(), embeddings=embeddings)

    info = topic_model.get_topic_info()
    info = info.rename(columns={"Topic": "topic_id", "Name": "label", "Count": "count"})
    # Attach top 5 words per topic
    top_words = []
    for tid in info["topic_id"]:
        words = topic_model.get_topic(tid)
        top_words.append(", ".join(w for w, _ in (words or [])[:5]))
    info["top_words"] = top_words

    df = df.copy()
    df["topic_id"] = topics
    return df, info


def topic_river(df: pd.DataFrame, info: pd.DataFrame, top_n: int = TOP_N_TOPICS) -> pd.DataFrame:
    """For the top N topics by overall paragraph count, compute year-level
    share of paragraphs assigned to that topic."""
    # Drop the "outlier" topic (-1 in BERTopic) from the top list.
    top = info.query("topic_id >= 0").nlargest(top_n, "count")
    keep_ids = set(top["topic_id"].tolist())

    yr_topic = (
        df.assign(in_top=df["topic_id"].isin(keep_ids))
          .groupby(["year", "topic_id"]).size().reset_index(name="paragraph_count")
    )
    yr_total = df.groupby("year").size().reset_index(name="year_total")
    yr_topic = yr_topic.merge(yr_total, on="year")
    yr_topic["share"] = yr_topic["paragraph_count"] / yr_topic["year_total"]

    # Attach human-readable labels and keep only the top N topics
    yr_topic = yr_topic.merge(
        top[["topic_id", "label", "top_words"]],
        on="topic_id", how="inner",
    )
    return yr_topic[["year", "topic_id", "label", "top_words", "share"]]


def main() -> None:
    print("== Stage 5: NLP ==")
    df = load_corpus_paragraphs()
    if df.empty:
        print("  (no SOTU files in data/corpus/sotu/, run 01_acquire first)")
        return
    print(f"  paragraphs: {len(df):,} across {df['year'].nunique()} years "
          f"({df['year'].min()} to {df['year'].max()})")

    df = run_sentiment(df)
    df.to_parquet(PROC / "sotu_paragraphs.parquet", index=False)
    print(f"  sotu_paragraphs.parquet: {len(df):,} rows with sentiment")

    df, info = run_topics(df)
    df.to_parquet(PROC / "sotu_paragraphs.parquet", index=False)  # overwrite with topics
    info.to_parquet(PROC / "sotu_topics.parquet", index=False)
    print(f"  sotu_topics.parquet: {len(info)} topics")

    river = topic_river(df, info)
    river.to_parquet(PROC / "sotu_topic_river.parquet", index=False)
    print(f"  sotu_topic_river.parquet: {len(river):,} year-topic rows")
    print("Done.")


if __name__ == "__main__":
    main()
