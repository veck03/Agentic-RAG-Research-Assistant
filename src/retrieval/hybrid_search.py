from pathlib import Path
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from reranker import rerank_results


# ============================================================
# Configuration
# ============================================================

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

DB_DIR = "data/vector_db"
COLLECTION_NAME = "ocean_papers"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 10
BM25_K = 10

# Number of candidates passed from RRF to the reranker
RERANK_K = 10

# Final number of results returned
FINAL_K = 5

# RRF constant
RRF_K = 60


# ============================================================
# Load chunks
# ============================================================

def load_chunks():
    print("Loading chunks...")

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            chunks.append(json.loads(line))

    return chunks


# ============================================================
# BM25 tokenization
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# Build BM25 index
# ============================================================

def build_bm25(chunks):

    print("Building BM25 index...")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ============================================================
# Dense retrieval
# ============================================================

def dense_search(
    query,
    collection,
    embedding_model,
    top_k=DENSE_K
):
    """
    Retrieve chunks using semantic similarity.
    """

    query_embedding = embedding_model.encode(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    retrieved_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):

        retrieved_results.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "paper_id": metadata["paper_id"],
                "page": metadata["page"],
                "dense_distance": float(distance)
            }
        )

    return retrieved_results


# ============================================================
# BM25 retrieval
# ============================================================

def bm25_search(
    query,
    chunks,
    bm25,
    top_k=BM25_K
):
    """
    Retrieve chunks using keyword-based BM25 search.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    retrieved_results = []

    for index in ranked_indices[:top_k]:

        chunk = chunks[index]

        retrieved_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "paper_id": chunk["paper_id"],
                "page": chunk["page"],
                "bm25_score": float(scores[index])
            }
        )

    return retrieved_results


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_k=RRF_K
):
    """
    Combine Dense and BM25 rankings using
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    Results appearing highly in both retrieval
    systems receive stronger combined scores.
    """

    fused = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "paper_id": result["paper_id"],
                "page": result["page"],
                "rrf_score": 0.0
            }

        fused[chunk_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "paper_id": result["paper_id"],
                "page": result["page"],
                "rrf_score": 0.0
            }

        fused[chunk_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    fused_results = list(fused.values())

    fused_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return fused_results


# ============================================================
# Display results
# ============================================================

def display_results(results):

    print("\n===== FINAL RERANKED RESULTS =====")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Paper: "
            f"{result['paper_id']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print("\n" + result["text"][:1200])

        print("\n" + "=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------------

    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25 = build_bm25(chunks)

    # --------------------------------------------------------
    # Load Cross-Encoder
    # --------------------------------------------------------

    print("Loading cross-encoder model...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    # --------------------------------------------------------
    # Get user query
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    )

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    print("\nRunning dense retrieval...")

    dense_results = dense_search(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
        top_k=DENSE_K
    )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    print("Running BM25 retrieval...")

    bm25_results = bm25_search(
        query=query,
        chunks=chunks,
        bm25=bm25,
        top_k=BM25_K
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    print("Applying RRF...")

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    # --------------------------------------------------------
    # Cross-Encoder reranking
    # --------------------------------------------------------

    print(
        f"Reranking top "
        f"{min(RERANK_K, len(hybrid_results))} "
        f"hybrid candidates..."
    )

    candidates = hybrid_results[:RERANK_K]

    final_results = rerank_results(
        query=query,
        results=candidates,
        model=reranker,
        top_k=FINAL_K
    )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    display_results(final_results)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()