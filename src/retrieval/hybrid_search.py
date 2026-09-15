import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retrieval.reranker import rerank_results


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = "data/processed/chunks.jsonl"
DB_DIR = "data/vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 10
BM25_K = 10
RERANK_K = 10
FINAL_K = 5

RRF_K = 60


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            chunks.append(json.loads(line))

    return chunks


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(chunks):

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, chunks, bm25, top_k=BM25_K):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "paper_id": chunk["paper_id"],
            "page": chunk["page"],
            "bm25_score": float(scores[index])
        })

    return results


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(query, chunks, collection, embedding_model, top_k=DENSE_K):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = []

    for i in range(len(results["documents"][0])):

        output.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "paper_id": results["metadatas"][0][i]["paper_id"],
            "page": results["metadatas"][0][i]["page"],
            "dense_distance": results["distances"][0][i]
        })

    return output


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K
):

    scores = {}
    result_map = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        result_map[chunk_id] = result

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        if chunk_id not in result_map:
            result_map[chunk_id] = result

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for chunk_id, score in ranked:

        result = result_map[chunk_id].copy()

        result["rrf_score"] = score

        results.append(result)

    return results


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

print("Loading retrieval models...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

reranker_model = CrossEncoder(
    RERANKER_MODEL
)

chunks = load_chunks()

bm25 = build_bm25(chunks)

chroma_client = chromadb.PersistentClient(
    path=DB_DIR
)

collection = chroma_client.get_collection(
    name="ocean_papers"
)


# ============================================================
# MAIN HYBRID SEARCH FUNCTION
# ============================================================

def hybrid_search(
    query,
    final_k=FINAL_K
):

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_results = dense_search(
        query,
        chunks,
        collection,
        embedding_model,
        DENSE_K
    )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query,
        chunks,
        bm25,
        BM25_K
    )

    # --------------------------------------------------------
    # Hybrid fusion
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    # --------------------------------------------------------
    # Select candidates for reranking
    # --------------------------------------------------------

    candidates = fused_results[:RERANK_K]

    # --------------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------------

    final_results = rerank_results(
        query,
        candidates,
        reranker_model,
        top_k=final_k
    )

    return final_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "Enter your question: "
    )

    results = hybrid_search(query)

    print("\n===== FINAL RESULTS =====")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            f"Paper: {result['paper_id']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"RRF Score: {result['rrf_score']:.6f}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"\n{result['text'][:1000]}"
        )