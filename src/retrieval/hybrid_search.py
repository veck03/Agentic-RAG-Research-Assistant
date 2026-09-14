import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_DIR = "data/vector_db"
COLLECTION_NAME = "ocean_papers"

MODEL_NAME = "all-MiniLM-L6-v2"

DENSE_K = 10
BM25_K = 10
FINAL_K = 5

RRF_K = 60


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def build_bm25(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def dense_search(query, model, collection, top_k):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    output = []

    for rank, chunk_id in enumerate(results["ids"][0], start=1):
        output.append({
            "chunk_id": chunk_id,
            "rank": rank
        })

    return output


def bm25_search(query, chunks, bm25, top_k):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    output = []

    for rank, index in enumerate(ranked_indices, start=1):
        output.append({
            "chunk_id": chunks[index]["chunk_id"],
            "rank": rank
        })

    return output


def reciprocal_rank_fusion(dense_results, bm25_results):
    scores = {}

    for result in dense_results:
        chunk_id = result["chunk_id"]
        rank = result["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += 1 / (RRF_K + rank)

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        rank = result["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += 1 / (RRF_K + rank)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


def main():
    print("Loading chunks...")
    chunks = load_chunks()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    query = input("\nEnter your question: ")

    print("\nRunning dense retrieval...")
    dense_results = dense_search(
        query,
        model,
        collection,
        DENSE_K
    )

    print("Running BM25 retrieval...")
    bm25_results = bm25_search(
        query,
        chunks,
        bm25,
        BM25_K
    )

    print("Applying RRF...")

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    print("\n===== HYBRID RESULTS =====\n")

    for i, (chunk_id, rrf_score) in enumerate(
        fused_results[:FINAL_K],
        start=1
    ):
        chunk = chunk_lookup[chunk_id]

        print(f"Result {i}")
        print(f"RRF Score: {rrf_score:.6f}")
        print(f"Paper: {chunk['paper_id']}")
        print(f"Page: {chunk['page']}")
        print(f"\n{chunk['text'][:1000]}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()