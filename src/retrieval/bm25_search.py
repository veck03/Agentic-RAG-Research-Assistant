import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    """
    Convert text into tokens for BM25.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def build_bm25(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def search(query, chunks, bm25, top_k=5):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(scores[index])
        })

    return results


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    bm25 = build_bm25(chunks)

    query = input("\nEnter your question: ")

    results = search(
        query,
        chunks,
        bm25,
        top_k=5
    )

    print("\n===== BM25 RESULTS =====\n")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"Result {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Paper: {chunk['paper_id']}")
        print(f"Page: {chunk['page']}")
        print(f"\n{chunk['text'][:1000]}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()