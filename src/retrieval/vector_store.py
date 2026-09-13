import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_DIR = "data/vector_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ocean_papers"


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def create_vector_store():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Found {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    ids = [chunk["chunk_id"] for chunk in chunks]

    metadatas = [
        {
            "paper_id": chunk["paper_id"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":
    create_vector_store()