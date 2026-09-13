import chromadb
from sentence_transformers import SentenceTransformer


DB_DIR = "data/vector_db"
COLLECTION_NAME = "ocean_papers"
MODEL_NAME = "all-MiniLM-L6-v2"


def search(query, top_k=5):
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


def main():
    query = input("\nEnter your question: ")

    results = search(query)

    print("\n===== SEARCH RESULTS =====\n")

    for i in range(len(results["documents"][0])):
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"Result {i + 1}")
        print(f"Paper: {metadata['paper_id']}")
        print(f"Page: {metadata['page']}")
        print(f"Distance: {distance:.4f}")
        print(f"\n{document[:1000]}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()