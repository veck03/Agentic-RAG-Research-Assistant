from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    """
    Load the cross-encoder model.

    The model is downloaded the first time it is used
    and then loaded from the local Hugging Face cache.
    """

    print("Loading cross-encoder model...")

    model = CrossEncoder(MODEL_NAME)

    return model


def rerank_results(query, results, model, top_k=5):
    """
    Rerank retrieved chunks using a cross-encoder.

    Each result is scored as:

        (query, chunk_text)

    The cross-encoder directly evaluates how relevant
    each chunk is to the user's query.
    """

    if not results:
        return []

    pairs = []

    for result in results:
        pairs.append(
            [query, result["text"]]
        )

    print("Running cross-encoder reranking...")

    scores = model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):

        result_copy = result.copy()

        result_copy["reranker_score"] = float(score)

        reranked_results.append(result_copy)

    # Higher cross-encoder score = more relevant
    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]