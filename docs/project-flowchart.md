# Agentic RAG: Current Project Flowchart

This diagram shows how the current code in `src/` works from PDF ingestion through retrieval.

## End-to-end architecture

```mermaid
flowchart TD
    START([Start])

    subgraph INGESTION[1. Document ingestion]
        PDF[(PDF papers\ndata/papers/*.pdf)]
        EXTRACT[src/ingestion/pdf_ingestion.py\nextract_text_from_pdf]
        PAGEJSON[(One JSON file per paper\ndata/processed/*.json\npage number + page text)]
        CHUNK[src/ingestion/chunking.py\nclean text + create overlapping chunks]
        CHUNKS[(chunks.jsonl\nchunk_id + paper_id + page + text)]

        PDF --> EXTRACT
        EXTRACT --> PAGEJSON
        PAGEJSON --> CHUNK
        CHUNK --> CHUNKS
    end

    subgraph INDEXING[2. Vector indexing]
        VSTORE[src/retrieval/vector_store.py\ncreate_vector_store]
        EMBEDDOC[SentenceTransformer\nall-MiniLM-L6-v2]
        VECTORDB[(ChromaDB\ndata/vector_db\ncollection: ocean_papers)]

        CHUNKS --> VSTORE
        VSTORE --> EMBEDDOC
        EMBEDDOC --> VECTORDB
    end

    CHUNKS --> BM25INDEX
    subgraph RETRIEVAL[3. Retrieval options]
        QUERY([User question])

        subgraph SEMANTIC[Standalone semantic search]
            SEARCH[src/retrieval/search.py\nsearch]
            QUERYEMBED[Embed query]
            DENSE[(ChromaDB nearest chunks)]
            SEARCHRESULTS[Print semantic results\nwith distance, paper, page]

            QUERY --> SEARCH
            SEARCH --> QUERYEMBED
            QUERYEMBED --> DENSE
            VECTORDB --> DENSE
            DENSE --> SEARCHRESULTS
        end

        subgraph KEYWORD[Standalone keyword search]
            BM25INDEX[Build BM25 index\nfrom chunks.jsonl]
            BM25[src/retrieval/bm25_search.py\nsearch]
            TOKENS[Tokenize query]
            BM25SCORES[Calculate BM25 scores]
            BM25RESULTS[Print keyword results\nwith score, paper, page]

            QUERY --> BM25
            BM25 --> TOKENS
            TOKENS --> BM25SCORES
            BM25INDEX --> BM25SCORES
            BM25SCORES --> BM25RESULTS
        end

        subgraph HYBRID[Combined hybrid search]
            HYBRIDFILE[src/retrieval/hybrid_search.py\nmain]
            DENSEHYBRID[dense_search\nsemantic candidates]
            BM25HYBRID[bm25_search\nkeyword candidates]
            RRF[reciprocal_rank_fusion\ncombine rankings]
            CANDIDATES[Keep top RERANK_K\nhybrid candidates]
            RERANK[src/retrieval/reranker.py\nrerank_results]
            CROSS[CrossEncoder\nms-marco-MiniLM-L-6-v2]
            FINAL[Print FINAL_K results\nRRF score + reranker score + text]

            QUERY --> HYBRIDFILE
            HYBRIDFILE --> DENSEHYBRID
            HYBRIDFILE --> BM25HYBRID
            VECTORDB --> DENSEHYBRID
            CHUNKS --> BM25HYBRID
            DENSEHYBRID --> RRF
            BM25HYBRID --> RRF
            RRF --> CANDIDATES
            CANDIDATES --> RERANK
            CROSS --> RERANK
            RERANK --> FINAL
        end
    end

    subgraph OUTPUTS[Stored outputs]
        PAPERSJSON[(Processed page JSON files)]
        CHUNKFILE[(Processed chunks.jsonl)]
        CHROMA[(Persistent ChromaDB)]
    end

    PAGEJSON -. saved as .-> PAPERSJSON
    CHUNKS -. saved as .-> CHUNKFILE
    VECTORDB -. persisted as .-> CHROMA

    START --> PDF
```

## Hybrid search flow in detail

```mermaid
flowchart TD
    A([User enters a question])
    B[Load chunks.jsonl]
    C[Build BM25 index]
    D[Load SentenceTransformer]
    E[Connect to ChromaDB]
    F[Load CrossEncoder reranker]

    G[Run dense_search]
    H[Embed the question]
    I[Query ChromaDB]
    J[Return top DENSE_K semantic matches]

    K[Run bm25_search]
    L[Tokenize the question]
    M[Score all chunks with BM25]
    N[Return top BM25_K keyword matches]

    O[Run reciprocal_rank_fusion]
    P[Add score based on each result's rank]
    Q[Merge duplicate chunk IDs]
    R[Sort by RRF score]

    S[Select top RERANK_K candidates]
    T[Compare question with each candidate\nusing CrossEncoder]
    U[Sort by reranker score]
    V[Keep top FINAL_K results]
    W([Display paper, page, scores, and text])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F --> K

    G --> H
    H --> I
    I --> J
    J --> O

    K --> L
    L --> M
    M --> N
    N --> O

    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> U
    U --> V
    V --> W
```

## What each module does

| Module | Responsibility | Main input | Main output |
|---|---|---|---|
| `ingestion/pdf_ingestion.py` | Extract text page by page from PDFs | PDF files | One JSON file per paper |
| `ingestion/chunking.py` | Clean text and split it into overlapping chunks | Page JSON files | `chunks.jsonl` |
| `retrieval/vector_store.py` | Embed chunks and store them in ChromaDB | `chunks.jsonl` | Persistent vector collection |
| `retrieval/search.py` | Perform semantic-only retrieval | User query + ChromaDB | Nearest chunks |
| `retrieval/bm25_search.py` | Perform keyword-only retrieval | User query + `chunks.jsonl` | BM25-ranked chunks |
| `retrieval/reranker.py` | Score candidate chunks against the query | Query + candidate chunks | Reranked chunks |
| `retrieval/hybrid_search.py` | Combine semantic and keyword retrieval, then rerank | User query + both indexes | Final ranked results |

## Current execution order

Run the scripts in this order when preparing the data:

1. `python src/ingestion/pdf_ingestion.py`
2. `python src/ingestion/chunking.py`
3. `python src/retrieval/vector_store.py`
4. Run one of the search scripts:
   - `python src/retrieval/search.py` for semantic search
   - `python src/retrieval/bm25_search.py` for BM25 search
   - `python src/retrieval/hybrid_search.py` for combined search and reranking

The hybrid search path is currently the most complete retrieval workflow.
