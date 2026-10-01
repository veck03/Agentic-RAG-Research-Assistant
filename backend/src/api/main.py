import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.src.agent.orchestrator import run_agent
from backend.src.api.schemas import QueryRequest, QueryResponse


app = FastAPI(
    title="Agentic Scientific Research Assistant",
    description="API for the ocean and climate research assistant.",
    version="1.0.0",
)


# Allow the future React frontend to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Agentic Scientific Research Assistant API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/api/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    try:
        result = run_agent(request.query)

        return QueryResponse(
            query=request.query,
            route=result.get("route", "UNKNOWN"),
            answer=result.get("answer", ""),
            paper_evidence=result.get("paper_evidence", []),
            data_evidence=result.get("data_evidence", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        )