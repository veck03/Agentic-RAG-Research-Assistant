from pydantic import BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's research question")


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    paper_evidence: list[Any] = []
    data_evidence: dict[str, Any] = {}