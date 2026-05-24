"""Pydantic schemas for VulnMind API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryFilters(BaseModel):
    min_cvss: float | None = Field(None, ge=0.0, le=10.0)
    vendor: str | None = None
    year_range: list[int] | None = Field(None, min_length=2, max_length=2)

    @field_validator("year_range")
    @classmethod
    def validate_year_range(cls, v: list[int] | None) -> list[int] | None:
        if v and len(v) == 2 and v[0] > v[1]:
            raise ValueError("year_range start must be <= end")
        return v


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    filters: QueryFilters | None = None
    stream: bool = True


class SourceAttribution(BaseModel):
    chunk_id: str
    cve_id: str
    text_preview: str
    cvss_score: float | None
    vulnmind_score: float | None
    severity: str | None
    source: str


class RiskScore(BaseModel):
    cve_id: str
    vulnmind_score: float
    components: dict[str, float]
    raw_values: dict[str, Any]
    severity_label: str
    kev_confirmed: bool = False
    kev_date_added: str | None = None
    kev_required_action: str | None = None


class AgentStep(BaseModel):
    tool: str
    tool_input: Any
    observation: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceAttribution] = []
    risk_scores: list[RiskScore] = []
    agent_steps: list[AgentStep] = []
    query_id: str | None = None
    faithfulness: dict | None = None


class IngestRequest(BaseModel):
    delta_days: int | None = Field(None, ge=1, le=365)
    sources: list[str] = Field(
        default=["nvd", "mitre", "epss"],
        description="Which sources to ingest",
    )


class IngestResponse(BaseModel):
    status: str
    counts: dict[str, int]
    message: str


class HealthResponse(BaseModel):
    status: str
    chroma_count: int
    graph_nodes: int
    graph_edges: int
    redis_connected: bool


class SurfaceAnalysisRequest(BaseModel):
    software: list[str] = Field(..., min_length=1, max_length=20)
    include_graph_context: bool = True


class SurfaceAnalysisResponse(BaseModel):
    software_analyzed: list[str]
    total_cves_found: int
    critical_count: int
    high_count: int
    cves: list[dict]
    top_techniques: list[str]
    immediate_action: list[dict]
