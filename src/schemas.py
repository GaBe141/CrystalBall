from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    name: str
    family: str | None = None
    mean_rank: float | None = None
    weighted_score: float | None = None
    cv_mae: float | None = None
    cv_rmse: float | None = None


class ReportArtifacts(BaseModel):
    visuals: list[str] = Field(default_factory=list)
    exports: dict[str, list[str]] = Field(default_factory=dict)  # ext -> paths


class LLMConsensusItem(BaseModel):
    model: str
    consensus_score: float
    consensus_rank: int
    votes: int
    score_variance: float | None = None
    confidence: float | None = None


class SeriesSummary(BaseModel):
    base_name: str
    title: str | None = None
    n_obs: int | None = None
    start: str | None = None
    end: str | None = None
    top_model: ModelMetrics | None = None
    metrics: list[ModelMetrics] = Field(default_factory=list)
    artifacts: ReportArtifacts = Field(default_factory=ReportArtifacts)
    llm_top3: list[LLMConsensusItem] = Field(default_factory=list)
    robust_flags: dict[str, float | int | bool | None] | None = None


class ExecutiveSummary(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    total_series: int = 0
    series: list[SeriesSummary] = Field(default_factory=list)
