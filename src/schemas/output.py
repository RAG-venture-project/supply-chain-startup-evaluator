"""에이전트 출력 포맷 정의."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── 공통: 체크리스트 항목 ─────────────────────────────

class ChecklistItem(BaseModel):
    question: str
    answer: bool
    evidence: str


# ── 4개 RAG 에이전트 공통 출력 ────────────────────────

class AgentOutput(BaseModel):
    agent: str                         # "tech_summary" | "market_analysis" | "competitor_analysis" | "team_evaluation"
    startup_name: str
    checklist: list[ChecklistItem]     # 5개 항목
    summary: str                       # 종합 요약 (보고서용)


# ── 투자판단 에이전트 출력 ────────────────────────────

class InvestmentOutput(BaseModel):
    startup_name: str
    category_scores: dict[str, float]  # {"tech": 80, "market": 60, ...}
    total_score: float                 # 가중합 (0–100)
    decision: str = Field(pattern=r"^(투자 추천|보류)$")
    reason: str                        # 판단 근거


# ── 보고서 에이전트 출력 ──────────────────────────────

class ReportOutput(BaseModel):
    startup_name: str
    report_type: str                   # "추천 보고서" | "보류 보고서"
    content: str                       # 보고서 본문
