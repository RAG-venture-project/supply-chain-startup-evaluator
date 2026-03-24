"""LangGraph 워크플로우에서 사용하는 State 정의."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class InvestmentState(TypedDict):
    """LangGraph 전체 워크플로우 상태."""

    # ── 루프 제어 ─────────────────────────────────
    startups: list[str]                # 평가 대상 스타트업 리스트 (최대 5개)
    current_index: int                 # 현재 평가 중인 인덱스

    # ── 현재 스타트업 평가 ────────────────────────
    startup_name: str                  # 평가 대상 스타트업명
    startup_info: str                  # 스타트업 기본 정보 (탐색 결과)

    # ── 에이전트별 분석 결과 ──────────────────────
    tech_summary: str                  # 기술 요약 결과
    market_analysis: str               # 시장성 평가 결과
    competitor_analysis: str           # 경쟁사 비교 결과
    team_evaluation: str               # 팀 평가 결과

    # ── 투자 판단 ─────────────────────────────────
    checklist_result: dict             # Bessemer Checklist 결과
    investment_score: float            # Scorecard 총점 (0–100)
    investment_decision: str           # "투자 추천" | "보류"

    # ── 보류 누적 ─────────────────────────────────
    hold_records: Annotated[list[dict], add]  # 보류된 스타트업별 {name, score, reason}

    # ── 최종 산출물 ───────────────────────────────
    report: str                        # 최종 투자 보고서
