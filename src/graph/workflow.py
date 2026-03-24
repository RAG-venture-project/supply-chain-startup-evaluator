"""LangGraph 워크플로우 정의.

플로우:
  스타트업 선택 → [후보 남음?]
    → Yes: 기술요약 / 시장성 / 경쟁사 / 팀평가 (병렬)
           → 투자판단 → [추천?]
              → Yes: 추천 보고서 생성 → 종료
              → No:  보류 기록 저장 → 스타트업 선택 (루프)
    → No (전체 소진): 보류 보고서 생성 → 종료
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.competitor import competitor_analysis_agent
from src.agents.investment import investment_node
from src.agents.market_eval import market_eval_agent
from src.agents.report import hold_report_node, recommend_report_node
from src.agents.router import select_startup
from src.agents.team_eval import team_eval_agent
from src.agents.tech_summary import tech_summary_agent
from src.config import INVEST_THRESHOLD
from src.schemas.state import InvestmentState


def _route_after_select(state: InvestmentState) -> Literal["analyze", "hold_report"]:
    if state.get("current_index", 0) < len(state.get("startups", [])):
        return "analyze"
    return "hold_report"


def _dispatch_analysis(_: InvestmentState) -> dict:
    return {}


def _investment_step(state: InvestmentState) -> dict:
    result = investment_node(state)
    score = result["investment_score"]
    decision = result["investment_decision"]
    if decision == "투자 추천":
        reason = f"총점 {score}점으로 투자 기준 {INVEST_THRESHOLD}점을 충족했다."
    else:
        reason = f"총점 {score}점으로 투자 기준 {INVEST_THRESHOLD}점에 미달했다."
    result["investment_reason"] = reason
    return result


def _route_after_investment(
    state: InvestmentState,
) -> Literal["recommend_report", "record_hold"]:
    if state.get("investment_decision") == "투자 추천":
        return "recommend_report"
    return "record_hold"


def _record_hold(state: InvestmentState) -> dict:
    return {
        "hold_records": [{
            "name": state["startup_name"],
            "score": state["investment_score"],
            "reason": state.get("investment_reason", ""),
        }]
    }


def build_graph():
    """워크플로우 그래프를 빌드하여 반환한다."""
    graph = StateGraph(InvestmentState)

    graph.add_node("select_startup", select_startup)
    graph.add_node("dispatch_analysis", _dispatch_analysis)
    graph.add_node("tech_summary", tech_summary_agent)
    graph.add_node("market_eval", market_eval_agent)
    graph.add_node("competitor_analysis", competitor_analysis_agent)
    graph.add_node("team_eval", team_eval_agent)
    graph.add_node("investment", _investment_step)
    graph.add_node("record_hold", _record_hold)
    graph.add_node("recommend_report", recommend_report_node)
    graph.add_node("hold_report", hold_report_node)

    graph.add_edge(START, "select_startup")
    graph.add_conditional_edges(
        "select_startup",
        _route_after_select,
        {
            "analyze": "dispatch_analysis",
            "hold_report": "hold_report",
        },
    )

    graph.add_edge("dispatch_analysis", "tech_summary")
    graph.add_edge("dispatch_analysis", "market_eval")
    graph.add_edge("dispatch_analysis", "competitor_analysis")
    graph.add_edge("dispatch_analysis", "team_eval")
    graph.add_edge("tech_summary", "investment")
    graph.add_edge("market_eval", "investment")
    graph.add_edge("competitor_analysis", "investment")
    graph.add_edge("team_eval", "investment")

    graph.add_conditional_edges(
        "investment",
        _route_after_investment,
        {
            "recommend_report": "recommend_report",
            "record_hold": "record_hold",
        },
    )
    graph.add_edge("record_hold", "select_startup")
    graph.add_edge("recommend_report", END)
    graph.add_edge("hold_report", END)

    return graph.compile()
