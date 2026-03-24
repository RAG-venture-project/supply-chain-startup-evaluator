"""0. 스타트업 선택 라우터 (단순 함수 — LLM 없음)."""

from src.schemas.state import InvestmentState


def select_startup(state: InvestmentState) -> dict:
    """다음 평가 대상 스타트업을 꺼내서 state에 세팅한다."""
    startups = state.get("startups", [])
    next_index = state.get("current_index", -1) + 1
    if next_index >= len(startups):
        return {"current_index": next_index, "startup_name": ""}

    return {
        "current_index": next_index,
        "startup_name": startups[next_index],
        "startup_info": "",
        "tech_summary": "",
        "market_analysis": "",
        "competitor_analysis": "",
        "team_evaluation": "",
        "checklist_result": {},
        "investment_score": 0.0,
        "investment_decision": "",
        "investment_reason": "",
        "report": "",
    }
