"""5. 투자 판단 에이전트 (LLM only — RAG 없음).

4개 에이전트 결과를 종합하여 체크리스트 + 스코어카드 기반으로
"투자 추천" 또는 "보류"를 판단한다.

가중치:
  - 창업자/팀:  35%
  - 시장성:     30%
  - 제품/기술력: 20%
  - 경쟁 우위:  15%

임계값: 총점 60 이상 → 투자 추천, 미만 → 보류
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLM_MODEL, EVAL_WEIGHTS, INVEST_THRESHOLD
from src.schemas.state import InvestmentState
from src.schemas.output import AgentOutput, InvestmentOutput


# state 필드명 → 가중치 키 매핑
FIELD_TO_WEIGHT_KEY = {
    "tech_summary": "tech",
    "market_analysis": "market",
    "competitor_analysis": "competitor",
    "team_evaluation": "team",
}

REASON_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 보수적 성향의 VC 투자 심사역이다.\n\n"
     "## 규칙\n"
     "- 오직 아래 제공된 평가 데이터만 근거로 사용하라. 사전 지식이나 외부 정보를 절대 참고하지 마라.\n"
     "- 근거가 불충분한 항목은 무조건 부정적으로 판단하라. '가능성', '추정', '~일 수 있음'은 인정하지 않는다.\n"
     "- 점수가 높더라도 치명적 약점(팀 리스크, 매출 부재 등)이 있으면 반드시 지적하라.\n"
     "- 이전 스타트업 평가 결과에 영향받지 마라. 이 스타트업만 독립적으로 판단하라.\n"
     "- 판단 근거를 3~5문장으로 작성하되, 각 문장에 구체적 수치나 사실을 포함하라.\n"),
    ("human",
     "스타트업: {startup_name}\n"
     "카테고리별 점수: {category_scores}\n"
     "총점: {total_score}/100\n"
     "판단: {decision}\n\n"
     "각 에이전트 요약:\n{summaries}\n\n"
     "위 데이터만 근거로 판단 근거를 작성하라."),
])


def _parse_agent_output(raw: str) -> AgentOutput:
    """state에 저장된 JSON 문자열을 AgentOutput으로 파싱."""
    return AgentOutput.model_validate_json(raw)


def _calc_category_score(output: AgentOutput) -> float:
    """체크리스트 answer 기반 카테고리 점수 계산 (0~100)."""
    true_count = sum(1 for item in output.checklist if item.answer)
    return true_count * 20.0


def investment_node(state: InvestmentState) -> dict:
    """투자 판단 노드 — LangGraph 노드 함수."""

    # 1) 4개 에이전트 결과 파싱 + 카테고리별 점수 계산
    category_scores: dict[str, float] = {}
    summaries: list[str] = []

    for field_name, weight_key in FIELD_TO_WEIGHT_KEY.items():
        output = _parse_agent_output(state[field_name])
        score = _calc_category_score(output)
        category_scores[weight_key] = score
        summaries.append(f"[{weight_key}] ({score}점) {output.summary}")

    # 2) 가중합 총점
    total_score = sum(
        category_scores[key] * weight
        for key, weight in EVAL_WEIGHTS.items()
    )

    # 3) 판단
    decision = "투자 추천" if total_score >= INVEST_THRESHOLD else "보류"

    # 4) LLM으로 판단 근거 생성
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    reason_response = llm.invoke(
        REASON_PROMPT.format_messages(
            startup_name=state["startup_name"],
            category_scores=json.dumps(category_scores, ensure_ascii=False),
            total_score=round(total_score, 1),
            decision=decision,
            summaries="\n".join(summaries),
        )
    )

    # 5) 출력 구성
    result = InvestmentOutput(
        startup_name=state["startup_name"],
        category_scores=category_scores,
        total_score=round(total_score, 1),
        decision=decision,
        reason=reason_response.content,
    )

    return {
        "checklist_result": result.category_scores,
        "investment_score": result.total_score,
        "investment_decision": result.decision,
    }
