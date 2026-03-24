"""6. 보고서 생성 에이전트 (LLM only — RAG 없음).

- 추천 시: 투자 추천 보고서 생성 (정해진 목차)
- 전체 보류 시: 보류 보고서 생성 (기업별 보류 이유 종합)
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLM_MODEL
from src.schemas.state import InvestmentState
from src.schemas.output import AgentOutput, ReportOutput


# state 필드명 → 에이전트 이름 매핑
AGENT_FIELDS = {
    "tech_summary": "기술",
    "market_analysis": "시장",
    "competitor_analysis": "경쟁사",
    "team_evaluation": "팀",
}

# ── 추천 보고서 프롬프트 ──────────────────────────────────────────────────────

RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 VC 투자 보고서 작성 전문가이다.\n\n"
     "## 규칙\n"
     "- 오직 아래 제공된 평가 데이터만 사용하라. 외부 지식을 절대 추가하지 마라.\n"
     "- 아래 목차를 반드시 순서대로 따르라. 목차를 임의로 추가하거나 생략하지 마라.\n"
     "- Summary는 분석 결과 기반으로 '왜 투자하는가/하지 않는가'를 결론 중심으로 작성하라. 반드시 A4 반 페이지 이내로 작성하라.\n"
     "- 각 섹션은 간결하게 핵심만 작성하라. 전체 보고서가 A4 2페이지를 넘지 않아야 한다.\n"
     "- 근거 없는 긍정적 표현을 사용하지 마라.\n"
     "- 2~5번 섹션은 반드시 해당 카테고리 점수와 항목별 점수표를 포함하라. 예시:\n"
     "  ## 2. 기술 (80/100)\n"
     "  | 항목 | 판정 | 배점 | 근거 |\n"
     "  |------|------|------|------|\n"
     "  | 상용화된 제품 존재 여부 | Yes | 20/20 | SaaS 플랫폼 운영 중 |\n"
     "  | 고객사 KPI 개선 입증 | No | 0/20 | 수치 확인 불가 |\n"
     "- 6번 한계점은 위 점수표에서 No로 판정된 항목을 중심으로 작성하라.\n"
     "- 8번 Reference는 아래 '참조 문서' 목록을 그대로 출력하라. 임의로 생성하지 마라.\n\n"
     "## 목차\n"
     "1. Summary — 평가 결과 요약 (총점, 투자 판단 결론, 핵심 근거)\n"
     "2. 기술 — 제품/기술력 점수표 + 분석\n"
     "3. 팀 구성 — 팀 평가 점수표 + 핵심 인력, 기술 역량\n"
     "4. 시장 분석 — 시장성 점수표 + 시장 규모, 타겟 고객, 성장성\n"
     "5. 경쟁사 — 경쟁 우위 점수표 + 경쟁 구도 및 차별점\n"
     "6. 한계점 — No 판정 항목 기반 현재 제약, 개선 필요 사항\n"
     "7. 결론 — 투자 권고 등급, 종합 투자 의견\n"
     "8. Reference — 참조 문서 목록\n"),
    ("human",
     "스타트업: {startup_name}\n"
     "기본 정보: {startup_info}\n"
     "투자 판단: {decision}\n"
     "총점: {score}/100\n"
     "카테고리별 점수: {category_scores}\n"
     "판단 근거: {reason}\n\n"
     "=== 에이전트별 분석 결과 ===\n{agent_details}\n\n"
     "=== 참조 문서 ===\n{references}\n\n"
     "위 데이터만 사용하여 보고서를 작성하라."),
])

# ── 보류 보고서 프롬프트 ──────────────────────────────────────────────────────

HOLD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 VC 투자 보고서 작성 전문가이다.\n\n"
     "## 규칙\n"
     "- 오직 아래 제공된 보류 기록만 사용하라.\n"
     "- 기업별로 보류 사유와 점수를 명확히 정리하라.\n"
     "- 전체 A4 1페이지 이내로 작성하라.\n\n"
     "## 목차\n"
     "1. Summary — 전체 보류 판단 요약\n"
     "2. 기업별 보류 사유 — 각 기업의 점수 및 보류 근거\n"
     "3. 결론 — 종합 의견\n"),
    ("human",
     "평가 대상 {total}개 기업 전체 보류.\n\n"
     "=== 보류 기록 ===\n{hold_details}\n\n"
     "위 데이터만 사용하여 보류 보고서를 작성하라."),
])


def _build_agent_details(state: InvestmentState) -> tuple[str, list[str]]:
    """4개 에이전트 결과를 보고서용 텍스트로 조합하고 참조 문서를 수집한다."""
    sections = []
    all_refs: list[str] = []
    for field_name, label in AGENT_FIELDS.items():
        output = AgentOutput.model_validate_json(state[field_name])
        score = sum(20 for item in output.checklist if item.answer)
        items = "\n".join(
            f"  - {item.question}: {'Yes (20/20)' if item.answer else 'No (0/20)'} — {item.evidence}"
            for item in output.checklist
        )
        sections.append(f"[{label}] ({score}/100)\n{items}\n요약: {output.summary}")
        all_refs.extend(output.references)

    # 중복 제거 (순서 유지)
    seen = set()
    unique_refs = [r for r in all_refs if not (r in seen or seen.add(r))]
    return "\n\n".join(sections), unique_refs


def _build_hold_details(hold_records: list[dict]) -> str:
    """보류 기록을 텍스트로 조합."""
    lines = []
    for record in hold_records:
        lines.append(
            f"- {record['name']}: {record['score']}점 — {record['reason']}"
        )
    return "\n".join(lines)


# ── LangGraph 노드 함수 ──────────────────────────────────────────────────────

def recommend_report_node(state: InvestmentState) -> dict:
    """투자 추천 보고서 생성 노드."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    agent_details, refs = _build_agent_details(state)
    ref_text = "\n".join(f"- {r}" for r in refs) if refs else "- 내부 평가 데이터"

    response = llm.invoke(
        RECOMMEND_PROMPT.format_messages(
            startup_name=state["startup_name"],
            startup_info=state.get("startup_info", ""),
            decision=state["investment_decision"],
            score=state["investment_score"],
            category_scores=json.dumps(state["checklist_result"], ensure_ascii=False),
            reason=state.get("investment_reason", ""),
            agent_details=agent_details,
            references=ref_text,
        )
    )

    result = ReportOutput(
        startup_name=state["startup_name"],
        report_type="추천 보고서",
        content=response.content,
    )

    return {"report": result.content}


def hold_report_node(state: InvestmentState) -> dict:
    """전체 보류 보고서 생성 노드."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    hold_details = _build_hold_details(state["hold_records"])

    response = llm.invoke(
        HOLD_PROMPT.format_messages(
            total=len(state["hold_records"]),
            hold_details=hold_details,
        )
    )

    result = ReportOutput(
        startup_name="전체",
        report_type="보류 보고서",
        content=response.content,
    )

    return {"report": result.content}
