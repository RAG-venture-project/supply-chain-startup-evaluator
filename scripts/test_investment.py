"""투자판단 + 보고서 에이전트 단위 테스트.

4개 RAG 에이전트 출력을 mock으로 구성하여 테스트한다.

사용법:
    python -m scripts.test_investment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.schemas.output import AgentOutput, ChecklistItem

# ── Mock 데이터: Altana 기준 ──────────────────────────────────────────────────

mock_tech = AgentOutput(
    agent="tech_summary",
    startup_name="Altana",
    checklist=[
        ChecklistItem(question="상용화된 제품(MVP 이상)이 존재하는가?", answer=True, evidence="SaaS 플랫폼 상용 운영 중, 다수 기업 고객 보유"),
        ChecklistItem(question="특허를 1건 이상 보유하고 있는가?", answer=True, evidence="공급망 그래프 관련 특허 2건 출원"),
        ChecklistItem(question="기존 인프라 변경 없이 도입 가능한가?", answer=True, evidence="API 연동 방식으로 기존 시스템 변경 불필요"),
        ChecklistItem(question="AI/자동화 핵심 기술을 자체 개발했는가?", answer=True, evidence="자체 knowledge graph 엔진 및 AI 모델 보유"),
        ChecklistItem(question="고객사 KPI 개선이 입증되었는가?", answer=False, evidence="구체적 KPI 개선 수치 확인 불가"),
    ],
    summary="5개 항목 중 4개 충족. 자체 기술력은 우수하나 고객 성과 수치 부족.",
    references=["tech_summary_5companies.pdf"],
)

mock_market = AgentOutput(
    agent="market_analysis",
    startup_name="Altana",
    checklist=[
        ChecklistItem(question="TAM이 $10B 이상인가?", answer=True, evidence="글로벌 공급망 가시성 시장 $14.3B 규모"),
        ChecklistItem(question="CAGR이 15% 이상인가?", answer=True, evidence="연평균 성장률 18.2%로 15% 상회"),
        ChecklistItem(question="실제 매출 또는 유료 고객이 존재하는가?", answer=True, evidence="미 국방부, 관세청 등 유료 고객 다수"),
        ChecklistItem(question="글로벌 시장(2개국 이상)에 진출했는가?", answer=True, evidence="미국, 유럽, 아시아 시장 진출"),
        ChecklistItem(question="최근 1년 내 매출/고객 수 증가 추세인가?", answer=True, evidence="전년 대비 매출 2배 성장"),
    ],
    summary="5개 항목 모두 충족. TAM $14.3B, CAGR 18.2%로 시장성 매우 우수.",
    references=["시장성_평가_리포트.pdf"],
)

mock_competitor = AgentOutput(
    agent="competitor_analysis",
    startup_name="Altana",
    checklist=[
        ChecklistItem(question="직접 경쟁사 대비 명확한 차별점이 있는가?", answer=True, evidence="글로벌 공급망 지식 그래프 기반 접근은 경쟁사 대비 차별화"),
        ChecklistItem(question="대형 고객사(Fortune 500)를 확보했는가?", answer=True, evidence="미 정부기관 및 Fortune 500 기업 고객 보유"),
        ChecklistItem(question="네트워크 효과 또는 데이터 해자가 있는가?", answer=True, evidence="데이터 축적에 따른 지식 그래프 강화 구조"),
        ChecklistItem(question="전환 비용이 높은가?", answer=False, evidence="API 기반이라 전환 비용이 상대적으로 낮음"),
        ChecklistItem(question="최근 2년 내 전략적 파트너십을 체결했는가?", answer=True, evidence="미 관세국경보호청(CBP)과 공식 파트너십"),
    ],
    summary="5개 항목 중 4개 충족. 데이터 해자와 정부 파트너십이 강점이나 전환 비용은 낮음.",
    references=["경쟁사비교분석.pdf"],
)

mock_team = AgentOutput(
    agent="team_evaluation",
    startup_name="Altana",
    checklist=[
        ChecklistItem(question="창업자 도메인 경력 3년 이상인가?", answer=True, evidence="CEO Evan Smith, 공급망 분석 분야 7년 경력"),
        ChecklistItem(question="핵심 기술진(CTO)이 별도 존재하는가?", answer=True, evidence="CTO 별도 선임, AI/ML 전문"),
        ChecklistItem(question="글로벌 기업 출신 팀원이 있는가?", answer=True, evidence="전 Palantir, Google 출신 엔지니어 다수"),
        ChecklistItem(question="팀 규모 30명 이상인가?", answer=True, evidence="약 150명 규모"),
        ChecklistItem(question="주요 경영진 이탈 없이 안정적인가?", answer=True, evidence="창업 이후 주요 경영진 변동 없음"),
    ],
    summary="5개 항목 모두 충족. 150명 규모, 글로벌 기업 출신 다수, 경영진 안정적.",
    references=["team_report.pdf"],
)

# ── Mock State 구성 ───────────────────────────────────────────────────────────

mock_state = {
    "startups": ["Altana", "테크타카", "Tridge", "Fabric", "Seoul Robotics"],
    "current_index": 0,
    "startup_name": "Altana",
    "startup_info": "AI 기반 글로벌 공급망 지식 그래프 플랫폼",
    "tech_summary": mock_tech.model_dump_json(),
    "market_analysis": mock_market.model_dump_json(),
    "competitor_analysis": mock_competitor.model_dump_json(),
    "team_evaluation": mock_team.model_dump_json(),
    "checklist_result": {},
    "investment_score": 0.0,
    "investment_decision": "",
    "hold_records": [],
    "report": "",
}


def main():
    from src.agents.investment import investment_node
    from src.agents.report import recommend_report_node, hold_report_node

    # ── 1) 투자 판단 테스트 ───────────────────────────────────────────────────
    print("=" * 60)
    print("[TEST] 투자 판단 에이전트")
    print("=" * 60)

    result = investment_node(mock_state)
    print(f"카테고리별 점수: {result['checklist_result']}")
    print(f"총점: {result['investment_score']}")
    print(f"판단: {result['investment_decision']}")

    # state 업데이트
    mock_state.update(result)

    # ── 2) 보고서 테스트 ──────────────────────────────────────────────────────
    if result["investment_decision"] == "투자 추천":
        print("\n" + "=" * 60)
        print("[TEST] 추천 보고서 생성")
        print("=" * 60)
        report_result = recommend_report_node(mock_state)
        print(report_result["report"])
    else:
        print("\n→ 보류 판정. hold_report_node 테스트로 전환.")
        mock_state["hold_records"] = [
            {"name": "Altana", "score": result["investment_score"], "reason": "테스트 보류 사유"}
        ]
        report_result = hold_report_node(mock_state)
        print(report_result["report"])


if __name__ == "__main__":
    main()
