"""4. 팀 평가 에이전트 (LLM + RAG).

담당 평가 항목 — 창업자/팀 (가중치 35%)
 1. 창업자 해당 도메인 경력 3년 이상 여부
 2. 핵심 기술진(CTO 등) 별도 존재 여부
 3. 글로벌 기업(아마존, 구글 등) 출신 팀원 여부
 4. 팀 규모 30명 이상 여부
 5. 주요 경영진 이탈 없이 안정적인지 여부

RAG 소스: docs/team_report.pdf
"""


class TeamEvalAgent:
    """팀 평가 RAG 에이전트."""

    def __init__(self):
        # TODO: retriever, llm 초기화
        ...

    def run(self, startup_name: str) -> dict:
        """스타트업의 팀을 평가하고 결과를 반환한다."""
        # TODO: 구현
        ...
