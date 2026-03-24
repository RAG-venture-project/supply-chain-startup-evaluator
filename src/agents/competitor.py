"""3. 경쟁사 비교 에이전트 (LLM + RAG).

담당 평가 항목 — 경쟁 우위 (가중치 15%)
 1. 직접 경쟁사 대비 명확한 차별점 여부
 2. 대형 고객사(Fortune 500 또는 동급) 확보 여부
 3. 네트워크 효과 또는 데이터 해자(moat) 여부
 4. 전환 비용(switching cost) 높음 여부
 5. 최근 2년 내 전략적 파트너십 체결 여부

RAG 소스: docs/경쟁사비교분석.pdf
"""


class CompetitorAgent:
    """경쟁사 비교 RAG 에이전트."""

    def __init__(self):
        # TODO: retriever, llm 초기화
        ...

    def run(self, startup_name: str) -> dict:
        """스타트업의 경쟁 우위를 평가하고 결과를 반환한다."""
        # TODO: 구현
        ...
