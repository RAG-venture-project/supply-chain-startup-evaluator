"""2. 시장성 평가 에이전트 (LLM + RAG).

담당 평가 항목 — 시장성 (가중치 30%)
 1. TAM $10B 이상 여부
 2. CAGR 15% 이상 여부
 3. 실제 매출 또는 유료 고객 존재 여부
 4. 글로벌 시장(2개국 이상) 진출 여부
 5. 최근 1년 내 매출/고객 수 증가 추세 여부

RAG 소스: docs/시장성_평가_리포트.pdf
"""


class MarketEvalAgent:
    """시장성 평가 RAG 에이전트."""

    def __init__(self):
        # TODO: retriever, llm 초기화
        ...

    def run(self, startup_name: str) -> dict:
        """스타트업의 시장성을 평가하고 결과를 반환한다."""
        # TODO: 구현
        ...
