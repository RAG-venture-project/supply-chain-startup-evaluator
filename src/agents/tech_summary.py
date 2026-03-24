"""1. 기술 요약 에이전트 (LLM + RAG).

담당 평가 항목 — 제품/기술력 (가중치 20%)
 1. 상용화된 제품(MVP 이상) 존재 여부
 2. 특허 보유 여부 (1건 이상)
 3. 기존 인프라 변경 없이 도입 가능 여부
 4. AI/자동화 핵심 기술 자체 개발 여부
 5. 고객사 측정 가능한 성과(KPI 개선) 입증 여부

RAG 소스: docs/tech_summary_5companies.pdf
"""


class TechSummaryAgent:
    """기술 요약 RAG 에이전트."""

    def __init__(self):
        # TODO: retriever, llm 초기화
        ...

    def run(self, startup_name: str) -> dict:
        """스타트업의 기술력을 평가하고 결과를 반환한다."""
        # TODO: 구현
        ...
