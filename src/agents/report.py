"""6. 보고서 생성 에이전트 (LLM only — RAG 없음).

- 추천 시: 투자 추천 보고서 생성
- 전체 보류 시: 보류 보고서 생성 (기업별 보류 이유 종합)
"""


class ReportAgent:
    """보고서 생성 에이전트."""

    def __init__(self):
        # TODO: llm 초기화
        ...

    def run(self, decision: dict, hold_records: list | None = None) -> str:
        """투자 추천 보고서 또는 보류 보고서를 생성한다."""
        # TODO: 구현
        ...
