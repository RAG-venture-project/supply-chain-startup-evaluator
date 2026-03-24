"""5. 투자 판단 에이전트 (LLM only — RAG 없음).

4개 에이전트 결과를 종합하여 체크리스트 + 스코어카드 기반으로
"추천" 또는 "보류"를 판단한다.

가중치:
  - 창업자/팀:  35%
  - 시장성:     30%
  - 제품/기술력: 20%
  - 경쟁 우위:  15%

임계값: 총점 60 이상 → 추천, 미만 → 보류
"""


class InvestmentAgent:
    """투자 판단 에이전트."""

    def __init__(self):
        # TODO: llm 초기화
        ...

    def run(self, tech_result: dict, market_result: dict,
            competitor_result: dict, team_result: dict) -> dict:
        """4개 평가 결과를 종합하여 투자 판단을 내린다."""
        # TODO: 구현
        ...
