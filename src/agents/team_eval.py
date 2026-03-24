"""4. 팀 평가 에이전트 (LLM + RAG).

담당 평가 항목 — 창업자/팀 (가중치 35%)
 1. 창업자 해당 도메인 경력 3년 이상 여부
 2. 핵심 기술진(CTO 등) 별도 존재 여부
 3. 글로벌 기업(아마존, 구글 등) 출신 팀원 여부
 4. 팀 규모 30명 이상 여부
 5. 주요 경영진 이탈 없이 안정적인지 여부

RAG 소스: docs/team_report.pdf
"""

import json

from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL
from src.schemas.output import AgentOutput, ChecklistItem
from src.schemas.state import InvestmentState
from src.vectorstore.store import get_retriever

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────
TEAM_EVAL_SYSTEM = """
당신은 Supply Chain 스타트업 팀 평가 전문가입니다.
아래 문서를 읽고 반드시 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

출력 형식:
{
  "checklist": [
    {
      "question": "창업자가 해당 도메인 경력 3년 이상인가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "핵심 기술진(CTO 등)이 별도로 존재하는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "글로벌 기업(아마존, 구글 등) 출신 팀원이 있는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "팀 규모가 30명 이상인가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "주요 경영진 이탈 없이 안정적인가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    }
  ],
  "summary": "팀 역량 종합 평가 (체크리스트 5개 항목을 모두 포함하여 반드시 10줄 이상 상세히 작성)"
}
"""


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────
def _query_vectorstore(startup_name: str, k: int = 4) -> str:
    """팀 평가 인덱스에서 관련 문서를 검색하여 하나의 문자열로 반환한다."""
    retriever = get_retriever("team_eval", k=k)
    docs = retriever.invoke(f"{startup_name} 창업자 팀 경력 조직 안정성")
    return "\n\n".join(doc.page_content for doc in docs)


def _call_llm_json(system_prompt: str, context: str) -> dict:
    """LLM 호출 후 JSON 파싱. 실패 시 빈 dict 반환."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 문서를 바탕으로 분석하라:\n\n{context}"},
    ]
    response = llm.invoke(messages)
    try:
        raw = response.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}


def _parse_output(startup_name: str, raw: dict) -> AgentOutput:
    """LLM 응답 dict → AgentOutput 모델로 변환한다."""
    checklist = [
        ChecklistItem(
            question=item["question"],
            answer=item["answer"],
            evidence=item.get("evidence", ""),
        )
        for item in raw.get("checklist", [])
    ]
    return AgentOutput(
        agent="team_evaluation",
        startup_name=startup_name,
        checklist=checklist,
        summary=raw.get("summary", ""),
    )


# ── LangGraph 노드 함수 ────────────────────────────────────────────────────────
def team_eval_agent(state: InvestmentState) -> dict:
    """스타트업의 팀 역량을 평가하고 state를 업데이트한다."""
    startup = state["startup_name"]
    context = _query_vectorstore(startup)
    raw = _call_llm_json(TEAM_EVAL_SYSTEM, context)
    output = _parse_output(startup, raw)
    return {"team_evaluation": output.model_dump_json()}
