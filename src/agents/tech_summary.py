"""1. 기술 요약 에이전트 (LLM + RAG).

담당 평가 항목 — 제품/기술력 (가중치 20%)
 1. 상용화된 제품(MVP 이상) 존재 여부
 2. 특허 보유 여부 (1건 이상)
 3. 기존 인프라 변경 없이 도입 가능 여부
 4. AI/자동화 핵심 기술 자체 개발 여부
 5. 고객사 측정 가능한 성과(KPI 개선) 입증 여부

RAG 소스: docs/tech_summary_5companies.pdf
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
TECH_SYSTEM = """
당신은 Supply Chain 스타트업 기술력 평가 전문가입니다.
아래 문서를 읽고 반드시 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

출력 형식:
{
  "checklist": [
    {
      "question": "상용화된 제품(MVP 이상)이 존재하는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "특허를 1건 이상 보유하고 있는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "기존 인프라 변경 없이 도입이 가능한가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "AI/자동화 핵심 기술을 자체 개발하였는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "고객사에서 측정 가능한 성과(KPI 개선)를 입증하였는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    }
  ],
  "summary": "기술력 종합 평가 (체크리스트 5개 항목을 모두 포함하여 반드시 10줄 이상 상세히 작성)"
}
"""


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────
def _query_vectorstore(startup_name: str, k: int = 5) -> str:
    """기술 요약 인덱스에서 관련 문서를 검색하여 하나의 문자열로 반환한다."""
    retriever = get_retriever("tech_summary", k=k)
    docs = retriever.invoke(f"{startup_name} 기술 특허 제품 도입 KPI 성과")
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
        agent="tech_summary",
        startup_name=startup_name,
        checklist=checklist,
        summary=raw.get("summary", ""),
    )


# ── LangGraph 노드 함수 ────────────────────────────────────────────────────────
def tech_summary_agent(state: InvestmentState) -> dict:
    """스타트업의 기술력을 평가하고 state를 업데이트한다."""
    startup = state["startup_name"]
    context = _query_vectorstore(startup)
    raw = _call_llm_json(TECH_SYSTEM, context)
    output = _parse_output(startup, raw)
    return {"tech_summary": output.model_dump_json()}
