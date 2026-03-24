"""3. 경쟁사 비교 에이전트 (LLM + RAG).

담당 평가 항목 — 경쟁 우위 (가중치 15%)
 1. 직접 경쟁사 대비 명확한 차별점 여부
 2. 대형 고객사(Fortune 500 또는 동급) 확보 여부
 3. 네트워크 효과 또는 데이터 해자(moat) 여부
 4. 전환 비용(switching cost) 높음 여부
 5. 최근 2년 내 전략적 파트너십 체결 여부

RAG 소스: docs/경쟁사비교분석.pdf
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
COMPETITOR_SYSTEM = """
당신은 Supply Chain 스타트업 경쟁 분석 전문가입니다.
아래 문서를 읽고 반드시 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

출력 형식:
{
  "checklist": [
    {
      "question": "직접 경쟁사 대비 명확한 차별점이 있는가?",
      "answer": true | false,
      "evidence": "근거 1~2문장"
    },
    {
      "question": "대형 고객사(Fortune 500 또는 동급)를 확보하였는가?",
      "answer": true | false,
      "evidence": "근거 1~2문장"
    },
    {
      "question": "네트워크 효과 또는 데이터 해자(moat)가 있는가?",
      "answer": true | false,
      "evidence": "근거 1~2문장"
    },
    {
      "question": "전환 비용(switching cost)이 높은가?",
      "answer": true | false,
      "evidence": "근거 1~2문장"
    },
    {
      "question": "최근 2년 내 전략적 파트너십을 체결하였는가?",
      "answer": true | false,
      "evidence": "근거 1~2문장"
    }
  ],
  "summary": "경쟁 우위 종합 요약 (2~3문장)"
}
"""


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────
def _query_vectorstore(startup_name: str, k: int = 4) -> str:
    """경쟁사 인덱스에서 관련 문서를 검색하여 하나의 문자열로 반환한다."""
    retriever = get_retriever("competitor", k=k, company=startup_name)
    docs = retriever.invoke(f"{startup_name} 경쟁사 차별점 해자 파트너십")
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
        agent="competitor_analysis",
        startup_name=startup_name,
        checklist=checklist,
        summary=raw.get("summary", ""),
    )


# ── LangGraph 노드 함수 ────────────────────────────────────────────────────────
def competitor_analysis_agent(state: InvestmentState) -> dict:
    """스타트업의 경쟁 우위를 평가하고 state를 업데이트한다."""
    startup = state["startup_name"]
    context = _query_vectorstore(startup)
    raw = _call_llm_json(COMPETITOR_SYSTEM, context)
    output = _parse_output(startup, raw)
    return {"competitor_analysis": output.model_dump_json()}
