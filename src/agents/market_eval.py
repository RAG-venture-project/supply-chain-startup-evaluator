"""2. 시장성 평가 에이전트 (LLM + RAG).

담당 평가 항목 — 시장성 (가중치 30%)
 1. TAM $10B 이상 여부
 2. CAGR 15% 이상 여부
 3. 실제 매출 또는 유료 고객 존재 여부
 4. 글로벌 시장(2개국 이상) 진출 여부
 5. 최근 1년 내 매출/고객 수 증가 추세 여부

RAG 소스: docs/시장성_평가_리포트.pdf
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
MARKET_SYSTEM = """
당신은 Supply Chain 스타트업 시장성 평가 전문가입니다.
아래 문서를 읽고 반드시 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

출력 형식:
{
  "checklist": [
    {
      "question": "TAM(전체 시장 규모)이 $10B(약 10조 원) 이상인가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "시장 연평균 성장률(CAGR)이 15% 이상인가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "실제 매출 또는 유료 고객이 존재하는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "글로벌 시장(2개국 이상)에 진출해 있는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    },
    {
      "question": "최근 1년 내 매출 또는 고객 수의 증가 추세가 확인되는가?",
      "answer": true | false,
      "evidence": "근거 반드시 4문장 이상으로 상세히 작성"
    }
  ],
  "summary": "시장성 종합 평가 (체크리스트 5개 항목을 모두 포함하여 반드시 10줄 이상 상세히 작성)"
}
"""


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────
def _query_vectorstore(startup_name: str, k: int = 5) -> tuple[str, list[str]]:
    """시장성 인덱스에서 관련 문서를 검색하여 (컨텍스트 문자열, 출처 파일명 목록)을 반환한다."""
    retriever = get_retriever("market_eval", k=k)
    docs = retriever.invoke(f"{startup_name} 시장 규모 성장률 매출 고객")
    context = "\n\n".join(doc.page_content for doc in docs)
    references = list(dict.fromkeys(
        doc.metadata.get("file_path", doc.metadata.get("source", "")).split("/")[-1]
        for doc in docs
    ))
    return context, references


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


def _parse_output(startup_name: str, raw: dict, references: list[str]) -> AgentOutput:
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
        agent="market_analysis",
        startup_name=startup_name,
        checklist=checklist,
        summary=raw.get("summary", ""),
        references=references,
    )


# ── LangGraph 노드 함수 ────────────────────────────────────────────────────────
def market_eval_agent(state: InvestmentState) -> dict:
    """스타트업의 시장성을 평가하고 state를 업데이트한다."""
    startup = state["startup_name"]
    context, references = _query_vectorstore(startup)
    raw = _call_llm_json(MARKET_SYSTEM, context)
    output = _parse_output(startup, raw, references)
    return {"market_analysis": output.model_dump_json()}
