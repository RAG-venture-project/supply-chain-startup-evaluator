"""청크 기반 Ground Truth 생성 스크립트.

각 청크를 LLM에 넘겨 질문을 생성하고, 해당 청크를 정답 문서로 매핑한다.
결과는 data/ground_truth.json에 저장된다.

사용법:
    python -m scripts.generate_ground_truth
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import DOCS_DIR, LLM_MODEL
from src.vectorstore.loader import load_and_chunk_pdf
from src.vectorstore.store import AGENT_INDEX_MAP

# ── 설정 ──────────────────────────────────────────────────────────────────────

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
QUESTIONS_PER_CHUNK = 1  # 청크당 생성할 질문 수

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 평가 데이터셋 생성 전문가이다.\n\n"
     "## 규칙\n"
     "- 아래 문서 내용을 읽고, 이 문서를 검색해야만 답할 수 있는 구체적인 질문을 {n}개 생성하라.\n"
     "- 질문은 반드시 문서 내 사실에 기반해야 한다. 일반 상식으로 답할 수 있는 질문은 생성하지 마라.\n"
     "- 질문에 회사명을 반드시 포함하라.\n"
     "- 답변도 문서 내용에서 직접 발췌하여 1~2문장으로 작성하라.\n"
     "- JSON 배열로만 응답하라. 다른 텍스트를 포함하지 마라.\n\n"
     "출력 형식:\n"
     '[{{"question": "...", "answer": "..."}}]'),
    ("human",
     "회사: {company}\n"
     "문서: {file_name}\n"
     "카테고리: {category}\n\n"
     "--- 문서 내용 ---\n{content}"),
])


def generate_qa_from_chunk(llm: ChatOpenAI, chunk, category: str) -> list[dict]:
    """단일 청크에서 Q&A 쌍을 생성한다."""
    company = chunk.metadata.get("company", "Unknown")
    file_name = chunk.metadata.get("file_name", "Unknown")

    response = llm.invoke(
        QA_PROMPT.format_messages(
            n=QUESTIONS_PER_CHUNK,
            company=company,
            file_name=file_name,
            category=category,
            content=chunk.page_content,
        )
    )

    try:
        raw = response.content
        start = raw.find("[")
        end = raw.rfind("]") + 1
        qa_pairs = json.loads(raw[start:end])
    except Exception:
        return []

    results = []
    for qa in qa_pairs:
        results.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "source_file": file_name,
            "company": company,
            "category": category,
            "chunk_content": chunk.page_content,
        })
    return results


def main():
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    all_qa: list[dict] = []

    for index_name, pdf_filename in AGENT_INDEX_MAP.items():
        pdf_path = DOCS_DIR / pdf_filename
        if not pdf_path.exists():
            print(f"[SKIP] {pdf_path}")
            continue

        chunks = load_and_chunk_pdf(pdf_path)
        print(f"\n[{index_name}] {pdf_filename} — {len(chunks)}개 청크")

        for i, chunk in enumerate(chunks):
            company = chunk.metadata.get("company", "Unknown")
            qa_pairs = generate_qa_from_chunk(llm, chunk, index_name)
            all_qa.extend(qa_pairs)
            print(f"  청크 {i+1}/{len(chunks)} ({company}) → {len(qa_pairs)}개 Q&A")

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"\n완료! 총 {len(all_qa)}개 Q&A 생성 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
