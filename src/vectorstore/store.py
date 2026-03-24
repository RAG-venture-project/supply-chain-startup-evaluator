"""FAISS 벡터스토어 생성 및 로드."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR
from src.vectorstore.embeddings import get_embedding_model

# 에이전트별 인덱스 이름 → PDF 파일 매핑
AGENT_INDEX_MAP = {
    "tech_summary": "tech_summary_5companies.pdf",
    "market_eval": "시장성_평가_리포트.pdf",
    "competitor": "경쟁사비교분석.pdf",
    "team_eval": "team_report.pdf",
}


def create_faiss_index(docs: list[Document], index_name: str) -> FAISS:
    """문서 리스트로 FAISS 인덱스를 생성하고 디스크에 저장한다."""
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embeddings)

    save_path = VECTORSTORE_DIR / index_name
    vectorstore.save_local(str(save_path))
    print(f"[OK] FAISS 인덱스 저장 완료: {save_path}")
    return vectorstore


def load_faiss_index(index_name: str) -> FAISS:
    """디스크에서 FAISS 인덱스를 로드한다."""
    embeddings = get_embedding_model()
    load_path = VECTORSTORE_DIR / index_name
    return FAISS.load_local(
        str(load_path), embeddings, allow_dangerous_deserialization=True
    )


def get_retriever(index_name: str, k: int = 4, company: str | None = None):
    """FAISS 인덱스를 로드하여 retriever를 반환한다."""
    vectorstore = load_faiss_index(index_name)
    search_kwargs: dict = {"k": k}
    if company:
        search_kwargs["filter"] = {"company": company}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
