"""PDF 문서를 FAISS 벡터스토어에 적재하는 스크립트.

사용법:
    python -m scripts.ingest
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCS_DIR, VECTORSTORE_DIR
from src.vectorstore.loader import load_and_chunk_pdf
from src.vectorstore.store import create_faiss_index, AGENT_INDEX_MAP


def main():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    for index_name, pdf_filename in AGENT_INDEX_MAP.items():
        pdf_path = DOCS_DIR / pdf_filename
        if not pdf_path.exists():
            print(f"[SKIP] 파일 없음: {pdf_path}")
            continue

        print(f"\n{'='*50}")
        print(f"[처리 중] {pdf_filename} → {index_name}")
        print(f"{'='*50}")

        chunks = load_and_chunk_pdf(pdf_path)
        print(f"  - 청크 수: {len(chunks)}")

        # 주석 해제 시 청크 내용 미리보기
        # for i, chunk in enumerate(chunks):
        #     print(f"\n  [청크 {i+1}] (company={chunk.metadata.get('company')})")
        #     print(f"  {'-'*40}")
        #     print(f"  {chunk.page_content}")
        #     print(f"  {'-'*40}")

        create_faiss_index(chunks, index_name)

    print(f"\n모든 인덱스 생성 완료! 저장 위치: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
