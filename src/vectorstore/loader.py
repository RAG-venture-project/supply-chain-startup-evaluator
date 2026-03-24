"""PDF 문서 로딩 및 텍스트 청킹."""

import re
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

COMPANY_KEYWORDS: dict[str, list[str]] = {
    "Seoul Robotics": ["Seoul Robotics", "서울로보틱스"],
    "Altana": ["Altana"],
    "테크타카": ["테크타카", "Techtaka", "TechTaka"],
    "Tridge": ["Tridge"],
    "Fabric": ["Fabric"],
}

# 섹션 헤더 패턴: "1. Altana", "02 | Altana", "Altana" 단독 등
_SECTION_HEADER_RE = re.compile(
    r"^(?:\d+[\.\s]*[\|―\-]?\s*)?({companies})".format(
        companies="|".join(
            re.escape(kw)
            for kws in COMPANY_KEYWORDS.values()
            for kw in kws
        )
    )
)


def _detect_company(text: str) -> str | None:
    """페이지 텍스트 상단에서 회사 섹션 헤더를 감지한다.

    첫 5줄 안에 섹션 헤더 패턴이 있으면 해당 회사명을 반환한다.
    """
    for line in text.strip().split("\n")[:5]:
        line = line.strip()
        m = _SECTION_HEADER_RE.match(line)
        if not m:
            continue
        matched_kw = m.group(1)
        for company, keywords in COMPANY_KEYWORDS.items():
            if matched_kw in keywords:
                return company
    return None


def _table_to_text(table: list[list]) -> str:
    """표를 마크다운 테이블 형식의 문자열로 변환한다.

    병합 셀로 인한 None 값을 빈 문자열로 처리하고,
    내용이 없는 행과 열을 제거한다.
    """
    rows = []
    for row in table:
        cleaned = [cell.strip() if isinstance(cell, str) else "" for cell in row]
        if any(cleaned):
            rows.append(cleaned)

    if not rows:
        return ""

    # 빈 열 제거
    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    non_empty_cols = [
        col_idx for col_idx in range(col_count)
        if any(row[col_idx] for row in rows)
    ]
    rows = [[row[i] for i in non_empty_cols] for row in rows]

    if not rows or not rows[0]:
        return ""

    lines = [" | ".join(rows[0])]
    lines.append(" | ".join(["---"] * len(rows[0])))
    for row in rows[1:]:
        lines.append(" | ".join(row))

    return "\n".join(lines)


def load_pdf(file_path: str | Path) -> list[Document]:
    """pdfplumber로 단일 PDF 파일을 로드한다.

    회사 섹션 단위로 페이지를 묶어 하나의 Document로 만든다.
    섹션 헤더를 감지할 때마다 이전 섹션을 flush하고 새 섹션을 시작한다.
    """
    file_path = Path(file_path)
    docs = []

    current_company: str | None = None
    section_parts: list[str] = []
    section_start_page: int = 1

    def flush() -> None:
        if not section_parts or current_company is None:
            return
        content = "\n\n".join(p for p in section_parts if p.strip())
        if content.strip():
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "company": current_company,
                },
            ))

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            plain_text = page.extract_text() or ""
            tables = page.extract_tables()

            table_texts = [_table_to_text(t) for t in tables]
            page_content = "\n\n".join(
                p for p in [plain_text] + table_texts if p.strip()
            )

            detected = _detect_company(plain_text)
            if detected and detected != current_company:
                flush()
                current_company = detected
                section_parts = []

            if page_content.strip():
                section_parts.append(page_content)

        flush()

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """문서를 청크 단위로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_and_chunk_pdf(file_path: str | Path) -> list[Document]:
    """PDF를 로드하고 청킹까지 수행한다."""
    docs = load_pdf(file_path)
    return chunk_documents(docs)
