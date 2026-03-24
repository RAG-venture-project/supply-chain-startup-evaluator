"""PDF 문서 로딩 및 텍스트 청킹."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str | Path) -> list[Document]:
    """단일 PDF 파일을 로드한다."""
    loader = PyMuPDFLoader(str(file_path))
    return loader.load()


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
