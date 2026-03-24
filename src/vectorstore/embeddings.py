"""BGE-M3 임베딩 모델 래퍼."""

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """BGE-M3 임베딩 모델을 로드하여 반환한다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
