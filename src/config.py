"""프로젝트 전역 설정."""

from pathlib import Path

# ── 경로 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore_index"

# ── 임베딩 모델 ──────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ── FAISS ─────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── LLM ───────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"  # 필요시 변경

# ── 평가 가중치 ───────────────────────────────────────
EVAL_WEIGHTS = {
    "team": 0.35,       # 창업자/팀
    "market": 0.30,     # 시장성
    "tech": 0.20,       # 제품/기술력
    "competitor": 0.15, # 경쟁 우위
}

# ── 투자 판단 임계값 ──────────────────────────────────
INVEST_THRESHOLD = 60  # 총점 60 이상이면 "추천"
