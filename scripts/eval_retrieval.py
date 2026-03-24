"""Retrieval 성능 평가 스크립트 (Hit Rate@K, MRR).

data/ground_truth.json을 기반으로 FAISS retriever 성능을 측정한다.

사용법:
    python -m scripts.eval_retrieval
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore.store import get_retriever

# ── 설정 ──────────────────────────────────────────────────────────────────────

GT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
K = 4  # Hit Rate@K의 K

# category → FAISS 인덱스 이름 매핑
CATEGORY_TO_INDEX = {
    "tech_summary": "tech_summary",
    "market_eval": "market_eval",
    "competitor": "competitor",
    "team_eval": "team_eval",
}


def evaluate():
    with open(GT_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    total = 0
    hits = 0
    reciprocal_ranks: list[float] = []

    # 카테고리별 결과
    category_stats: dict[str, dict] = {}

    # retriever를 카테고리당 1번만 로드 (rate limit 방지)
    retrievers: dict[str, object] = {}
    for cat, index_name in CATEGORY_TO_INDEX.items():
        print(f"[로드] {index_name} retriever...")
        retrievers[cat] = get_retriever(index_name, k=K)
    print()

    for item in tqdm(ground_truth, desc="평가 중"):
        category = item["category"]
        question = item["question"]
        expected_content = item["chunk_content"]

        if category not in retrievers:
            continue

        results = retrievers[category].invoke(question)

        # Hit Rate: 상위 K개 중 정답 청크 내용이 포함되어 있는지
        hit = False
        rr = 0.0
        for rank, doc in enumerate(results, 1):
            # 청크 내용의 앞 100자가 일치하면 정답으로 판정
            if expected_content[:100] in doc.page_content or doc.page_content[:100] in expected_content:
                hit = True
                rr = 1.0 / rank
                break

        total += 1
        if hit:
            hits += 1
        reciprocal_ranks.append(rr)

        # 카테고리별 집계
        if category not in category_stats:
            category_stats[category] = {"total": 0, "hits": 0, "rr_sum": 0.0}
        category_stats[category]["total"] += 1
        if hit:
            category_stats[category]["hits"] += 1
        category_stats[category]["rr_sum"] += rr

    # ── 결과 출력 ─────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Retrieval 성능 평가 (K={K})")
    print(f"{'='*60}")

    for cat, stats in category_stats.items():
        cat_hr = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
        cat_mrr = stats["rr_sum"] / stats["total"] if stats["total"] > 0 else 0
        print(f"\n  [{cat}]")
        print(f"    Hit Rate@{K}: {cat_hr:.4f} ({stats['hits']}/{stats['total']})")
        print(f"    MRR:          {cat_mrr:.4f}")


if __name__ == "__main__":
    evaluate()
