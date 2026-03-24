"""메인 실행 엔트리포인트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _load_startups(data_path: Path) -> list[str]:
    with data_path.open(encoding="utf-8") as f:
        records = json.load(f)
    return [item["name"] for item in records]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supply Chain 스타트업 투자 평가 워크플로우 실행"
    )
    parser.add_argument(
        "startups",
        nargs="*",
        help="평가할 스타트업 이름. 생략하면 data/startups.json 전체를 사용합니다.",
    )
    parser.add_argument(
        "--data",
        default="data/startups.json",
        help="기본 스타트업 목록 JSON 경로",
    )
    return parser.parse_args()


def main():
    from src.graph.workflow import build_graph

    args = _parse_args()
    startups = args.startups or _load_startups(Path(args.data))

    graph = build_graph()
    result = graph.invoke(
        {
            "startups": startups,
            "current_index": -1,
            "hold_records": [],
        }
    )

    print(f"평가 대상: {', '.join(startups)}")
    print(f"최종 판단: {result.get('investment_decision', '')}")
    print(f"최종 점수: {result.get('investment_score', '')}")
    print()
    print(result.get("report", ""))


if __name__ == "__main__":
    main()
