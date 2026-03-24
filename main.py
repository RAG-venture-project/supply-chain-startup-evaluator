"""메인 실행 엔트리포인트."""

from dotenv import load_dotenv

load_dotenv()


def main():
    # TODO: 그래프 빌드 및 실행
    from src.graph.workflow import build_graph

    graph = build_graph()
    # graph.invoke(...)


if __name__ == "__main__":
    main()
