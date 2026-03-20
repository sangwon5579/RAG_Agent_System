from __future__ import annotations
import argparse
import sys
from pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Agent System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark on dev set")
    bench_parser.add_argument("--dev", type=Path, default=Path("data/dev.csv"))
    bench_parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark_results.csv")
    )

    build_parser = subparsers.add_parser("build-index", help="Build embedding index")
    build_parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    build_parser.add_argument("--output-dir", type=Path, default=Path("data/index"))

    server_parser = subparsers.add_parser("server", help="Start FastAPI server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark(args.dev, args.output)
    elif args.command == "build-index":
        from src.build_index import build_index

        build_index(args.train, args.output_dir)
    elif args.command == "server":
        import uvicorn

        uvicorn.run(
            "src.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        print("Use 'python -m src.main --help' to see available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()