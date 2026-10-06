"""17개 관심 분야별 Top-20 dry-run/apply 명령이다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import RESULTS_DIR
from src.recommendation.batch_cli import DEFAULT_CACHE, DEFAULT_MANIFEST, _add_backend_options, _backend_config
from src.recommendation.cache import JsonModelCache
from src.recommendation.http import CkingBackendClient
from src.recommendation.interest import METHODS, InterestRecommendationConfig, InterestRecommender, _integer
from src.recommendation.interest_batch import InterestBatchPaths, InterestRecommendationBatch
from src.recommendation.interest_runtime import create_interest_embedding_client
from src.recommendation.manifest import load_manifest
from src.recommendation.locking import batch_locks
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories, taxonomy_hash


DEFAULT_INTEREST_DIR = RESULTS_DIR / "recommendation" / "interests"


def _validate_paths(paths: dict[str, Path]) -> None:
    """정규 경로와 파일 식별자를 한 번씩 읽어 입력·출력·캐시 충돌을 막는다."""
    seen = {}
    for name, path in paths.items():
        keys = [("path", path.resolve(strict=False))]
        try:
            stat = path.stat()
            keys.append(("file", stat.st_dev, stat.st_ino))
        except FileNotFoundError:
            pass
        for key in keys:
            other = seen.get(key)
            if other is not None and (name not in {"--manifest", "--categories"}
                                      or other not in {"--manifest", "--categories"}):
                raise ValueError(f"{name} 경로는 {other} 경로와 달라야 합니다.")
            seen[key] = name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v0.2 관심 분야별 M3 Top-20 생성 및 BE 적재")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "apply"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        sub.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_CSV)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_INTEREST_DIR)
        sub.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        sub.add_argument("--top-n", type=int, default=20)
        sub.add_argument("--evaluation-top-n", type=int, default=10)
        sub.add_argument("--zero-shot-tau", type=float, default=.4296248555)
        sub.add_argument("--zero-shot-max-tags", type=int, default=3)
        sub.add_argument("--m3-bonus", type=float, default=.1)
        sub.add_argument("--score-decimals", type=int, default=8)
        sub.add_argument("--embedding-batch-size", type=int, default=96)
        sub.add_argument("--embedding-provider", choices=("local", "deepinfra"), default="local")
        sub.add_argument("--embedding-model-version", help="BGE-M3 모델/런타임 변경 시 새 버전 사용")
        sub.add_argument("--taxonomy-version", default="v0.2")
        sub.add_argument("--taxonomy-hash", help="BE와 합의한 v0.2 해시를 명시적으로 검증")
        if command == "apply":
            _add_backend_options(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    categories = load_service_categories(args.categories)
    config = InterestRecommendationConfig(
        top_n=args.top_n, zero_shot_tau=args.zero_shot_tau,
        zero_shot_max_tags=args.zero_shot_max_tags, m3_bonus=args.m3_bonus,
        score_decimals=args.score_decimals, embedding_batch_size=args.embedding_batch_size,
        embedding_model_version=args.embedding_model_version or f"BAAI/bge-m3@{args.embedding_provider}-v1",
        taxonomy_version=args.taxonomy_version, taxonomy_hash=args.taxonomy_hash or taxonomy_hash(categories),
    )
    if taxonomy_hash(categories) != config.taxonomy_hash:
        raise ValueError("분야 CSV의 taxonomyHash가 지정 설정과 다릅니다.")
    _integer("evaluation_top_n", args.evaluation_top_n, 1, config.top_n)
    paths = InterestBatchPaths(args.output_dir)
    named_paths = {"--manifest": args.manifest, "--categories": args.categories, "--cache": args.cache,
                   "checkpoint": paths.checkpoint, "summary": paths.summary,
                   "judge-input": paths.evaluation_dir / "judge-input.json",
                   "provenance": paths.evaluation_dir / "provenance.json"}
    for category in categories:
        for method in METHODS:
            named_paths[f"{category.code}/{method}"] = paths.payload(category.code, method)
    _validate_paths(named_paths)
    with batch_locks(args.output_dir, args.cache):
        return _run_locked_batch(args, categories, config, paths)


def _run_locked_batch(args, categories, config, paths):
    manifest = load_manifest(args.manifest)
    backend = None
    api_key = os.environ.get("CKING_RECOMMENDATION_API_KEY")
    token = os.environ.get("CKING_ADMIN_ACCESS_TOKEN")
    if args.command == "apply":
        if not api_key:
            raise RuntimeError("apply에는 CKING_RECOMMENDATION_API_KEY 환경 변수가 필요합니다.")
        backend = CkingBackendClient(_backend_config(args), access_token=None, recommendation_api_key=api_key)
    recommender = InterestRecommender(
        manifest, categories, create_interest_embedding_client(args.embedding_provider),
        JsonModelCache(args.cache), config,
    )
    summary = InterestRecommendationBatch(
        recommender, paths, backend=backend, evaluation_top_n=args.evaluation_top_n,
        secret_values=tuple(value for value in (api_key, token, os.environ.get("DEEPINFRA_API_KEY"),
                                               os.environ.get("OPENAI_API_KEY")) if value),
    ).run(args.command)
    print(f"[interest-{args.command}] interests={summary['interestCount']}, "
          f"empty={summary['emptyGenerationCount']}, failures={summary['failureCount']}, summary={paths.summary}")
    return 1 if summary["failureCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
