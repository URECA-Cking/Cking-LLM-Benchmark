"""전체 Creator 추천 manifest 생성, dry-run, apply CLI다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import RESULTS_DIR
from src.recommendation.batch import BatchPaths, RecommendationBatch
from src.recommendation.cache import JsonModelCache
from src.recommendation.http import BackendClientConfig, CkingBackendClient
from src.recommendation.manifest import fetch_manifest, load_manifest, write_manifest
from src.recommendation.runtime import (
    DEFAULT_CATEGORIES_CSV,
    DEFAULT_TAG_MODEL,
    DEFAULT_TAG_PROMPT_VERSION,
    DEFAULT_TAXONOMY_VERSION,
    build_service_tag_prompt,
    create_external_clients,
    load_service_categories,
)
from src.recommendation.service import RecommendationConfig, SimilarCreatorRecommender
from src.recommendation.taxonomy import taxonomy_hash


DEFAULT_BATCH_DIR = RESULTS_DIR / "recommendation" / "batch"
DEFAULT_MANIFEST = DEFAULT_BATCH_DIR / "creator-manifest.json"
DEFAULT_CACHE = RESULTS_DIR / "recommendation" / "model-cache.json"


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--be-base-url", required=True, help="Cking-BE base URL")
    parser.add_argument("--timeout", type=float, default=10.0, help="BE 요청 timeout 초")
    parser.add_argument("--retries", type=int, default=2, help="일시적 BE 오류 재시도 횟수")
    parser.add_argument("--retry-backoff", type=float, default=0.5, help="첫 재시도 대기 초")


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_CSV)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--short-introduction-chars", type=int, default=15)
    parser.add_argument("--m4-bonus", type=float, default=0.2)
    parser.add_argument("--tag-model", default=DEFAULT_TAG_MODEL)
    parser.add_argument("--tag-prompt-version", default=DEFAULT_TAG_PROMPT_VERSION)
    parser.add_argument("--taxonomy-version", default=DEFAULT_TAXONOMY_VERSION)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="전체 Cking 크리에이터 유사 추천 오프라인 배치")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="BE 전체 Creator 목록을 고정 manifest로 저장")
    _add_backend_options(manifest)
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    manifest.add_argument("--page-size", type=int, default=100)
    manifest.add_argument("--refresh", action="store_true", help="기존 manifest를 명시적으로 다시 생성")

    dry_run = commands.add_parser("dry-run", help="BE 쓰기 없이 payload와 요약 생성")
    _add_generation_options(dry_run)

    apply = commands.add_parser("apply", help="payload를 생성하고 BE 관리자 API에 적재")
    _add_generation_options(apply)
    _add_backend_options(apply)
    return parser


def _backend_config(args: argparse.Namespace) -> BackendClientConfig:
    return BackendClientConfig(
        args.be_base_url,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        retry_backoff_seconds=args.retry_backoff,
    )


def _run_manifest(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.refresh:
        raise FileExistsError(f"manifest가 이미 있습니다. 다시 생성하려면 --refresh를 사용하세요: {args.output}")
    client = CkingBackendClient(_backend_config(args))
    manifest = fetch_manifest(client, page_size=args.page_size)
    write_manifest(args.output, manifest)
    print(f"[batch-manifest] creators={len(manifest.creators)}, sha256={manifest.manifest_hash}, output={args.output}")
    return 0


def _recommendation_config(args: argparse.Namespace) -> tuple[RecommendationConfig, list[str], str]:
    categories = load_service_categories(args.categories)
    prompt = build_service_tag_prompt(categories)
    codes = [category.code for category in categories]
    config = RecommendationConfig(
        short_introduction_chars=args.short_introduction_chars,
        top_n=args.top_n,
        m4_bonus=args.m4_bonus,
        tag_model_version=args.tag_model,
        tag_prompt_version=args.tag_prompt_version,
        tag_prompt=prompt,
        taxonomy_version=args.taxonomy_version,
        taxonomy_hash=taxonomy_hash(categories),
        allowed_tags=frozenset(codes),
    )
    return config, codes, prompt


def _run_batch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    config, category_codes, prompt = _recommendation_config(args)
    embedding_client, tagging_client = create_external_clients(category_codes, prompt, args.tag_model)
    recommender = SimilarCreatorRecommender(
        embedding_client,
        tagging_client,
        JsonModelCache(args.cache),
        config,
    )
    backend = None
    access_token = None
    if args.command == "apply":
        access_token = os.environ.get("CKING_ADMIN_ACCESS_TOKEN")
        if not access_token:
            raise RuntimeError("apply에는 CKING_ADMIN_ACCESS_TOKEN 환경 변수가 필요합니다.")
        backend = CkingBackendClient(_backend_config(args), access_token=access_token)
    secrets = tuple(
        value
        for value in (
            access_token,
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("DEEPINFRA_API_KEY"),
        )
        if value
    )
    batch = RecommendationBatch(
        manifest,
        recommender,
        BatchPaths(args.output_dir),
        config,
        top_n=args.top_n,
        backend=backend,
        secret_values=secrets,
    )
    summary = batch.run(args.command)
    print(
        f"[batch-{args.command}] creators={summary['creatorCount']}, "
        f"empty={summary['emptyGenerationCount']}, failures={summary['failureCount']}, "
        f"summary={BatchPaths(args.output_dir).summary}"
    )
    return 1 if summary["failureCount"] else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        return _run_manifest(args)
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
