"""매일 실행하는 manifest 감지·Top-20 생성·API Key 적재 단일 명령이다."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from src.config import RESULTS_DIR
from src.recommendation.batch import BatchPaths, RecommendationBatch, generation_config_hash, _json_hash
from src.recommendation.batch_cli import DEFAULT_CACHE, _add_backend_options, _backend_config, _recommendation_config
from src.recommendation.cache import JsonModelCache
from src.recommendation.daily import DailyPaths, DailyRecommendationBatch
from src.recommendation.http import CkingBackendClient
from src.recommendation.interest import M2_METHOD, M3_METHOD, InterestRecommendationConfig, InterestRecommender
from src.recommendation.interest_batch import InterestBatchPaths, InterestRecommendationBatch
from src.recommendation.interest_runtime import create_interest_embedding_client
from src.recommendation.runtime import DEFAULT_TAG_MODEL, DEFAULT_TAG_PROMPT_VERSION, DEFAULT_CATEGORIES_CSV, LazyOpenAITagger
from src.recommendation.taxonomy import load_service_categories, taxonomy_hash
from src.recommendation.service import SimilarCreatorRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="manifest 변경 감지와 두 추천 배치 통합 실행")
    parser.add_argument("mode", choices=("apply", "dry-run"))
    _add_backend_options(parser)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "recommendation" / "daily")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_CSV)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--short-introduction-chars", type=int, default=15)
    parser.add_argument("--m4-bonus", type=float, default=.2)
    parser.add_argument("--tag-model", default=DEFAULT_TAG_MODEL)
    parser.add_argument("--tag-prompt-version", default=DEFAULT_TAG_PROMPT_VERSION)
    parser.add_argument("--taxonomy-version", default="v0.2")
    parser.add_argument("--embedding-provider", choices=("local", "deepinfra"), default="deepinfra")
    parser.add_argument("--embedding-model-version")
    parser.add_argument("--decision-report", type=Path, help="#44 report.json 또는 public-summary.json")
    return parser


def _interest_selection(args, categories):
    version = args.embedding_model_version or f"BAAI/bge-m3@{args.embedding_provider}-v1"
    selected = {"method": M2_METHOD, "tau": .4296248555, "bonus": 0., "maxTags": 3,
                "modelVersion": version, "taxonomyVersion": "v0.2", "taxonomyHash": taxonomy_hash(categories),
                "decisionVersion": "interest-decision-v1"}
    if args.decision_report:
        report = json.loads(args.decision_report.read_text(encoding="utf-8"))
        decision = report.get("decision", {})
        selected = decision.get("selectedConfig")
        if (not isinstance(selected, dict) or set(selected) != {
                "method", "tau", "bonus", "maxTags", "modelVersion", "taxonomyVersion", "taxonomyHash", "decisionVersion"}
                or decision.get("selectedConfigHash") != _json_hash(selected)
                or selected["method"] not in {M2_METHOD, M3_METHOD}
                or selected["decisionVersion"] != "interest-decision-v1"
                or (selected["method"] == M2_METHOD and selected["bonus"] != 0)
                or (selected["method"] == M3_METHOD and decision.get("provisional") is not False)
                or selected["modelVersion"] != version):
            raise ValueError("품질 평가 selectedConfig와 해시·방식·모델 버전이 실행 계약과 다릅니다.")
    config = InterestRecommendationConfig(
        top_n=args.top_n, zero_shot_tau=selected["tau"], m3_bonus=selected["bonus"],
        zero_shot_max_tags=selected["maxTags"], embedding_model_version=selected["modelVersion"],
        taxonomy_version=selected["taxonomyVersion"], taxonomy_hash=selected["taxonomyHash"],
    )
    return selected, config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # JWT만 있는 환경에서도 새 apply 계약은 키를 요구한다. 모델 생성보다 먼저 확인한다.
    api_key = os.environ.get("CKING_RECOMMENDATION_API_KEY")
    if args.mode == "apply" and not api_key:
        raise RuntimeError("apply에는 CKING_RECOMMENDATION_API_KEY 환경 변수가 필요합니다.")
    backend = CkingBackendClient(_backend_config(args),
                                recommendation_api_key=api_key if args.mode == "apply" else None)
    output = args.output_dir.resolve()
    for path in (args.categories, args.decision_report):
        if path and (path.resolve().is_relative_to(output) or path.resolve() == args.cache.resolve()
                     or (path.exists() and args.cache.exists() and path.samefile(args.cache))):
            raise ValueError("설정 입력 경로는 배치 출력·캐시 경로와 분리해야 합니다.")
    config, _, prompt = _recommendation_config(args)
    categories = load_service_categories(args.categories)
    selected, interest_config = _interest_selection(args, categories)
    config = replace(config, embedding_model_version=interest_config.embedding_model_version)
    secrets = tuple(value for name in ("CKING_RECOMMENDATION_API_KEY", "CKING_ADMIN_ACCESS_TOKEN",
                                      "OPENAI_API_KEY", "DEEPINFRA_API_KEY")
                    if (value := os.environ.get(name)))
    # 캐시는 잠금 획득·skip 판단 이후에 한 번 로드해 두 생성기가 같은 메모리 상태를 쓴다.
    cache = None
    embedding = None

    def shared_models():
        nonlocal cache, embedding
        if cache is None:
            cache = JsonModelCache(args.cache)
            embedding = create_interest_embedding_client(args.embedding_provider)
        return cache, embedding

    def similar_factory(manifest, directory):
        model_cache, client = shared_models()
        tagger = LazyOpenAITagger([row.code for row in categories], prompt, args.tag_model)
        recommender = SimilarCreatorRecommender(client, tagger, model_cache, config)
        return RecommendationBatch(manifest, recommender, BatchPaths(directory), config,
                                   top_n=args.top_n, backend=backend, secret_values=secrets)

    def interest_factory(manifest, directory):
        model_cache, client = shared_models()
        recommender = InterestRecommender(manifest, categories, client, model_cache, interest_config)
        return InterestRecommendationBatch(recommender, InterestBatchPaths(directory), backend=backend,
                                           evaluation_top_n=min(10, args.top_n), secret_values=secrets,
                                           selected_method=selected["method"])

    identity = {"schemaVersion": 1, "similarConfigHash": generation_config_hash(config, args.top_n),
                "interestConfig": interest_config.identity(), "selectedConfig": selected}
    summary = DailyRecommendationBatch(backend, DailyPaths(args.output_dir), args.cache, identity,
                                       similar_factory, interest_factory, page_size=args.page_size).run(args.mode)
    print(f"[daily-{args.mode}] status={summary['status']}, hash={summary['manifestHash']}, "
          f"failures={summary['failureCount']}")
    return 1 if summary["failureCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
