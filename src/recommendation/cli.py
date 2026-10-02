"""Cking-BE 적재용 단일 크리에이터 추천 후보 JSON을 생성하는 CLI다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.config import CATEGORIES_CSV, RESULTS_DIR
from src.recommendation.cache import JsonModelCache
from src.recommendation.models import MAX_BACKEND_CANDIDATES, CreatorProfile
from src.recommendation.runtime import (
    DEFAULT_TAG_MODEL,
    DEFAULT_TAG_PROMPT_VERSION,
    DEFAULT_TAXONOMY_VERSION,
    build_service_tag_prompt,
    create_external_clients,
    load_service_categories,
)
from src.recommendation.service import RecommendationConfig, SimilarCreatorRecommender


DEFAULT_CACHE = RESULTS_DIR / "recommendation" / "model-cache.json"


def _same_path(first: Path, second: Path) -> bool:
    """정규화된 경로와 기존 파일 식별자를 함께 비교한다."""
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def validate_write_paths(paths: dict[str, Path], writable_options: tuple[str, ...]) -> None:
    """출력·캐시 쓰기가 입력 파일이나 서로의 파일을 덮어쓰지 않게 한다."""
    for writable_option in writable_options:
        writable_path = paths[writable_option]
        for other_option, other_path in paths.items():
            if other_option == writable_option:
                continue
            if _same_path(writable_path, other_path):
                resolved = writable_path.resolve(strict=False)
                raise ValueError(f"{writable_option} 경로는 {other_option} 경로와 달라야 합니다: {resolved}")


def write_backend_payload(path: Path, payload: dict[str, object]) -> None:
    """완성된 JSON만 기존 BE 전달 파일과 원자적으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def load_request(path: Path) -> tuple[CreatorProfile, list[CreatorProfile], int | None]:
    """서비스 입력 JSON을 읽고 seed·후보·선택 Top-N으로 변환한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("입력 JSON 최상위 값은 객체여야 합니다.")
    try:
        seed = CreatorProfile(payload["creatorId"], payload["introduction"])
        raw_candidates = payload["candidateCreators"]
    except KeyError as error:
        raise ValueError(f"입력 JSON 필드가 없습니다: {error.args[0]}") from error
    if not isinstance(raw_candidates, list):
        raise ValueError("candidateCreators는 배열이어야 합니다.")
    try:
        candidates = [
            CreatorProfile(candidate["creatorId"], candidate["introduction"])
            for candidate in raw_candidates
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("candidateCreators의 각 항목에는 creatorId와 introduction이 필요합니다.") from error

    top_n = payload.get("topN")
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= MAX_BACKEND_CANDIDATES
    ):
        raise ValueError("topN은 1~100의 정수여야 합니다.")
    return seed, candidates, top_n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="단일 Cking 크리에이터의 유사 추천 후보 생성")
    parser.add_argument("--input", type=Path, required=True, help="creatorId/introduction/candidateCreators JSON")
    parser.add_argument("--output", type=Path, required=True, help="Cking-BE 적재용 결과 JSON")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="모델 호출 결과 JSON 캐시")
    parser.add_argument("--categories", type=Path, default=CATEGORIES_CSV, help="상위 분야 CSV")
    parser.add_argument("--top-n", type=int, default=5, help="입력 JSON에 topN이 없을 때 후보 수(1~100)")
    parser.add_argument("--short-introduction-chars", type=int, default=15, help="이 값 미만이면 M2 적용")
    parser.add_argument("--m4-bonus", type=float, default=0.2, help="M4 상위 분야 일치 가산점(0~1)")
    parser.add_argument("--tag-model", default=DEFAULT_TAG_MODEL, help="M4 분류 OpenAI 모델")
    parser.add_argument("--tag-prompt-version", default=DEFAULT_TAG_PROMPT_VERSION)
    parser.add_argument("--taxonomy-version", default=DEFAULT_TAXONOMY_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_write_paths(
        {
            "--input": args.input,
            "--output": args.output,
            "--cache": args.cache,
            "--categories": args.categories,
        },
        ("--output", "--cache"),
    )
    seed, candidates, request_top_n = load_request(args.input)
    categories = load_service_categories(args.categories)
    tag_prompt = build_service_tag_prompt(categories)
    config = RecommendationConfig(
        short_introduction_chars=args.short_introduction_chars,
        top_n=args.top_n,
        m4_bonus=args.m4_bonus,
        tag_model_version=args.tag_model,
        tag_prompt_version=args.tag_prompt_version,
        tag_prompt=tag_prompt,
        taxonomy_version=args.taxonomy_version,
        allowed_tags=frozenset(category.code for category in categories),
    )
    embedding_client, tagging_client = create_external_clients(
        [category.code for category in categories],
        tag_prompt,
        args.tag_model,
    )
    recommender = SimilarCreatorRecommender(
        embedding_client,
        tagging_client,
        JsonModelCache(args.cache),
        config,
    )
    result = recommender.recommend(seed, candidates, request_top_n)
    write_backend_payload(args.output, result.to_backend_payload())
    print(f"[recommendation] creatorId={seed.creator_id}, candidates={len(result.candidates)}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
