"""단일 크리에이터의 M2/M4 유사 추천 후보를 결정적으로 생성한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from src.clients.base import EmbeddingClient, normalize_rows
from src.clients.openai_tagger import TagResult
from src.recommendation.cache import ModelCache, text_hash
from src.recommendation.models import CreatorProfile, RecommendationCandidate, RecommendationResult


class TaggingClient(Protocol):
    """M4 상위 분야 분류기의 최소 인터페이스다."""

    def tag(self, input_text: str) -> TagResult:
        ...


@dataclass(frozen=True)
class RecommendationConfig:
    """서비스 적용 방식과 캐시 무효화 버전을 한곳에 고정한다."""

    short_introduction_chars: int = 15
    top_n: int = 5
    m4_bonus: float = 0.2
    score_decimals: int = 8
    embedding_model_version: str = "BAAI/bge-m3@deepinfra-v1"
    tag_model_version: str = "gpt-5.4-nano-2026-03-17"
    tag_prompt_version: str = "creator-category-v1"
    tag_prompt: str = "creator-category-v1"
    taxonomy_version: str = "v0.1"
    allowed_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.short_introduction_chars <= 0:
            raise ValueError("short_introduction_chars는 양수여야 합니다.")
        if self.top_n <= 0:
            raise ValueError("top_n은 양수여야 합니다.")
        if not np.isfinite(self.m4_bonus) or self.m4_bonus < 0:
            raise ValueError("m4_bonus는 0 이상의 유한한 수여야 합니다.")
        if self.score_decimals < 0:
            raise ValueError("score_decimals는 0 이상이어야 합니다.")
        for name, value in (
            ("embedding_model_version", self.embedding_model_version),
            ("tag_model_version", self.tag_model_version),
            ("tag_prompt_version", self.tag_prompt_version),
            ("tag_prompt", self.tag_prompt),
            ("taxonomy_version", self.taxonomy_version),
        ):
            if not value.strip():
                raise ValueError(f"{name}은 비어 있을 수 없습니다.")

    @property
    def tag_cache_version(self) -> str:
        """프롬프트 본문이나 분류체계가 바뀌어도 태그 캐시가 무효화되는 식별자다."""
        prompt_hash = hashlib.sha256(self.tag_prompt.encode("utf-8")).hexdigest()
        return f"{self.tag_prompt_version}:{self.taxonomy_version}:{prompt_hash}"

    def model_version(self, method: str) -> str:
        if method == "M2":
            return self.embedding_model_version
        if method == "M4":
            return f"{self.embedding_model_version}+{self.tag_model_version}@{self.tag_prompt_version}"
        raise ValueError(f"지원하지 않는 추천 방식입니다: {method}")


class SimilarCreatorRecommender:
    """주입된 모델 클라이언트와 캐시로 Top-N 유사 후보를 만든다."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        tagging_client: TaggingClient,
        cache: ModelCache,
        config: RecommendationConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.tagging_client = tagging_client
        self.cache = cache
        self.config = config or RecommendationConfig()

    def recommend(
        self,
        seed: CreatorProfile,
        candidates: Iterable[CreatorProfile],
        top_n: int | None = None,
    ) -> RecommendationResult:
        """seed 자신과 중복을 제외하고 점수·ID 순으로 정렬한 후보 묶음을 반환한다.

        빈 seed 소개는 추측하지 않고 빈 결과를 낸다. 일반 소개의 태그가 유효한 빈 결과
        (UNCLASSIFIED)이면 M2로 폴백한다. 모델 API 예외는 잡지 않아 부분 결과가 저장되지
        않게 하며, 호출 전까지 성공한 개별 캐시는 다음 배치 재시도에서 재사용한다.
        """
        limit = self.config.top_n if top_n is None else top_n
        if limit <= 0:
            raise ValueError("top_n은 양수여야 합니다.")

        pool = self._deduplicate(seed, candidates)
        request_profiles = [seed, *pool]
        if not seed.text or not pool:
            return RecommendationResult(seed.creator_id, ())

        method = "M2"
        tags_by_id: dict[int, frozenset[str]] = {}
        if len(seed.text) >= self.config.short_introduction_chars:
            seed_tags = self._tags(seed.text)
            if seed_tags:
                method = "M4"
                tags_by_id[seed.creator_id] = seed_tags
                for candidate in pool:
                    tags_by_id[candidate.creator_id] = self._tags(candidate.text)

        vectors = self._embeddings(request_profiles)
        seed_vector = vectors[seed.creator_id]
        scored: list[tuple[int, float]] = []
        for candidate in pool:
            score = float(seed_vector @ vectors[candidate.creator_id])
            if method == "M4" and tags_by_id[seed.creator_id] & tags_by_id[candidate.creator_id]:
                score += self.config.m4_bonus
            scored.append((candidate.creator_id, round(score, self.config.score_decimals)))

        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = scored[:limit]
        model_version = self.config.model_version(method)
        input_hash = self._generation_input_hash(seed, pool, limit, method, model_version)
        result = tuple(
            RecommendationCandidate(
                creator_id=seed.creator_id,
                similar_creator_id=candidate_id,
                score=score,
                rank=rank,
                method=method,
                model_version=model_version,
                input_hash=input_hash,
            )
            for rank, (candidate_id, score) in enumerate(selected, start=1)
        )
        return RecommendationResult(seed.creator_id, result)

    def _deduplicate(self, seed: CreatorProfile, candidates: Iterable[CreatorProfile]) -> list[CreatorProfile]:
        by_id: dict[int, CreatorProfile] = {}
        for candidate in candidates:
            if candidate.creator_id == seed.creator_id:
                continue
            current = by_id.get(candidate.creator_id)
            if current is not None and current.text != candidate.text:
                raise ValueError(f"같은 creatorId에 서로 다른 소개가 있습니다: {candidate.creator_id}")
            by_id[candidate.creator_id] = candidate
        return [by_id[creator_id] for creator_id in sorted(by_id) if by_id[creator_id].text]

    def _embeddings(self, profiles: list[CreatorProfile]) -> dict[int, np.ndarray]:
        by_text: dict[str, list[CreatorProfile]] = {}
        for profile in profiles:
            by_text.setdefault(profile.text, []).append(profile)

        vector_by_text: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in by_text:
            cached = self.cache.get_embedding(text_hash(text), self.config.embedding_model_version)
            if cached is None:
                missing.append(text)
            else:
                vector_by_text[text] = np.asarray(cached, dtype=np.float32)

        if missing:
            generated = np.asarray(self.embedding_client.embed(missing), dtype=np.float32)
            if generated.ndim != 2 or generated.shape[0] != len(missing):
                raise ValueError("임베딩 응답의 행 수가 요청 텍스트 수와 다릅니다.")
            if generated.shape[1] == 0 or not np.isfinite(generated).all():
                raise ValueError("임베딩 응답에 유효하지 않은 벡터가 있습니다.")
            generated = normalize_rows(generated)
            for text, vector in zip(missing, generated):
                vector_by_text[text] = vector
                self.cache.put_embedding(text_hash(text), self.config.embedding_model_version, vector.tolist())

        dimensions = {vector.shape for vector in vector_by_text.values()}
        if len(dimensions) != 1 or any(len(shape) != 1 or shape[0] == 0 for shape in dimensions):
            raise ValueError("캐시와 모델의 임베딩 차원이 일치하지 않습니다.")
        return {
            profile.creator_id: vector_by_text[profile.text]
            for profiles_with_text in by_text.values()
            for profile in profiles_with_text
        }

    def _tags(self, text: str) -> frozenset[str]:
        input_hash = text_hash(text)
        cached = self.cache.get_tags(
            input_hash,
            self.config.tag_model_version,
            self.config.tag_cache_version,
        )
        if cached is None:
            result = self.tagging_client.tag(text)
            tags = tuple(sorted(set(result.tags)))
            if self.config.allowed_tags and not set(tags) <= self.config.allowed_tags:
                unknown = sorted(set(tags) - self.config.allowed_tags)
                raise ValueError(f"분류체계에 없는 태그가 반환됐습니다: {unknown}")
            self.cache.put_tags(
                input_hash,
                self.config.tag_model_version,
                self.config.tag_cache_version,
                tags,
            )
            cached = tags
        return frozenset(cached)

    def _generation_input_hash(
        self,
        seed: CreatorProfile,
        pool: list[CreatorProfile],
        top_n: int,
        method: str,
        model_version: str,
    ) -> str:
        manifest = {
            "creatorId": seed.creator_id,
            "introduction": seed.text,
            "candidates": [
                {"creatorId": candidate.creator_id, "introduction": candidate.text}
                for candidate in pool
            ],
            "topN": top_n,
            "method": method,
            "modelVersion": model_version,
            "shortIntroductionChars": self.config.short_introduction_chars,
            "m4Bonus": self.config.m4_bonus,
            "scoreDecimals": self.config.score_decimals,
            "taxonomyVersion": self.config.taxonomy_version,
        }
        if method == "M4":
            manifest["tagCacheVersion"] = self.config.tag_cache_version
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
