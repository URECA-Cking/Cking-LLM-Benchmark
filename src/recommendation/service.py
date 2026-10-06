"""단일 크리에이터의 M2/M4 유사 추천 후보를 결정적으로 생성한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from src.clients.base import EmbeddingClient
from src.clients.openai_tagger import TagResult
from src.recommendation.cache import ModelCache, text_hash
from src.recommendation.models import (
    MAX_BACKEND_CANDIDATES,
    CreatorProfile,
    RecommendationCandidate,
    RecommendationResult,
)
from src.recommendation.taxonomy import (
    DEFAULT_TAG_PROMPT_VERSION,
    DEFAULT_TAXONOMY_VERSION,
    default_taxonomy_hash,
)


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
    embedding_batch_size: int = 96
    tag_cache_flush_size: int = 100
    embedding_model_version: str = "BAAI/bge-m3@deepinfra-v1"
    tag_model_version: str = "gpt-5.4-nano-2026-03-17"
    tag_prompt_version: str = DEFAULT_TAG_PROMPT_VERSION
    tag_prompt: str = DEFAULT_TAG_PROMPT_VERSION
    taxonomy_version: str = DEFAULT_TAXONOMY_VERSION
    taxonomy_hash: str = field(default_factory=default_taxonomy_hash)
    allowed_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.short_introduction_chars <= 0:
            raise ValueError("short_introduction_chars는 양수여야 합니다.")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or not 1 <= self.top_n <= MAX_BACKEND_CANDIDATES:
            raise ValueError("top_n은 1~100의 정수여야 합니다.")
        if not np.isfinite(self.m4_bonus) or not 0 <= self.m4_bonus <= 1:
            raise ValueError("m4_bonus는 0~1의 유한한 수여야 합니다.")
        if self.score_decimals < 0:
            raise ValueError("score_decimals는 0 이상이어야 합니다.")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size는 양수여야 합니다.")
        if self.tag_cache_flush_size <= 0:
            raise ValueError("tag_cache_flush_size는 양수여야 합니다.")
        for name, value in (
            ("embedding_model_version", self.embedding_model_version),
            ("tag_model_version", self.tag_model_version),
            ("tag_prompt_version", self.tag_prompt_version),
            ("tag_prompt", self.tag_prompt),
            ("taxonomy_version", self.taxonomy_version),
        ):
            if not value.strip():
                raise ValueError(f"{name}은 비어 있을 수 없습니다.")
        if len(self.taxonomy_hash) != 64 or any(char not in "0123456789abcdef" for char in self.taxonomy_hash):
            raise ValueError("taxonomy_hash는 64자리 SHA-256 소문자 hex여야 합니다.")
        for method in ("M2", "M4"):
            if len(self.model_version(method)) > 255:
                raise ValueError(f"{method} modelVersion은 255자를 초과할 수 없습니다.")

    @property
    def tag_cache_version(self) -> str:
        """프롬프트 본문이나 분류체계가 바뀌어도 태그 캐시가 무효화되는 식별자다."""
        prompt_hash = hashlib.sha256(self.tag_prompt.encode("utf-8")).hexdigest()
        return f"{self.tag_prompt_version}:{self.taxonomy_version}:{self.taxonomy_hash}:{prompt_hash}"

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
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_BACKEND_CANDIDATES:
            raise ValueError("top_n은 1~100의 정수여야 합니다.")

        pool = self._deduplicate(seed, candidates)
        if not seed.text or not pool:
            method = "M2"
            model_version = self.config.model_version(method)
            return RecommendationResult(
                creator_id=seed.creator_id,
                method=method,
                model_version=model_version,
                input_hash=self._generation_input_hash(seed, pool, limit, method, model_version),
                candidates=(),
            )

        request_profiles = [seed, *pool]
        method = "M2"
        tags_by_id: dict[int, frozenset[str]] = {}
        if len(seed.text) >= self.config.short_introduction_chars:
            seed_tags = self._tags_many([seed.text])[seed.text]
            if seed_tags:
                method = "M4"
                tags_by_id[seed.creator_id] = seed_tags
                candidate_tags = self._tags_many([candidate.text for candidate in pool])
                for candidate in pool:
                    tags_by_id[candidate.creator_id] = candidate_tags[candidate.text]

        vectors = self._embeddings(request_profiles)
        seed_vector = vectors[seed.creator_id]
        scored: list[tuple[int, float]] = []
        for candidate in pool:
            score = float(np.clip(seed_vector @ vectors[candidate.creator_id], -1.0, 1.0))
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
        return RecommendationResult(
            creator_id=seed.creator_id,
            method=method,
            model_version=model_version,
            input_hash=input_hash,
            candidates=result,
        )

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
            try:
                cached = self.cache.get_embedding(text_hash(text), self.config.embedding_model_version)
                if cached is not None:
                    vector_by_text[text] = self._validated_embeddings(
                        [cached],
                        expected_rows=1,
                        source="캐시",
                    )[0]
            except (TypeError, ValueError, OverflowError):
                cached = None
            if cached is None:
                missing.append(text)

        if missing:
            for start in range(0, len(missing), self.config.embedding_batch_size):
                texts = missing[start : start + self.config.embedding_batch_size]
                generated = self._validated_embeddings(
                    self.embedding_client.embed(texts),
                    expected_rows=len(texts),
                    source="모델 응답",
                )
                records = []
                for text, vector in zip(texts, generated):
                    vector_by_text[text] = vector
                    records.append((text_hash(text), self.config.embedding_model_version, vector.tolist()))
                self.cache.put_embeddings(records)

        return {
            profile.creator_id: vector_by_text[profile.text]
            for profiles_with_text in by_text.values()
            for profile in profiles_with_text
        }

    def _validated_embeddings(self, values: object, expected_rows: int, source: str) -> np.ndarray:
        """캐시와 모델 응답을 같은 계약으로 검증하고 단위 벡터로 정규화한다."""
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(f"{source}의 임베딩 행 수가 요청 텍스트 수와 다릅니다.")
        expected_dim = self.embedding_client.dim
        if vectors.shape[1] != expected_dim:
            raise ValueError(
                f"{source}의 임베딩 차원이 클라이언트 기대값과 다릅니다: "
                f"expected={expected_dim}, actual={vectors.shape[1]}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{source}의 임베딩에 유한하지 않은 값이 있습니다.")

        vectors_64 = vectors.astype(np.float64)
        norms = np.linalg.norm(vectors_64, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError(f"{source}의 임베딩에는 norm이 양수인 유효한 벡터가 필요합니다.")
        return (vectors_64 / norms[:, np.newaxis]).astype(np.float32)

    def _tags_many(self, texts: list[str]) -> dict[str, frozenset[str]]:
        by_text: dict[str, frozenset[str]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self.cache.get_tags(
                text_hash(text),
                self.config.tag_model_version,
                self.config.tag_cache_version,
            )
            if cached is None:
                missing.append(text)
            else:
                by_text[text] = frozenset(cached)

        pending_records: list[tuple[str, str, str, tuple[str, ...]]] = []
        try:
            for text in missing:
                result = self.tagging_client.tag(text)
                tags = tuple(sorted(set(result.tags)))
                if self.config.allowed_tags and not set(tags) <= self.config.allowed_tags:
                    unknown = sorted(set(tags) - self.config.allowed_tags)
                    raise ValueError(f"분류체계에 없는 태그가 반환됐습니다: {unknown}")
                by_text[text] = frozenset(tags)
                pending_records.append(
                    (
                        text_hash(text),
                        self.config.tag_model_version,
                        self.config.tag_cache_version,
                        tags,
                    )
                )
                if len(pending_records) == self.config.tag_cache_flush_size:
                    self.cache.put_tag_records(pending_records)
                    pending_records = []
        except Exception:
            self.cache.put_tag_records(pending_records)
            raise
        self.cache.put_tag_records(pending_records)
        return by_text

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
            "taxonomyHash": self.config.taxonomy_hash,
            "tagCacheVersion": self.config.tag_cache_version,
        }
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
