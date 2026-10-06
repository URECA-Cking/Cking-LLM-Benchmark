"""v0.2 관심 분야 설명을 쿼리로 M3 후보와 M2 비교 기준선을 생성한다."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.clients.base import EmbeddingClient
from src.recommendation.batch import _json_hash
from src.recommendation.cache import ModelCache, text_hash
from src.recommendation.manifest import CreatorManifest, build_manifest
from src.recommendation.taxonomy import (
    DEFAULT_TAXONOMY_VERSION, ServiceCategory, default_taxonomy_hash, taxonomy_hash,
)


M3_METHOD = "INTEREST_M3_V1"
M2_METHOD = "INTEREST_M2_V1"
METHODS = (M2_METHOD, M3_METHOD)


def _integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name}은 {minimum}~{maximum}의 정수여야 합니다.")


@dataclass(frozen=True)
class InterestRecommendationConfig:
    top_n: int = 20
    zero_shot_tau: float = 0.4296248555
    zero_shot_max_tags: int = 3
    m3_bonus: float = 0.1
    score_decimals: int = 8
    embedding_batch_size: int = 96
    embedding_model_version: str = "BAAI/bge-m3@local-v1"
    taxonomy_version: str = DEFAULT_TAXONOMY_VERSION
    taxonomy_hash: str = field(default_factory=default_taxonomy_hash)

    def __post_init__(self) -> None:
        _integer("top_n", self.top_n, 1, 100)
        _integer("zero_shot_max_tags", self.zero_shot_max_tags, 1, 17)
        _integer("score_decimals", self.score_decimals, 0, 15)
        _integer("embedding_batch_size", self.embedding_batch_size, 1, 10000)
        for name, value, minimum in (
            ("zero_shot_tau", self.zero_shot_tau, -1), ("m3_bonus", self.m3_bonus, 0),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not minimum <= value <= 1):
                raise ValueError(f"{name}은 {minimum}~1의 유한한 수여야 합니다.")
        if (not isinstance(self.embedding_model_version, str)
                or not self.embedding_model_version.startswith("BAAI/bge-m3@")
                or not self.embedding_model_version[len("BAAI/bge-m3@"):].strip()
                or self.embedding_model_version != self.embedding_model_version.strip()
                or len(self.embedding_model_version) > 255):
            raise ValueError("embedding_model_version은 버전이 있는 BAAI/bge-m3 식별자여야 합니다.")
        if self.taxonomy_version != DEFAULT_TAXONOMY_VERSION:
            raise ValueError("관심 분야 생성기는 taxonomyVersion=v0.2만 지원합니다.")
        if self.taxonomy_hash != default_taxonomy_hash():
            raise ValueError("taxonomyHash가 v0.2 정본과 다릅니다.")

    def identity(self) -> dict[str, object]:
        """점수·태그·정렬 계약과 결과에 영향을 주는 설정을 모두 해시에 넣는다."""
        return {
            "schemaVersion": 1, "taxonomyVersion": self.taxonomy_version,
            "taxonomyHash": self.taxonomy_hash, "topN": self.top_n,
            "embeddingModelVersion": self.embedding_model_version,
            "zeroShotTau": float(self.zero_shot_tau), "zeroShotMaxTags": self.zero_shot_max_tags,
            "m3Bonus": float(self.m3_bonus), "scoreDecimals": self.score_decimals,
            "zeroShotTieBreak": "taxonomy display order ASC",
            "ranking": "rounded score DESC, creatorId ASC",
            "rounding": "python round, ties to even", "emptyIntroduction": "exclude",
        }


@dataclass(frozen=True)
class InterestRecommendationResult:
    interest_code: str
    method: str
    model_version: str
    input_hash: str
    config: InterestRecommendationConfig
    # (creatorId, 반올림된 점수) 순서 자체가 1부터 시작하는 rank를 결정한다.
    candidates: tuple[tuple[int, float], ...]

    def to_backend_payload(self) -> dict[str, object]:
        metadata = {
            "taxonomyVersion": self.config.taxonomy_version,
            "taxonomyHash": self.config.taxonomy_hash, "interestCode": self.interest_code,
            "method": self.method, "modelVersion": self.model_version, "inputHash": self.input_hash,
        }
        return {
            **metadata,
            "candidates": [
                {"creatorId": creator_id, "rank": rank, "score": score, **metadata}
                for rank, (creator_id, score) in enumerate(self.candidates, 1)
            ],
        }


class InterestRecommender:
    """소개·설명 임베딩 하나를 공유해 17개 독립 쿼리와 zero-shot 태그를 계산한다."""

    def __init__(
        self, manifest: CreatorManifest, categories: Iterable[ServiceCategory],
        embedding_client: EmbeddingClient, cache: ModelCache,
        config: InterestRecommendationConfig | None = None,
    ) -> None:
        self.config = config or InterestRecommendationConfig()
        self.categories = tuple(categories)
        if (len(self.categories) != 17
                or taxonomy_hash(self.categories) != self.config.taxonomy_hash
                or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", row.code) for row in self.categories)):
            raise ValueError("분야 목록은 v0.2의 17개 정본과 일치해야 합니다.")
        normalized = build_manifest(list(manifest.creators), page_size=manifest.page_size)
        if normalized.creators != manifest.creators or normalized.manifest_hash != manifest.manifest_hash:
            raise ValueError("manifest 내용·순서·해시가 일치하지 않습니다.")
        _integer("embedding dim", embedding_client.dim, 1, 100000)
        self.manifest = manifest
        self.embedding_client = embedding_client
        self.cache = cache
        self._pool = tuple(profile for profile in manifest.creators if profile.text)
        self._scores: np.ndarray | None = None
        self._tags: list[frozenset[int]] = []
        self._preparation_error: Exception | None = None

    def input_hash(self, interest_code: str, method: str = M3_METHOD) -> str:
        category = self._category(interest_code)
        if method not in METHODS:
            raise ValueError(f"지원하지 않는 관심 분야 방식입니다: {method}")
        return _json_hash({
            **self.config.identity(), "manifestHash": self.manifest.manifest_hash,
            "interestCode": category.code, "interestName": category.name,
            "interestDescription": category.description, "method": method,
        })

    def _category(self, code: str) -> ServiceCategory:
        for category in self.categories:
            if category.code == code:
                return category
        raise ValueError(f"분류체계에 없는 interestCode입니다: {code}")

    def recommend(self, interest_code: str, method: str = M3_METHOD) -> InterestRecommendationResult:
        input_hash = self.input_hash(interest_code, method)
        index = self.categories.index(self._category(interest_code))
        if self._pool:
            self._prepare()
        scored = []
        for position, profile in enumerate(self._pool):
            score = float(self._scores[position, index])
            if method == M3_METHOD and index in self._tags[position]:
                score += self.config.m3_bonus
            scored.append((profile.creator_id, round(score, self.config.score_decimals)))
        scored.sort(key=lambda row: (-row[1], row[0]))
        return InterestRecommendationResult(
            interest_code, method, self.config.embedding_model_version, input_hash,
            self.config, tuple(scored[:self.config.top_n]),
        )

    def _prepare(self) -> None:
        if self._scores is not None:
            return
        if self._preparation_error is not None:
            raise self._preparation_error
        texts = [row.description for row in self.categories] + [profile.text for profile in self._pool]
        try:
            vectors = self._embeddings(texts)
        except Exception as error:
            # 같은 manifest의 공통 임베딩 실패를 17번 유료 재호출하지 않는다.
            # 체크포인트 재실행의 새 생성기에서 미완료 임베딩만 다시 시도한다.
            self._preparation_error = error
            raise
        # 동일 쌍은 동일한 내적 연산으로 계산한다. 배치 크기는 점수 계약에 영향을 주지 않는다.
        category_vectors = [vectors[row.description] for row in self.categories]
        scores = np.asarray([
            [float(np.clip(vectors[profile.text] @ vector, -1, 1)) for vector in category_vectors]
            for profile in self._pool
        ])
        tags = []
        for row in scores:
            order = sorted(range(len(self.categories)), key=lambda index: (-row[index], index))
            tags.append(frozenset(
                index for index in order[:self.config.zero_shot_max_tags]
                if row[index] >= self.config.zero_shot_tau
            ))
        self._tags = tags
        self._scores = scores

    def _validated(self, values: object, rows: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(values, dtype=np.float64)
        if raw.shape != (rows, self.embedding_client.dim):
            raise ValueError("임베딩 행 수 또는 차원이 요청과 다릅니다.")
        if not np.isfinite(raw).all():
            raise ValueError("임베딩에 유한하지 않은 값이 있습니다.")
        # 큰 유한 벡터도 overflow 없이 정규화하며 영벡터는 명시적으로 거부한다.
        scale = np.max(np.abs(raw), axis=1)
        if np.any(scale <= 0):
            raise ValueError("임베딩 norm은 양수여야 합니다.")
        scaled = raw / scale[:, None]
        return raw, scaled / np.linalg.norm(scaled, axis=1)[:, None]

    def _embeddings(self, texts: list[str]) -> dict[str, np.ndarray]:
        vectors = {}
        missing = []
        for text in dict.fromkeys(texts):
            try:
                cached = self.cache.get_embedding(text_hash(text), self.config.embedding_model_version)
                if cached is not None:
                    vectors[text] = self._validated([cached], 1)[1][0]
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
            missing.append(text)
        for start in range(0, len(missing), self.config.embedding_batch_size):
            batch = missing[start:start + self.config.embedding_batch_size]
            raw, normalized = self._validated(self.embedding_client.embed(batch), len(batch))
            # 원벡터를 저장하고 첫 실행과 캐시 실행에 같은 정규화를 적용한다.
            # 정규화 벡터를 재정규화하면 임계값 경계나 마지막 소수점이 달라질 수 있다.
            self.cache.put_embeddings([
                (text_hash(text), self.config.embedding_model_version, vector.tolist())
                for text, vector in zip(batch, raw)
            ])
            vectors.update(zip(batch, normalized))
        return vectors
