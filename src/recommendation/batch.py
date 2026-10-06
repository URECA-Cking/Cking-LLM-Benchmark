"""고정 manifest 전체에 추천 생성과 선택적 BE 적재를 수행한다."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.recommendation.http import JsonBackend
from src.recommendation.manifest import CreatorManifest
from src.recommendation.models import CreatorProfile, RecommendationResult
from src.recommendation.embeddings import EMBEDDING_CONTRACT_VERSION
from src.recommendation.service import RecommendationConfig


CHECKPOINT_SCHEMA_VERSION = 1


class Recommender(Protocol):
    def recommend(
        self,
        seed: CreatorProfile,
        candidates: tuple[CreatorProfile, ...],
        top_n: int | None = None,
    ) -> RecommendationResult:
        ...


@dataclass(frozen=True)
class BatchPaths:
    output_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.output_dir / "checkpoint.json"

    @property
    def summary(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def payload_dir(self) -> Path:
        return self.output_dir / "payloads"

    def payload(self, creator_id: int) -> Path:
        return self.payload_dir / f"creator-{creator_id}.json"


def generation_config_hash(config: RecommendationConfig, top_n: int) -> str:
    """체크포인트 재사용 여부를 결정하는 생성 설정 SHA-256이다."""
    identity = {
        "embeddingContract": EMBEDDING_CONTRACT_VERSION,
        "topN": top_n,
        "shortIntroductionChars": config.short_introduction_chars,
        "m4Bonus": config.m4_bonus,
        "scoreDecimals": config.score_decimals,
        "embeddingModelVersion": config.embedding_model_version,
        "tagModelVersion": config.tag_model_version,
        "tagPromptVersion": config.tag_prompt_version,
        "tagPrompt": config.tag_prompt,
        "taxonomyVersion": config.taxonomy_version,
        "taxonomyHash": config.taxonomy_hash,
        "allowedTags": sorted(config.allowed_tags),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RecommendationBatch:
    """seed별 완료 상태를 원자적으로 남기며 dry-run 또는 apply를 수행한다."""

    def __init__(
        self,
        manifest: CreatorManifest,
        recommender: Recommender,
        paths: BatchPaths,
        config: RecommendationConfig,
        *,
        top_n: int,
        backend: JsonBackend | None = None,
        secret_values: tuple[str, ...] = (),
    ) -> None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
            raise ValueError("batch top_n은 1~100의 정수여야 합니다.")
        self.manifest = manifest
        self.recommender = recommender
        self.paths = paths
        self.config = config
        self.top_n = top_n
        self.backend = backend
        self._secret_values = tuple(value for value in secret_values if value)
        self._config_hash = generation_config_hash(config, top_n)

    def run(self, mode: str) -> dict[str, object]:
        if mode not in {"dry-run", "apply"}:
            raise ValueError("batch mode는 dry-run 또는 apply여야 합니다.")
        if mode == "apply" and self.backend is None:
            raise ValueError("apply에는 Cking-BE 클라이언트가 필요합니다.")

        checkpoint = self._load_checkpoint()
        if mode == "apply":
            self._bind_apply_target(checkpoint)
        generated_now = 0
        reused_generation = 0
        applied_now = 0
        reused_apply = 0

        for seed in self.manifest.creators:
            key = str(seed.creator_id)
            record = checkpoint["creators"].get(key, {})  # type: ignore[index]
            payload = self._reusable_payload(seed, record)
            if payload is None:
                try:
                    result = self.recommender.recommend(seed, self.manifest.creators, self.top_n)
                    payload = result.to_backend_payload()
                    payload_path = self.paths.payload(seed.creator_id)
                    _write_json_atomic(payload_path, payload)
                    record = {
                        "generationStatus": "empty" if not payload["candidates"] else "success",
                        "payloadFile": str(payload_path.relative_to(self.paths.output_dir)).replace("\\", "/"),
                        "payloadSha256": _json_hash(payload),
                        "applyStatus": "pending",
                    }
                    generated_now += 1
                except Exception as error:
                    record = {
                        "generationStatus": "failed",
                        "applyStatus": "pending",
                        "error": self._safe_error("generation", error),
                    }
                    checkpoint["creators"][key] = record  # type: ignore[index]
                    self._save_checkpoint(checkpoint)
                    continue
                checkpoint["creators"][key] = record  # type: ignore[index]
                self._save_checkpoint(checkpoint)
            else:
                reused_generation += 1

            if mode == "apply":
                if record.get("applyStatus") in {"success", "idempotent"}:
                    reused_apply += 1
                    continue
                try:
                    response = self.backend.put(  # type: ignore[union-attr]
                        f"/api/admin/creators/{seed.creator_id}/similar",
                        payload,
                    )
                    self._validate_apply_response(seed, payload, response)
                    record["applyStatus"] = "success" if response["applied"] else "idempotent"
                    record.pop("error", None)
                    applied_now += 1
                except Exception as error:
                    record["applyStatus"] = "failed"
                    record["error"] = self._safe_error("apply", error)
                checkpoint["creators"][key] = record  # type: ignore[index]
                self._save_checkpoint(checkpoint)

        summary = self._build_summary(
            mode,
            checkpoint,
            generated_now=generated_now,
            reused_generation=reused_generation,
            applied_now=applied_now,
            reused_apply=reused_apply,
        )
        _write_json_atomic(self.paths.summary, summary)
        return summary

    def _new_checkpoint(self) -> dict[str, object]:
        return {
            "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
            "manifestHash": self.manifest.manifest_hash,
            "generationConfigHash": self._config_hash,
            "creatorCount": len(self.manifest.creators),
            "applyTarget": None,
            "creators": {},
        }

    def _load_checkpoint(self) -> dict[str, object]:
        if not self.paths.checkpoint.exists():
            return self._new_checkpoint()
        payload = json.loads(self.paths.checkpoint.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schemaVersion") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("지원하지 않는 batch 체크포인트 형식입니다.")
        if payload.get("manifestHash") != self.manifest.manifest_hash:
            raise ValueError("체크포인트의 manifestHash가 현재 manifest와 다릅니다.")
        if payload.get("generationConfigHash") != self._config_hash:
            raise ValueError("체크포인트의 생성 설정이 현재 설정과 다릅니다.")
        if payload.get("creatorCount") != len(self.manifest.creators) or not isinstance(payload.get("creators"), dict):
            raise ValueError("체크포인트 Creator 메타데이터가 올바르지 않습니다.")
        return payload

    def _save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        _write_json_atomic(self.paths.checkpoint, checkpoint)

    def _bind_apply_target(self, checkpoint: dict[str, object]) -> None:
        target = self.backend.target_identity  # type: ignore[union-attr]
        if checkpoint.get("applyTarget") == target:
            return

        records: dict[str, object] = checkpoint["creators"]  # type: ignore[assignment]
        for record in records.values():
            if not isinstance(record, dict):
                continue
            record["applyStatus"] = "pending"
            error = record.get("error")
            if isinstance(error, dict) and error.get("stage") == "apply":
                record.pop("error")
        checkpoint["applyTarget"] = target
        self._save_checkpoint(checkpoint)

    def _reusable_payload(self, seed: CreatorProfile, record: object) -> dict[str, object] | None:
        if not isinstance(record, dict) or record.get("generationStatus") not in {"success", "empty"}:
            return None
        relative = record.get("payloadFile")
        if not isinstance(relative, str):
            return None
        path = self.paths.payload(seed.creator_id)
        expected_relative = str(path.relative_to(self.paths.output_dir)).replace("\\", "/")
        if relative != expected_relative:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("creatorId") != seed.creator_id
            or not isinstance(payload.get("candidates"), list)
            or record.get("payloadSha256") != _json_hash(payload)
        ):
            return None
        return payload

    @staticmethod
    def _validate_apply_response(
        seed: CreatorProfile,
        payload: dict[str, object],
        response: dict[str, object],
    ) -> None:
        candidates = payload["candidates"]
        if (
            response.get("creatorId") != seed.creator_id
            or response.get("inputHash") != payload.get("inputHash")
            or response.get("candidateCount") != len(candidates)  # type: ignore[arg-type]
            or not isinstance(response.get("applied"), bool)
        ):
            raise ValueError("BE 적재 응답이 요청 payload와 일치하지 않습니다.")

    def _safe_error(self, stage: str, error: Exception) -> dict[str, str]:
        message = str(error)
        for secret in self._secret_values:
            message = message.replace(secret, "[REDACTED]")
        return {"stage": stage, "type": type(error).__name__, "message": message}

    def _build_summary(
        self,
        mode: str,
        checkpoint: dict[str, object],
        *,
        generated_now: int,
        reused_generation: int,
        applied_now: int,
        reused_apply: int,
    ) -> dict[str, object]:
        records: dict[str, dict[str, object]] = checkpoint["creators"]  # type: ignore[assignment]
        failures = []
        for creator_id, record in sorted(records.items(), key=lambda item: int(item[0])):
            is_failure = record.get("generationStatus") == "failed" or (
                mode == "apply" and record.get("applyStatus") == "failed"
            )
            if is_failure:
                error = record.get("error")
                if not isinstance(error, dict):
                    error = {"stage": "checkpoint", "type": "ValueError", "message": "오류 정보가 없습니다."}
                failures.append({"creatorId": int(creator_id), **error})
        return {
            "schemaVersion": 1,
            "mode": mode,
            "manifestHash": self.manifest.manifest_hash,
            "generationConfigHash": self._config_hash,
            "applyTarget": checkpoint.get("applyTarget"),
            "creatorCount": len(self.manifest.creators),
            "generatedNowCount": generated_now,
            "reusedGenerationCount": reused_generation,
            "nonEmptyGenerationCount": sum(record.get("generationStatus") == "success" for record in records.values()),
            "emptyGenerationCount": sum(record.get("generationStatus") == "empty" for record in records.values()),
            "appliedNowCount": applied_now,
            "reusedApplyCount": reused_apply,
            "appliedSuccessCount": sum(record.get("applyStatus") == "success" for record in records.values()),
            "idempotentCount": sum(record.get("applyStatus") == "idempotent" for record in records.values()),
            "failureCount": len(failures),
            "failures": failures,
        }


def _json_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
