"""관심 분야별 M3 세대 적재와 M2 비교 자료 생성을 체크포인트로 재개한다."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from src.recommendation.batch import _json_hash, _write_json_atomic
from src.recommendation.http import JsonBackend
from src.recommendation.interest import METHODS, M3_METHOD, InterestRecommender, _integer
from src.recommendation.interest_evaluation import write_interest_evaluation


@dataclass(frozen=True)
class InterestBatchPaths:
    output_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.output_dir / "checkpoint.json"

    @property
    def summary(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_dir / "evaluation"

    def payload(self, code: str, method: str = M3_METHOD) -> Path:
        # 코드는 생성기의 v0.2 검증을 통과한 정본 값만 사용한다.
        directory = "payloads" if method == M3_METHOD else "baselines"
        return self.output_dir / directory / f"interest-{code}.json"


class InterestRecommendationBatch:
    def __init__(
        self, recommender: InterestRecommender, paths: InterestBatchPaths, *,
        backend: JsonBackend | None = None, evaluation_top_n: int = 10,
        secret_values: tuple[str, ...] = (),
    ) -> None:
        _integer("evaluation_top_n", evaluation_top_n, 1, recommender.config.top_n)
        self.recommender = recommender
        self.paths = paths
        self.backend = backend
        self.evaluation_top_n = evaluation_top_n
        self._secrets = tuple(value for value in secret_values if value)
        self._config_hash = _json_hash({**recommender.config.identity(), "methods": list(METHODS),
                                      "evaluationTopN": evaluation_top_n})

    def run(self, mode: str) -> dict[str, object]:
        if mode not in {"dry-run", "apply"}:
            raise ValueError("mode는 dry-run 또는 apply여야 합니다.")
        if mode == "apply" and self.backend is None:
            raise ValueError("apply에는 BE 클라이언트가 필요합니다.")
        checkpoint = self._load_checkpoint()
        if mode == "apply" and checkpoint["applyTarget"] != self.backend.target_identity:
            checkpoint["applyTarget"] = self.backend.target_identity
            for record in checkpoint["interests"].values():
                record["applyStatus"] = "pending"
                if record.get("error", {}).get("stage") == "apply":
                    record.pop("error", None)
            self._save(checkpoint)
        generated = reused = applied = skipped = 0
        bundles = {}
        records = checkpoint["interests"]
        for category in self.recommender.categories:
            code = category.code
            record = records.get(code, {})
            bundle = self._reusable_bundle(code, record)
            if bundle is None:
                try:
                    # M2도 같은 Top-N과 manifest로 생성하되 BE에는 M3만 전달한다.
                    bundle = {method: self.recommender.recommend(code, method).to_backend_payload()
                              for method in METHODS}
                    for method, payload in bundle.items():
                        self._validate_payload(code, method, payload)
                    for method, payload in bundle.items():
                        _write_json_atomic(self.paths.payload(code, method), payload)
                    record = {
                        "generationStatus": "success" if bundle[M3_METHOD]["candidates"] else "empty",
                        "payloadHashes": {method: _json_hash(payload) for method, payload in bundle.items()},
                        "applyStatus": "pending",
                    }
                    generated += 1
                except Exception as error:
                    records[code] = {"generationStatus": "failed", "applyStatus": "pending",
                                     "error": self._safe_error("generation", error)}
                    self._save(checkpoint)
                    continue
                records[code] = record
                self._save(checkpoint)
            else:
                reused += 1
            bundles[code] = bundle
            if mode == "apply":
                if record.get("applyStatus") in {"success", "idempotent"}:
                    skipped += 1
                    continue
                try:
                    payload = bundle[M3_METHOD]
                    response = self.backend.put(
                        f"/api/admin/interests/{quote(code, safe='')}/recommendations", payload,
                    )
                    self._validate_response(payload, response)
                    record["applyStatus"] = "success" if response["applied"] else "idempotent"
                    record.pop("error", None)
                    applied += 1
                except Exception as error:
                    record["applyStatus"] = "failed"
                    record["error"] = self._safe_error("apply", error)
                self._save(checkpoint)
        evaluation = {"status": "pending"}
        if len(bundles) == len(self.recommender.categories):
            try:
                evaluation = write_interest_evaluation(
                    self.paths.evaluation_dir, self.recommender, bundles, top_n=self.evaluation_top_n,
                )
            except Exception as error:
                evaluation = {"status": "failed", "error": self._safe_error("evaluation", error)}
        failures = [{"interestCode": row.code, **records[row.code]["error"]}
                    for row in self.recommender.categories
                    if records[row.code]["generationStatus"] == "failed"
                    or (mode == "apply" and records[row.code]["applyStatus"] == "failed")]
        if evaluation["status"] == "failed":
            failures.append(evaluation["error"])
        summary = {
            "schemaVersion": 1, "mode": mode,
            "manifestHash": self.recommender.manifest.manifest_hash,
            "generationConfigHash": self._config_hash,
            "generationConfig": self.recommender.config.identity(),
            "applyTarget": checkpoint["applyTarget"], "interestCount": len(self.recommender.categories),
            "creatorCount": len(self.recommender.manifest.creators),
            "eligibleCreatorCount": sum(bool(profile.text) for profile in self.recommender.manifest.creators),
            "generatedNowCount": generated, "reusedGenerationCount": reused,
            "nonEmptyGenerationCount": sum(row["generationStatus"] == "success" for row in records.values()),
            "emptyGenerationCount": sum(row["generationStatus"] == "empty" for row in records.values()),
            "generationFailureCount": sum(row["generationStatus"] == "failed" for row in records.values()),
            "appliedNowCount": applied, "reusedApplyCount": skipped,
            "appliedSuccessCount": sum(row["applyStatus"] == "success" for row in records.values()),
            "idempotentCount": sum(row["applyStatus"] == "idempotent" for row in records.values()),
            "failureCount": len(failures), "failures": failures,
            "interests": records, "evaluation": evaluation,
        }
        _write_json_atomic(self.paths.summary, summary)
        return summary

    def _save(self, checkpoint: dict[str, object]) -> None:
        _write_json_atomic(self.paths.checkpoint, checkpoint)

    def _load_checkpoint(self) -> dict[str, object]:
        expected = {
            "schemaVersion": 1, "manifestHash": self.recommender.manifest.manifest_hash,
            "generationConfigHash": self._config_hash, "applyTarget": None, "interests": {},
        }
        if not self.paths.checkpoint.exists():
            return expected
        checkpoint = json.loads(self.paths.checkpoint.read_text(encoding="utf-8"))
        if not isinstance(checkpoint, dict) or any(checkpoint.get(key) != expected[key]
                for key in ("schemaVersion", "manifestHash", "generationConfigHash")):
            raise ValueError("체크포인트의 manifest 또는 생성 설정이 현재 계약과 다릅니다.")
        records = checkpoint.get("interests")
        codes = {row.code for row in self.recommender.categories}
        if ("applyTarget" not in checkpoint or not isinstance(records, dict) or not set(records) <= codes
                or any(not isinstance(record, dict)
                       or record.get("generationStatus") not in {"success", "empty", "failed"}
                       or record.get("applyStatus") not in {"pending", "success", "idempotent", "failed"}
                       or ("error" in record and not isinstance(record["error"], dict))
                       or ((record.get("generationStatus") == "failed" or record.get("applyStatus") == "failed")
                           and (not isinstance(record.get("error"), dict)
                                or not all(isinstance(record["error"].get(key), str)
                                           for key in ("stage", "type", "message"))))
                       for record in records.values())):
            raise ValueError("관심 분야 체크포인트 상태가 올바르지 않습니다.")
        return checkpoint

    def _reusable_bundle(self, code: str, record: dict[str, object]) -> dict[str, dict[str, object]] | None:
        if record.get("generationStatus") not in {"success", "empty"}:
            return None
        bundle = {}
        try:
            for method in METHODS:
                payload = json.loads(self.paths.payload(code, method).read_text(encoding="utf-8"))
                self._validate_payload(code, method, payload)
                if record.get("payloadHashes", {}).get(method) != _json_hash(payload):
                    return None
                bundle[method] = payload
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None
        if (record["generationStatus"] == "empty") != (not bundle[M3_METHOD]["candidates"]):
            return None
        return bundle

    def _validate_payload(self, code: str, method: str, payload: dict[str, object]) -> None:
        config = self.recommender.config
        metadata = {
            "taxonomyVersion": config.taxonomy_version, "taxonomyHash": config.taxonomy_hash,
            "interestCode": code, "method": method, "modelVersion": config.embedding_model_version,
            "inputHash": self.recommender.input_hash(code, method),
        }
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in metadata.items()):
            raise ValueError("관심 분야 payload의 세대 메타데이터가 일치하지 않습니다.")
        candidates = payload.get("candidates")
        eligible = {profile.creator_id for profile in self.recommender.manifest.creators if profile.text}
        if not isinstance(candidates, list) or len(candidates) != min(config.top_n, len(eligible)):
            raise ValueError("관심 분야 후보 수가 입력 계약과 다릅니다.")
        seen = set()
        order = []
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                raise ValueError("후보는 JSON 객체여야 합니다.")
            creator_id, score = candidate.get("creatorId"), candidate.get("score")
            if (type(creator_id) is not int or creator_id not in eligible or creator_id in seen
                    or type(candidate.get("rank")) is not int or candidate["rank"] != rank
                    or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                    or not -1 <= score <= round(1 + (config.m3_bonus if method == M3_METHOD else 0), config.score_decimals)
                    or round(score, config.score_decimals) != score
                    or any(candidate.get(key) != value for key, value in metadata.items())):
                raise ValueError("후보 ID·순위·점수·세대 메타데이터가 올바르지 않습니다.")
            seen.add(creator_id)
            order.append((-score, creator_id))
        if order != sorted(order):
            raise ValueError("후보는 score DESC, creatorId ASC 순서여야 합니다.")

    @staticmethod
    def _validate_response(payload: dict[str, object], response: dict[str, object]) -> None:
        if (not isinstance(response, dict)
                or any(response.get(key) != payload[key]
                       for key in ("taxonomyVersion", "interestCode", "inputHash"))
                or ("taxonomyHash" in response and response["taxonomyHash"] != payload["taxonomyHash"])
                or type(response.get("candidateCount")) is not int
                or response["candidateCount"] != len(payload["candidates"])
                or not isinstance(response.get("applied"), bool)):
            raise ValueError("BE 관심 분야 적재 응답이 요청과 일치하지 않습니다.")

    def _safe_error(self, stage: str, error: Exception) -> dict[str, str]:
        message = str(error)
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        return {"stage": stage, "type": type(error).__name__, "message": message}
