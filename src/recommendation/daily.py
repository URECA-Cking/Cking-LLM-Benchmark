"""현재 manifest를 감지하고 두 추천 배치의 완전 적용 상태를 관리한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from src.recommendation.batch import _json_hash, _write_json_atomic
from src.recommendation.http import JsonBackend
from src.recommendation.application import allocate_sequence, validate_sequence
from src.recommendation.locking import batch_locks
from src.recommendation.manifest import CreatorManifest, fetch_manifest, write_manifest


class RunnableBatch(Protocol):
    def run(self, mode: str) -> dict[str, object]: ...


BatchFactory = Callable[[CreatorManifest, Path], RunnableBatch]


@dataclass(frozen=True)
class DailyPaths:
    output_dir: Path

    @property
    def completed(self) -> Path:
        return self.output_dir / "last-applied.json"

    @property
    def active(self) -> Path:
        return self.output_dir / "active-run.json"

    @property
    def summary(self) -> Path:
        return self.output_dir / "summary.json"

    def run_dir(self, manifest_hash: str, config_hash: str) -> Path:
        return self.output_dir / "runs" / manifest_hash / config_hash


class DailyRecommendationBatch:
    """마지막 완전 적용 hash와 대상·설정이 같을 때만 모델·쓰기를 건너뛴다.

    새 입력은 새 디렉터리로 전환한다. 미완료 실행의 산출물은 보존하되 현재 입력만
    처리한다. 내용 산출물은 재사용하고, 새 적용 의도마다 영속 증가 번호로 전부
    다시 적용한다. 같은 실행의 중단·재개에는 번호와 payload를 유지한다.
    """

    def __init__(self, backend: JsonBackend, paths: DailyPaths, cache_path: Path,
                 generation_identity: dict[str, object], similar_factory: BatchFactory,
                 interest_factory: BatchFactory, *, page_size: int = 100) -> None:
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("page_size는 1~100의 정수여야 합니다.")
        if cache_path.resolve().is_relative_to(paths.output_dir.resolve()):
            raise ValueError("공유 모델 캐시는 일일 배치 출력 디렉터리 밖에 두어야 합니다.")
        self.backend = backend
        self.paths = paths
        self.cache_path = cache_path
        self.generation_identity = generation_identity
        self.config_hash = _json_hash(generation_identity)
        self.similar_factory = similar_factory
        self.interest_factory = interest_factory
        self.page_size = page_size

    def run(self, mode: str = "apply") -> dict[str, object]:
        if mode not in {"apply", "dry-run"}:
            raise ValueError("mode는 apply 또는 dry-run이어야 합니다.")
        with batch_locks(self.paths.output_dir, self.cache_path):
            return self._run_locked(mode)

    @staticmethod
    def _read_state(path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or payload.get("schemaVersion") not in {1, 2}
                or any(not isinstance(payload.get(key), str) for key in
                       ("manifestHash", "generationConfigHash", "applyTarget"))):
            raise ValueError("일일 배치 상태 파일이 올바르지 않습니다.")
        if payload["schemaVersion"] == 2:
            validate_sequence(payload.get("applicationSequence"))
        return payload

    def _run_locked(self, mode: str) -> dict[str, object]:
        # 매 실행 공개 목록을 끝까지 읽는다. hash에는 소개·ID만 들어간다.
        manifest = fetch_manifest(self.backend, page_size=self.page_size)
        write_manifest(self.paths.output_dir / "current-manifest.json", manifest)
        identity = {"schemaVersion": 2, "manifestHash": manifest.manifest_hash,
                    "generationConfigHash": self.config_hash, "applyTarget": self.backend.target_identity}
        active = self._read_state(self.paths.active)
        completed = self._read_state(self.paths.completed)
        ledger = self.cache_path.with_suffix(".applications.json")
        if mode == "apply" and not ledger.exists() and any(
                state and state.get("schemaVersion") == 2 for state in (active, completed)):
            raise ValueError("적용 ledger가 유실되었습니다. BE 최대 번호를 확인하고 복구하세요.")
        if active is None and completed is not None and all(completed.get(k) == v for k, v in identity.items()):
            summary = {**identity, "mode": mode, "status": "skipped", "failureCount": 0}
            _write_json_atomic(self.paths.summary, summary)
            return summary

        run_dir = self.paths.run_dir(manifest.manifest_hash, self.config_hash)
        similar_dir, interest_dir = run_dir / "similar", run_dir / "interests"
        write_manifest(run_dir / "creator-manifest.json", manifest)
        _write_json_atomic(run_dir / "generation-config.json", self.generation_identity)
        superseded = None
        if mode == "apply":
            if active is None or any(active.get(key) != value for key, value in identity.items()):
                superseded = active
                # resetRequired는 초기화 도중 중단되어도 다음 실행이 초기화를 마치게 한다.
                sequence = allocate_sequence(self.cache_path.with_suffix(".applications.json"), self.backend.target_identity)
                active = {**identity, "applicationSequence": sequence, "resetRequired": True}
                _write_json_atomic(self.paths.active, active)
            if active.get("resetRequired") is not False:
                for directory, records_key in ((similar_dir, "creators"), (interest_dir, "interests")):
                    checkpoint_path = directory / "checkpoint.json"
                    if checkpoint_path.exists():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        checkpoint["applyTarget"] = None
                        for record in checkpoint[records_key].values():
                            record["applyStatus"] = "pending"
                            if record.get("error", {}).get("stage") == "apply":
                                record.pop("error", None)
                        _write_json_atomic(checkpoint_path, checkpoint)
                active = {**active, "resetRequired": False}
                _write_json_atomic(self.paths.active, active)

        stages = {}
        for name, factory, directory in (("similar", self.similar_factory, similar_dir),
                                         ("interests", self.interest_factory, interest_dir)):
            try:
                batch = factory(manifest, directory)
                if mode == "apply":
                    batch.application_sequence = validate_sequence(active["applicationSequence"])
                stage = batch.run(mode)
                expected = len(manifest.creators) if name == "similar" else 17
                generated = stage.get("nonEmptyGenerationCount", 0) + stage.get("emptyGenerationCount", 0)
                applied = stage.get("appliedSuccessCount", 0) + stage.get("idempotentCount", 0)
                complete = (stage.get("failureCount") == 0 and generated == expected
                            and (mode == "dry-run" or applied == expected))
                stages[name] = {"status": "success" if complete else "failed",
                                "failureCount": stage.get("failureCount", 0),
                                "generatedCount": generated, "appliedCount": applied}
            except Exception as error:
                # 프로필·키를 포함할 수 있는 외부 예외 원문은 통합 상태에 저장하지 않는다.
                stages[name] = {"status": "failed", "failureCount": 1, "errorType": type(error).__name__}
        failed = sum(stage["status"] != "success" for stage in stages.values())
        summary = {**identity, "mode": mode,
                   "status": "partial" if failed else ("completed" if mode == "apply" else "generated"),
                   "failureCount": failed, "stages": stages,
                   "runDirectory": str(run_dir.relative_to(self.paths.output_dir)).replace("\\", "/"),
                   "supersededRun": superseded,
                   "applicationSequence": active.get("applicationSequence") if active else None}
        _write_json_atomic(self.paths.summary, summary)
        if not failed and mode == "apply":
            # 두 하위 배치가 모두 완전 적용된 뒤에만 마지막 완료 상태를 원자적으로 교체한다.
            _write_json_atomic(self.paths.completed, {**identity, "applicationSequence": active["applicationSequence"]})
            self.paths.active.unlink(missing_ok=True)
        return summary
