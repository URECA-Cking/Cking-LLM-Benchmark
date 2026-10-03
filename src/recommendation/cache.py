"""모델 호출 결과를 입력·모델·프롬프트 버전에 묶어 저장한다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol


def text_hash(text: str) -> str:
    """정규화가 끝난 모델 입력의 전체 SHA-256을 반환한다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_key(kind: str, input_hash: str, model_version: str, prompt_version: str | None) -> str:
    identity = {
        "kind": kind,
        "inputHash": input_hash,
        "modelVersion": model_version,
        "promptVersion": prompt_version,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ModelCache(Protocol):
    """추천 생성기가 의존하는 모델 캐시 인터페이스다."""

    def get_embedding(self, input_hash: str, model_version: str) -> list[float] | None:
        ...

    def put_embedding(self, input_hash: str, model_version: str, vector: list[float]) -> None:
        ...

    def put_embeddings(self, records: list[tuple[str, str, list[float]]]) -> None:
        ...

    def get_tags(self, input_hash: str, model_version: str, prompt_version: str) -> tuple[str, ...] | None:
        ...

    def put_tags(self, input_hash: str, model_version: str, prompt_version: str, tags: tuple[str, ...]) -> None:
        ...

    def put_tag_records(self, records: list[tuple[str, str, str, tuple[str, ...]]]) -> None:
        ...


class JsonModelCache:
    """프로세스가 다시 시작돼도 재사용하는 원자적 JSON 모델 캐시다."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()

    def _empty(self) -> dict[str, object]:
        return {"schemaVersion": self.SCHEMA_VERSION, "embeddings": {}, "tags": {}}

    def _load(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schemaVersion") != self.SCHEMA_VERSION:
            return self._empty()
        if not isinstance(data.get("embeddings"), dict) or not isinstance(data.get("tags"), dict):
            raise ValueError(f"모델 캐시 형식이 올바르지 않습니다: {self.path}")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(self._data, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp, self.path)

    def get_embedding(self, input_hash: str, model_version: str) -> list[float] | None:
        key = _record_key("embedding", input_hash, model_version, None)
        record = self._data["embeddings"].get(key)  # type: ignore[index]
        if record is None:
            return None
        return [float(value) for value in record["vector"]]

    def put_embedding(self, input_hash: str, model_version: str, vector: list[float]) -> None:
        self.put_embeddings([(input_hash, model_version, vector)])

    def put_embeddings(self, records: list[tuple[str, str, list[float]]]) -> None:
        for input_hash, model_version, vector in records:
            key = _record_key("embedding", input_hash, model_version, None)
            self._data["embeddings"][key] = {  # type: ignore[index]
                "inputHash": input_hash,
                "modelVersion": model_version,
                "vector": vector,
            }
        if records:
            self._save()

    def get_tags(self, input_hash: str, model_version: str, prompt_version: str) -> tuple[str, ...] | None:
        key = _record_key("tags", input_hash, model_version, prompt_version)
        record = self._data["tags"].get(key)  # type: ignore[index]
        if record is None:
            return None
        return tuple(str(tag) for tag in record["tags"])

    def put_tags(self, input_hash: str, model_version: str, prompt_version: str, tags: tuple[str, ...]) -> None:
        self.put_tag_records([(input_hash, model_version, prompt_version, tags)])

    def put_tag_records(self, records: list[tuple[str, str, str, tuple[str, ...]]]) -> None:
        for input_hash, model_version, prompt_version, tags in records:
            key = _record_key("tags", input_hash, model_version, prompt_version)
            self._data["tags"][key] = {  # type: ignore[index]
                "inputHash": input_hash,
                "modelVersion": model_version,
                "promptVersion": prompt_version,
                "tags": list(tags),
            }
        if records:
            self._save()
