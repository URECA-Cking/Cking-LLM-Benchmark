"""Cking-BE의 전체 크리에이터 입력을 결정적 manifest로 고정한다."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.recommendation.http import JsonBackend
from src.recommendation.models import CreatorProfile


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CreatorManifest:
    creators: tuple[CreatorProfile, ...]
    manifest_hash: str
    source_path: str = "/api/creators"
    page_size: int = 100

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": MANIFEST_SCHEMA_VERSION,
            "sourcePath": self.source_path,
            "pageSize": self.page_size,
            "sort": "creatorId ASC",
            "creatorCount": len(self.creators),
            "manifestHash": self.manifest_hash,
            "creators": [
                {"creatorId": profile.creator_id, "introText": profile.text}
                for profile in self.creators
            ],
        }


def _canonical_payload(creators: tuple[CreatorProfile, ...]) -> bytes:
    payload = {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "creators": [
            {"creatorId": profile.creator_id, "introText": profile.text}
            for profile in creators
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(creators: tuple[CreatorProfile, ...]) -> str:
    return hashlib.sha256(_canonical_payload(creators)).hexdigest()


def build_manifest(profiles: list[CreatorProfile], *, page_size: int = 100) -> CreatorManifest:
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
        raise ValueError("manifest page_size는 1~100의 정수여야 합니다.")
    by_id: dict[int, CreatorProfile] = {}
    for profile in profiles:
        if profile.creator_id in by_id:
            raise ValueError(f"Creator 목록에 중복 creatorId가 있습니다: {profile.creator_id}")
        by_id[profile.creator_id] = CreatorProfile(profile.creator_id, profile.text)
    creators = tuple(by_id[creator_id] for creator_id in sorted(by_id))
    return CreatorManifest(creators, manifest_hash(creators), page_size=page_size)


def fetch_manifest(backend: JsonBackend, *, page_size: int = 100) -> CreatorManifest:
    """hasNext가 false가 될 때까지 공개 Creator 목록을 검증하며 읽는다."""
    profiles: list[CreatorProfile] = []
    expected_total: int | None = None
    expected_pages: int | None = None
    page = 0
    while True:
        response = backend.get(f"/api/creators?page={page}&size={page_size}")
        items = response.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Creator 목록 page={page}의 items는 배열이어야 합니다.")
        if response.get("page") != page:
            raise ValueError(f"Creator 목록 응답 page가 요청과 다릅니다: requested={page}")
        response_size = response.get("size")
        if isinstance(response_size, bool) or not isinstance(response_size, int) or not 1 <= response_size <= page_size:
            raise ValueError(f"Creator 목록 page={page}의 size가 올바르지 않습니다.")

        total = response.get("totalElements")
        total_pages = response.get("totalPages")
        has_next = response.get("hasNext")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Creator 목록 totalElements는 0 이상의 정수여야 합니다.")
        if isinstance(total_pages, bool) or not isinstance(total_pages, int) or total_pages < 0:
            raise ValueError("Creator 목록 totalPages는 0 이상의 정수여야 합니다.")
        if not isinstance(has_next, bool):
            raise ValueError("Creator 목록 hasNext는 boolean이어야 합니다.")
        if expected_total is None:
            expected_total, expected_pages = total, total_pages
        elif total != expected_total or total_pages != expected_pages:
            raise ValueError("페이지 순회 중 Creator 목록 건수 메타데이터가 바뀌었습니다.")
        if has_next and not items:
            raise ValueError("hasNext=true인 Creator 목록 페이지가 비어 있습니다.")

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Creator 목록 page={page} 항목은 객체여야 합니다.")
            try:
                profiles.append(CreatorProfile(item["creatorId"], item["introText"]))
            except KeyError as error:
                raise ValueError(f"Creator 목록 항목 필드가 없습니다: {error.args[0]}") from error
        if not has_next:
            break
        page += 1
        if expected_pages is not None and page >= expected_pages:
            raise ValueError("Creator 목록 hasNext와 totalPages가 일치하지 않습니다.")

    if expected_total != len(profiles):
        raise ValueError(f"Creator 목록 건수가 다릅니다: expected={expected_total}, actual={len(profiles)}")
    if expected_pages is not None and expected_pages != (0 if expected_total == 0 else page + 1):
        raise ValueError("Creator 목록 실제 페이지 수와 totalPages가 다릅니다.")
    return build_manifest(profiles, page_size=page_size)


def write_manifest(path: Path, manifest: CreatorManifest) -> None:
    _write_json_atomic(path, manifest.to_dict())


def load_manifest(path: Path) -> CreatorManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("지원하지 않는 Creator manifest 형식입니다.")
    raw_creators = payload.get("creators")
    if not isinstance(raw_creators, list):
        raise ValueError("Creator manifest의 creators는 배열이어야 합니다.")
    try:
        profiles = [CreatorProfile(item["creatorId"], item["introText"]) for item in raw_creators]
    except (KeyError, TypeError) as error:
        raise ValueError("Creator manifest 항목에는 creatorId와 introText가 필요합니다.") from error
    page_size = payload.get("pageSize")
    manifest = build_manifest(profiles, page_size=page_size)  # type: ignore[arg-type]
    if profiles != list(manifest.creators):
        raise ValueError("Creator manifest는 creatorId 오름차순이어야 합니다.")
    if payload.get("creatorCount") != len(manifest.creators):
        raise ValueError("Creator manifest creatorCount가 실제 건수와 다릅니다.")
    if payload.get("manifestHash") != manifest.manifest_hash:
        raise ValueError("Creator manifest SHA-256이 내용과 다릅니다.")
    return manifest


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
