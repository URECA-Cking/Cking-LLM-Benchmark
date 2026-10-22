"""Python·Java가 공유하는 분류체계 canonical JSON과 SHA-256 계약이다."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import DATA_DIR


DEFAULT_CATEGORIES_CSV = DATA_DIR / "categories_v2.csv"
DEFAULT_TAXONOMY_VERSION = "v0.2"
DEFAULT_TAG_PROMPT_VERSION = "creator-category-v2"


@dataclass(frozen=True)
class ServiceCategory:
    code: str
    name: str
    description: str


def normalize_taxonomy_field(value: str) -> str:
    """줄바꿈 → ASCII 양끝 공백 제거 → NFC 순서를 지킨다."""
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n").strip(" \t\n"))


def _normalized_categories(categories: Iterable[ServiceCategory]) -> list[ServiceCategory]:
    normalized = [
        ServiceCategory(*(normalize_taxonomy_field(value) for value in (row.code, row.name, row.description)))
        for row in categories
    ]
    if not normalized or any(not row.code or not row.name or not row.description for row in normalized):
        raise ValueError("분야 CSV의 code, name, description은 모두 필요합니다.")
    codes = [row.code for row in normalized]
    if len(codes) != len(set(codes)):
        raise ValueError("분야 CSV에 중복 code가 있습니다.")
    return normalized


def load_service_categories(path: Path) -> list[ServiceCategory]:
    """BOM은 CSV 입력에서만 허용하며 정규화 후 빈 필드와 코드 중복을 거부한다."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["code", "name", "description"]:
            raise ValueError("분야 CSV 헤더는 code,name,description이어야 합니다.")
        categories = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("분야 CSV의 각 행에는 code, name, description이 필요합니다.")
            categories.append(ServiceCategory(row["code"], row["name"], row["description"]))
    return _normalized_categories(categories)


def canonical_taxonomy_json(categories: Iterable[ServiceCategory]) -> str:
    """행 순서와 categories/code/name/description 키 순서를 고정한다."""
    payload = {"categories": [
        {"code": row.code, "name": row.name, "description": row.description}
        for row in _normalized_categories(categories)
    ]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def taxonomy_hash(categories: Iterable[ServiceCategory]) -> str:
    """BOM·마지막 개행 없는 canonical JSON UTF-8 바이트를 해시한다."""
    return hashlib.sha256(canonical_taxonomy_json(categories).encode("utf-8")).hexdigest()


def default_taxonomy_hash() -> str:
    return taxonomy_hash(load_service_categories(DEFAULT_CATEGORIES_CSV))
