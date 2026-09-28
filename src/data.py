"""카테고리·크리에이터·분할 데이터를 읽고 hash로 무결성을 검증한다."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from src.config import CATEGORIES_CSV, CREATORS_CSV, CREATORS_CSV_SHA256, SPLIT_CSV, SPLIT_CSV_SHA256


@dataclass(frozen=True)
class Category:
    """카테고리 코드와 zero-shot 태깅에 쓸 설명문이다."""

    code: str
    name: str
    description: str


@dataclass(frozen=True)
class Creator:
    """가상 크리에이터 한 명의 입력 텍스트와 정답·메타 정보다."""

    id: str
    name: str
    bio: str
    events: tuple[str, ...]
    subtopic: str
    gold: tuple[str, ...]
    declared: tuple[str, ...]
    written_by: str
    note: str
    split: str
    is_query: bool

    def input_text(self) -> str:
        """임베딩·태깅에 넣을 텍스트를 만든다. 소개가 없으면 이벤트 제목으로 대체한다."""
        if self.bio.strip():
            return self.bio.strip()
        return " / ".join(self.events)


def _sha256(path: Path) -> str:
    """파일 내용의 SHA-256 hex digest를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_data_hashes() -> None:
    """creators.csv·split.csv가 실험 착수 시 확정한 내용과 같은지 확인한다."""
    mismatches = []
    for path, expected in ((CREATORS_CSV, CREATORS_CSV_SHA256), (SPLIT_CSV, SPLIT_CSV_SHA256)):
        actual = _sha256(path)
        if actual != expected:
            mismatches.append(f"{path.name}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError(
            "데이터 파일이 실험 착수 시 확정본과 다릅니다. gold·declared·split 수정은 금지되어 있으니, "
            "의도한 변경이면 src/config.py의 hash와 README를 함께 갱신하세요.\n" + "\n".join(mismatches)
        )


def load_categories() -> list[Category]:
    """카테고리 10종을 code 순서 그대로 읽는다."""
    with CATEGORIES_CSV.open(encoding="utf-8") as f:
        return [Category(row["code"], row["name"], row["description"]) for row in csv.DictReader(f)]


def _split_codes(value: str | None) -> tuple[str, ...]:
    """`FITNESS|FOOD` 형태의 셀을 코드 튜플로 나눈다. 칸이 아예 없어 None이어도 빈 튜플로 본다."""
    if not value:
        return ()
    return tuple(code.strip() for code in value.split("|") if code.strip())


def load_creators() -> list[Creator]:
    """크리에이터 100명을 split.csv와 결합해 읽는다. 데이터 hash를 먼저 검증한다."""
    verify_data_hashes()
    with SPLIT_CSV.open(encoding="utf-8") as f:
        split_by_id = {row["id"]: row for row in csv.DictReader(f)}

    creators: list[Creator] = []
    with CREATORS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_row = split_by_id.get(row["id"])
            if split_row is None:
                raise ValueError(f"split.csv에 {row['id']}가 없습니다.")
            gold = _split_codes(row["gold"])
            declared = _split_codes(row["declared"]) or gold
            events = tuple(e.strip() for e in row["events"].split("/") if e.strip())
            creators.append(
                Creator(
                    id=row["id"],
                    name=row["name"],
                    bio=row["bio"],
                    events=events,
                    subtopic=row["subtopic"],
                    gold=gold,
                    declared=declared,
                    written_by=row["written_by"],
                    note=row["note"],
                    split=split_row["split"],
                    is_query=split_row["query"].strip().upper() == "Y",
                )
            )
    if len(creators) != 100:
        raise ValueError(f"크리에이터는 100명이어야 하는데 {len(creators)}명입니다.")
    return creators


def dev_creators(creators: list[Creator]) -> list[Creator]:
    """dev 분할만 반환한다. τ·bonus·모델 선택에만 사용한다."""
    return [c for c in creators if c.split == "dev"]


def test_creators(creators: list[Creator]) -> list[Creator]:
    """test 분할만 반환한다. 성적표 계산에 사용한다."""
    return [c for c in creators if c.split == "test"]


def query_creators(creators: list[Creator]) -> list[Creator]:
    """평가 쿼리로 지정된 test 크리에이터만 반환한다."""
    return [c for c in creators if c.is_query]
