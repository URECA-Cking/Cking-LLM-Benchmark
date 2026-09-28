"""쿼리별 설정 11개의 상위 5명을 합집합(pooling)해 블라인드 판정 시트를 만든다."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeRow:
    """판정자가 볼 한 줄이다. 어떤 설정이 뽑았는지는 별도 매핑 파일에만 남는다."""

    pair_id: str
    query_id: str
    candidate_id: str
    score: str = ""


def build_judge_pairs(top5_by_query_and_method: dict[str, dict[str, list[tuple[str, float]]]]) -> tuple[list[JudgeRow], dict[str, list[str]]]:
    """{query_id: {method_id: [(candidate_id, score), ...]}}에서 쿼리·후보 쌍을 합집합으로 모은다.

    반환값은 (판정 행 목록, 쿼리별로 어떤 방식이 그 후보를 뽑았는지 기록한 매핑)이다.
    매핑은 채점용 내부 자료이며 판정자에게는 보여주지 않는다.
    """
    pairs_seen: dict[tuple[str, str], list[str]] = {}
    for query_id, by_method in top5_by_query_and_method.items():
        for method_id, ranked in by_method.items():
            for candidate_id, _score in ranked:
                pairs_seen.setdefault((query_id, candidate_id), []).append(method_id)

    rows = [
        JudgeRow(pair_id=f"{query_id}::{candidate_id}", query_id=query_id, candidate_id=candidate_id)
        for (query_id, candidate_id) in pairs_seen
    ]
    provenance = {row.pair_id: sorted(set(pairs_seen[(row.query_id, row.candidate_id)])) for row in rows}
    return rows, provenance


def shuffle_rows(rows: list[JudgeRow], seed: int) -> list[JudgeRow]:
    """판정 순서에서 방식·쿼리 패턴을 알아채지 못하게 무작위로 섞는다."""
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled


def load_existing_scores(path: Path) -> dict[str, str]:
    """이미 있는 판정 시트에서 pair_id별 score를 읽는다. 파일이 없으면 빈 dict를 반환한다."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["pair_id"]: row["score"] for row in csv.DictReader(f) if row["score"].strip()}


def write_judge_sheet(rows: list[JudgeRow], path: Path) -> None:
    """판정자가 0/1/2점을 적어 넣을 CSV를 만든다.

    같은 경로에 이미 채워진 판정이 있으면 pair_id가 같은 행은 그 점수를 그대로 이어받는다
    (리뷰 P1: 재실행 시 기존 사람 판정·유료 자동 판정이 사라지던 문제 수정). 새로 생긴
    pair_id만 빈 칸으로 남는다.
    """
    existing = load_existing_scores(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "query_id", "candidate_id", "score"])
        for row in rows:
            score = existing.get(row.pair_id, row.score)
            writer.writerow([row.pair_id, row.query_id, row.candidate_id, score])


def write_provenance(provenance: dict[str, list[str]], path: Path) -> None:
    """pair_id별로 어떤 방식이 후보를 뽑았는지 기록한다. 채점 전용, 판정자에게 공개하지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "methods"])
        for pair_id, methods in provenance.items():
            writer.writerow([pair_id, ";".join(methods)])
