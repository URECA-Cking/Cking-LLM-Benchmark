"""쿼리별 설정 11개의 상위 5명을 합집합(pooling)해 블라인드 판정 시트를 만든다."""

from __future__ import annotations

import csv
import hashlib
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
    text_hash: str = ""


def pair_text_hash(query_text: str, candidate_text: str) -> str:
    """쿼리·후보 텍스트의 해시를 만든다. 점수를 보존해도 되는지 판단하는 기준이다

    (PR #5 재리뷰로 발견: pair_id만 보고 점수를 보존하면, id는 같아도 크리에이터의
    input_text()가 바뀐 경우 예전 텍스트로 매긴 점수가 그대로 재사용될 수 있었다).
    """
    return hashlib.sha256(f"{query_text}\x1f{candidate_text}".encode("utf-8")).hexdigest()[:16]


def build_judge_pairs(
    top5_by_query_and_method: dict[str, dict[str, list[tuple[str, float]]]],
    text_by_id: dict[str, str] | None = None,
) -> tuple[list[JudgeRow], dict[str, list[str]]]:
    """{query_id: {method_id: [(candidate_id, score), ...]}}에서 쿼리·후보 쌍을 합집합으로 모은다.

    text_by_id를 주면 각 행에 쿼리·후보 텍스트 해시를 같이 계산해 붙인다(점수 보존 판단용).
    반환값은 (판정 행 목록, 쿼리별로 어떤 방식이 그 후보를 뽑았는지 기록한 매핑)이다.
    매핑은 채점용 내부 자료이며 판정자에게는 보여주지 않는다.
    """
    pairs_seen: dict[tuple[str, str], list[str]] = {}
    for query_id, by_method in top5_by_query_and_method.items():
        for method_id, ranked in by_method.items():
            for candidate_id, _score in ranked:
                pairs_seen.setdefault((query_id, candidate_id), []).append(method_id)

    rows = [
        JudgeRow(
            pair_id=f"{query_id}::{candidate_id}",
            query_id=query_id,
            candidate_id=candidate_id,
            text_hash=pair_text_hash(text_by_id[query_id], text_by_id[candidate_id]) if text_by_id else "",
        )
        for (query_id, candidate_id) in pairs_seen
    ]
    provenance = {row.pair_id: sorted(set(pairs_seen[(row.query_id, row.candidate_id)])) for row in rows}
    return rows, provenance


def shuffle_rows(rows: list[JudgeRow], seed: int) -> list[JudgeRow]:
    """판정 순서에서 방식·쿼리 패턴을 알아채지 못하게 무작위로 섞는다."""
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled


def load_existing_scores(path: Path) -> dict[str, tuple[str, str]]:
    """이미 있는 판정 시트에서 pair_id별 (score, text_hash)를 읽는다. 없으면 빈 dict를 반환한다.

    옛 형식(4열, text_hash 없음)의 행은 text_hash가 빈 문자열로 읽힌다 — 새 해시와는
    항상 다르다고 취급되어(둘 다 빈 문자열이 아닌 한) 재검증 대상이 된다.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {
            row["pair_id"]: (row["score"], row.get("text_hash", ""))
            for row in csv.DictReader(f)
            if row["score"].strip()
        }


def write_judge_sheet(rows: list[JudgeRow], path: Path) -> None:
    """판정자가 0/1/2점을 적어 넣을 CSV를 만든다.

    같은 경로에 이미 채워진 판정이 있고 pair_id·text_hash가 둘 다 같으면 그 점수를 그대로
    이어받는다 (리뷰 P1: 재실행 시 기존 판정이 사라지던 문제 수정. 재리뷰 지적: pair_id만
    보면 크리에이터 텍스트가 바뀌어도 예전 텍스트로 매긴 점수가 남을 수 있어, text_hash가
    다르면 점수를 비워 재판정하게 한다). text_hash를 계산하지 않은 행(빈 문자열)은 절대
    보존되지 않는다 — 텍스트 변경 여부를 모르면 안전하게 다시 판정한다.
    """
    existing = load_existing_scores(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        for row in rows:
            prev = existing.get(row.pair_id)
            keep = bool(row.text_hash) and prev is not None and prev[1] == row.text_hash
            score = prev[0] if keep else row.score
            writer.writerow([row.pair_id, row.query_id, row.candidate_id, score, row.text_hash])


def write_provenance(provenance: dict[str, list[str]], path: Path) -> None:
    """pair_id별로 어떤 방식이 후보를 뽑았는지 기록한다. 채점 전용, 판정자에게 공개하지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "methods"])
        for pair_id, methods in provenance.items():
            writer.writerow([pair_id, ";".join(methods)])
