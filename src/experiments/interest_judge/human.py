"""Judge 판정과 분리한 빈 블라인드 사람 시트와 독립 판정 집계."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from src.recommendation.batch import _json_hash, _write_json_atomic
from src.recommendation.interest import METHODS, M2_METHOD, M3_METHOD
from .contract import read_json, validate_judgment
from .scoring import aggregate


BLIND_FIELDS = ("sampleId", "interestCode", "interestName", "interestDescription", "candidateId", "introduction")
LABEL_FIELDS = ("relevance", "isNoise", "noiseType", "reason")
FIELDS = BLIND_FIELDS + LABEL_FIELDS


def _blind_row(pair):
    return {"sampleId": pair["pairId"], **{key: pair[key] for key in BLIND_FIELDS if key != "sampleId"}}


def _csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_csv(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_human_sheet(contract, scores, work_dir: Path, *, sample_size: int = 50):
    if type(sample_size) is not int or sample_size < 50:
        raise ValueError("사람 시트는 최소 50쌍이어야 합니다.")
    if len(scores) != len(contract["pairs"]):
        raise ValueError("Judge 판정이 완료되어야 불일치·개선·악화 사례를 층화할 수 있습니다.")
    if len(contract["pairs"]) < sample_size:
        raise ValueError("유일한 평가 쌍이 부족해 최소 사람 표본을 만들 수 없습니다.")
    report = aggregate(contract, scores)
    groups = {}
    strata = {}
    for pair in contract["pairs"]:
        code, cid, pid = pair["interestCode"], pair["creatorId"], pair["pairId"]
        memberships = [method for method in METHODS if cid in contract["rankings"][code][method]]
        tags = ["interest:" + code, "outcome:" + report["perInterest"][code]["outcome"]]
        if scores[pid]["isNoise"]:
            tags.append("noise")
        top5_memberships = [method for method in METHODS if cid in contract["rankings"][code][method][:5]]
        if len(top5_memberships) == 1:
            # 동일 쌍의 판정은 공유하므로 서로 다른 단독 후보의 판정 차이를 불일치 사례로 삼는다.
            other_method = M3_METHOD if top5_memberships[0] == M2_METHOD else M2_METHOD
            other_ids = set(contract["rankings"][code][other_method][:5]) - set(contract["rankings"][code][top5_memberships[0]][:5])
            other_scores = [scores[p["pairId"]]["relevance"] for p in contract["pairs"]
                            if p["interestCode"] == code and p["creatorId"] in other_ids]
            if any(value != scores[pid]["relevance"] for value in other_scores):
                tags.append("judgeDisagreement:" + top5_memberships[0])
        for method in memberships:
            tags.append("methodCoverage:" + method)
        strata[pid] = tags
        for tag in tags:
            groups.setdefault(tag, []).append(pair)
    selected = {}
    # 분야별 한 쌍을 먼저 보장하고 나머지는 잡음·차이·승패 그룹을 순환한다.
    for interest in contract["interests"]:
        group = groups.get("interest:" + interest["code"], [])
        if group:
            selected[group[0]["pairId"]] = group[0]
    order = sorted(groups, key=lambda tag: (not tag.startswith("judgeDisagreement:"), tag != "noise",
                                           not tag.startswith("outcome:"), tag))
    while len(selected) < sample_size:
        before = len(selected)
        for tag in order:
            if len(selected) >= sample_size:
                break
            for pair in groups[tag]:
                if pair["pairId"] not in selected:
                    selected[pair["pairId"]] = pair
                    break
        if len(selected) == before:
            raise ValueError("사람 표본의 유일한 쌍이 부족합니다.")
    # 층화 정보나 Judge 점수는 사람에게 보이는 CSV에 포함하지 않는다.
    rows = [{**_blind_row(pair), **dict.fromkeys(LABEL_FIELDS, "")}
            for pair in contract["pairs"] if pair["pairId"] in selected]
    sampling = {"schemaVersion": 1, "contractHash": contract["contractHash"], "sampleSize": sample_size,
                "samples": {row["sampleId"]: {"blindRowHash": _json_hash({key: row[key] for key in BLIND_FIELDS}),
                                               "strata": strata[row["sampleId"]]} for row in rows},
                "strataAvailableCounts": {tag: len(group) for tag, group in groups.items()},
                "strataSelectedCounts": {tag: sum(tag in strata[pid] for pid in selected) for tag in groups}}
    path, metadata_path = work_dir / "human-blank.csv", work_dir / "human-sample.json"
    text = _csv_text(rows)
    # 사람이 작성한 파일이나 다른 계약의 표본을 덮어쓰지 않는다.
    if path.exists() and path.read_bytes() != text.encode("utf-8"):
        raise ValueError("기존 사람 시트가 다르거나 작성되었습니다. 별도 filled 파일을 사용하세요.")
    if metadata_path.exists() and read_json(metadata_path) != sampling:
        raise ValueError("기존 사람 표본 계약이 다릅니다.")
    if not path.exists():
        _write_csv(path, text)
    if not metadata_path.exists():
        _write_json_atomic(metadata_path, sampling)
    return sampling


def score_human(contract, judge_scores, work_dir: Path, filled_path: Path):
    if filled_path.resolve() == (work_dir / "human-blank.csv").resolve():
        raise ValueError("빈 시트를 복사한 별도 filled 파일을 입력하세요.")
    sampling = read_json(work_dir / "human-sample.json")
    if sampling["contractHash"] != contract["contractHash"]:
        raise ValueError("사람 표본과 평가 계약이 다릅니다.")
    by_id = {pair["pairId"]: pair for pair in contract["pairs"]}
    expected = sampling["samples"]
    if (sampling.get("sampleSize") != len(expected) or len(expected) < 50
            or not set(expected) <= set(by_id)
            or any(record["blindRowHash"] != _json_hash(_blind_row(by_id[pid])) for pid, record in expected.items())):
        raise ValueError("사람 표본 지문이나 최소 50쌍 계약이 다릅니다.")
    human_scores, seen = {}, set()
    with filled_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("사람 시트 헤더가 다릅니다.")
        for row in reader:
            pid = row.get("sampleId")
            if (pid not in expected or pid in seen or set(row) != set(FIELDS)
                    or _json_hash({key: row[key] for key in BLIND_FIELDS}) != expected[pid]["blindRowHash"]):
                raise ValueError("사람 시트에 중복·알 수 없는 ID·변경된 원문이 있습니다.")
            seen.add(pid)
            labels = [row[key] for key in LABEL_FIELDS]
            if all(value == "" for value in labels):
                continue
            if row["relevance"] not in ("0", "1", "2") or row["isNoise"] not in ("0", "1"):
                raise ValueError("사람 판정은 relevance=0/1/2, isNoise=0/1이어야 합니다.")
            score = {"candidateId": row["candidateId"], "relevance": int(row["relevance"]),
                     "isNoise": row["isNoise"] == "1", "noiseType": row["noiseType"],
                     "reason": row["reason"] or "사람 판정"}
            validate_judgment(score)
            human_scores[pid] = score
    if seen != set(expected):
        raise ValueError("사람 시트에 누락된 표본 행이 있습니다.")
    compared = set(human_scores) & set(judge_scores)
    methods = {}
    def fit_rates(reviewed):
        n = len(reviewed)
        return {"pairCount": n,
                "strictFitRate": sum(row["relevance"] == 2 for row in reviewed) / n if n else None,
                "binaryFitRate": sum(row["relevance"] >= 1 for row in reviewed) / n if n else None,
                "noiseRate": sum(row["isNoise"] for row in reviewed) / n if n else None}
    for method in METHODS:
        reviewed = [score for pid, score in human_scores.items()
                    if by_id[pid]["creatorId"] in contract["rankings"][by_id[pid]["interestCode"]][method]]
        methods[method] = {**fit_rates(reviewed), "byCutoff": {
            str(k): fit_rates([score for pid, score in human_scores.items()
                              if by_id[pid]["creatorId"] in contract["rankings"][by_id[pid]["interestCode"]][method][:k]])
            for k in (5, 10)}}
    def agreement(predicate):
        return sum(predicate(human_scores[pid]) == predicate(judge_scores[pid]) for pid in compared) / len(compared) if compared else None
    return {"complete": len(human_scores) == len(expected), "pairCount": len(human_scores),
            "requestedPairCount": len(expected), "comparedPairCount": len(compared),
            "threeLevelExactAgreement": agreement(lambda row: row["relevance"]),
            "binaryAgreement": agreement(lambda row: row["relevance"] >= 1),
            "methods": methods, "metricScope": "stratified reviewed subset; not population P@k",
            "fullRankingMetrics": aggregate(contract, human_scores) if len(human_scores) == len(contract["pairs"]) else None}
