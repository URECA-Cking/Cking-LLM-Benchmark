"""같은 manifest의 M2/M3 Top-N 합집합을 방법 정보 없는 Judge 입력으로 내보낸다."""

from __future__ import annotations

import json
from pathlib import Path

from src.recommendation.batch import _json_hash, _write_json_atomic
from src.recommendation.cache import text_hash
from src.recommendation.interest import M2_METHOD, M3_METHOD, InterestRecommender, _integer


def write_interest_evaluation(
    output_dir: Path, recommender: InterestRecommender,
    bundles: dict[str, dict[str, dict[str, object]]], *, top_n: int = 10,
) -> dict[str, object]:
    """입력 지문 변경을 해시로 추적하고 판정 자료와 순위·방식 출처를 분리한다.

    모델/Judge 호출이나 평가 점수 작성은 하지 않는다. 동일 파일은 그대로 보존하고
    다른 계약의 파일이 있으면 어느 파일도 덮어쓰기 전에 실패한다.
    """
    _integer("evaluation top_n", top_n, 1, recommender.config.top_n)
    if set(bundles) != {row.code for row in recommender.categories}:
        raise ValueError("평가 산출물에는 17개 분야의 완성된 M2/M3 세대가 필요합니다.")
    profiles = {profile.creator_id: profile for profile in recommender.manifest.creators}
    pairs: dict[str, dict[str, object]] = {}
    sources: dict[str, list[dict[str, object]]] = {}
    rankings = {}
    for category in recommender.categories:
        rankings[category.code] = {}
        for method in (M2_METHOD, M3_METHOD):
            payload = bundles[category.code][method]
            selected = payload["candidates"][:top_n]
            rankings[category.code][method] = {
                "inputHash": payload["inputHash"], "requestedCount": top_n,
                "candidateCount": len(selected), "shortageCount": top_n - len(selected),
                "candidates": selected,
            }
            for candidate in selected:
                profile = profiles[candidate["creatorId"]]
                identity = {
                    "taxonomyVersion": recommender.config.taxonomy_version,
                    "taxonomyHash": recommender.config.taxonomy_hash,
                    "interestCode": category.code, "queryTextHash": text_hash(category.description),
                    "creatorId": profile.creator_id, "candidateTextHash": text_hash(profile.text),
                }
                pair_id = _json_hash(identity)
                pairs[pair_id] = {
                    "pairId": pair_id, "interestCode": category.code,
                    "interestName": category.name, "interestDescription": category.description,
                    "creatorId": profile.creator_id, "introduction": profile.text,
                    "queryTextHash": identity["queryTextHash"],
                    "candidateTextHash": identity["candidateTextHash"],
                }
                sources.setdefault(pair_id, []).append({
                    "method": method, "rank": candidate["rank"],
                    "score": candidate["score"], "inputHash": payload["inputHash"],
                })
    # 방법별 순위로 나열하지 않아 행 순서로도 방법 정보가 노출되지 않는다.
    judge_input = {"schemaVersion": 1, "pairs": [pairs[key] for key in sorted(pairs)]}
    plan = {
        "schemaVersion": 1, "manifestHash": recommender.manifest.manifest_hash,
        "generationConfig": recommender.config.identity(), "evaluationTopN": top_n,
        "methods": [M2_METHOD, M3_METHOD], "pairCount": len(pairs),
        "judgeInputHash": _json_hash(judge_input), "rankings": rankings,
        "pairSources": {key: sources[key] for key in sorted(sources)},
        "limitations": "LLM Judge 비교 입력이며 사람 품질·개인화·서비스 채택은 검증하지 않음",
    }
    artifacts = {"judge-input.json": judge_input, "provenance.json": plan}
    for filename, payload in artifacts.items():
        path = output_dir / filename
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"기존 평가 산출물 계약이 다릅니다. 새 출력 디렉터리를 사용하세요: {filename}")
    for filename, payload in artifacts.items():
        path = output_dir / filename
        if not path.exists():
            _write_json_atomic(path, payload)
    return {"status": "success", "topN": top_n, "pairCount": len(pairs),
            "judgeInputHash": plan["judgeInputHash"]}
