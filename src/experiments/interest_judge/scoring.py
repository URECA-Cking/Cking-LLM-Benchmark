"""17개 분야 단위의 지표와 paired bootstrap. 부분 완료를 전체 결과로 쓰지 않는다."""

from __future__ import annotations

import math

import numpy as np

from src.recommendation.interest import METHODS, M2_METHOD, M3_METHOD
from src.recommendation.batch import _json_hash


def metrics(rows, ideal_rows, k):
    selected = rows[:k]
    def dcg(values):
        return sum((2 ** row["relevance"] - 1) / math.log2(rank + 2) for rank, row in enumerate(values))
    ideal = sorted(ideal_rows, key=lambda row: -row["relevance"])[:k]
    denominator = dcg(ideal)
    return {f"strictPrecision@{k}": sum(row["relevance"] == 2 for row in selected) / k,
            f"lenientPrecision@{k}": sum(row["relevance"] >= 1 for row in selected) / k,
            f"gradedNdcg@{k}": dcg(selected) / denominator if denominator else 0.,
            f"noiseRate@{k}": sum(row["isNoise"] for row in selected) / k}


def aggregate(contract, scores):
    by_pair = {(pair["interestCode"], pair["creatorId"]): pair for pair in contract["pairs"]}
    per_interest, noise_cases = {}, []
    completed = 0
    for interest in contract["interests"]:
        code = interest["code"]
        pairs = [pair for pair in contract["pairs"] if pair["interestCode"] == code]
        pending = sum(pair["pairId"] not in scores for pair in pairs)
        if pending:
            per_interest[code] = {"complete": False, "pendingPairCount": pending}
            continue
        completed += 1
        ideal = [scores[pair["pairId"]] for pair in pairs]
        results = {}
        for method in METHODS:
            rows = [scores[by_pair[code, cid]["pairId"]] for cid in contract["rankings"][code][method]]
            results[method] = {**metrics(rows, ideal, 5), **metrics(rows, ideal, 10),
                               "candidateCount": len(rows), "shortageCount": 10 - len(rows)}
        delta = {key: results[M3_METHOD][key] - results[M2_METHOD][key]
                 for key in results[M2_METHOD] if "@" in key}
        primary = delta["strictPrecision@5"]
        per_interest[code] = {"complete": True, "methods": results, "deltaM3MinusM2": delta,
                              "outcome": "win" if primary > 1e-12 else "loss" if primary < -1e-12 else "tie"}
        for pair in pairs:
            score = scores[pair["pairId"]]
            if score["isNoise"]:
                noise_cases.append({"interestCode": code, "pairId": pair["pairId"],
                                    "noiseType": score["noiseType"], "reason": score["reason"]})
    complete = completed == len(contract["interests"]) and len(scores) == len(contract["pairs"])
    result = {"complete": complete, "completedInterestCount": completed, "perInterest": per_interest,
              "noiseCases": noise_cases, "overall": None, "pairedBootstrap": None,
              "winLossTie": None, "cases": None}
    if not complete:
        return result
    keys = [key for key in per_interest[contract["interests"][0]["code"]]["deltaM3MinusM2"]]
    result["overall"] = {
        "methods": {method: {key: float(np.mean([row["methods"][method][key] for row in per_interest.values()]))
                             for key in keys} for method in METHODS},
        "deltaM3MinusM2": {key: float(np.mean([row["deltaM3MinusM2"][key] for row in per_interest.values()]))
                           for key in keys},
    }
    rules = contract["rules"]
    draws = np.random.default_rng(rules["bootstrapSeed"]).integers(
        0, len(per_interest), size=(rules["bootstrapIterations"], len(per_interest)))
    intervals = {}
    for key in keys:
        differences = np.array([row["deltaM3MinusM2"][key] for row in per_interest.values()])
        means = differences[draws].mean(axis=1)
        low, high = np.quantile(means, [.025, .975])
        intervals[key] = {"mean": float(differences.mean()), "lower": float(low), "upper": float(high)}
    result["pairedBootstrap"] = {"unit": "interest", "unitCount": 17, "confidenceLevel": .95,
                                  "iterations": rules["bootstrapIterations"], "seed": rules["bootstrapSeed"],
                                  "intervals": intervals}
    result["winLossTie"] = {value: sum(row["outcome"] == value for row in per_interest.values())
                            for value in ("win", "loss", "tie")}
    # 사례 본문은 원문 없는 로컬 파일에도 개인정보가 있을 수 있어 공개 집계에서 제외한다.
    cases = []
    for code, row in per_interest.items():
        m2 = set(contract["rankings"][code][M2_METHOD][:5])
        m3 = set(contract["rankings"][code][M3_METHOD][:5])
        for method, ids in ((M2_METHOD, m2 - m3), (M3_METHOD, m3 - m2)):
            for cid in sorted(ids):
                pair = by_pair[code, cid]
                score = scores[pair["pairId"]]
                cases.append({"interestCode": code, "outcome": row["outcome"], "method": method,
                              "pairId": pair["pairId"], **score})
    result["cases"] = cases
    return result


def choose_decision(contract, judge_report, human_report):
    generation = contract["generationConfig"]
    result = {"conclusion": "M2 유지", "reason": "Judge 또는 최소 50쌍 사람 검증이 미완료",
              "method": M2_METHOD, "tau": generation["zeroShotTau"], "bonus": 0.,
              "maxTags": generation["zeroShotMaxTags"], "version": "interest-decision-v1",
              "evaluatedM3": {"method": M3_METHOD, "tau": generation["zeroShotTau"],
                              "bonus": generation["m3Bonus"], "modelVersion": generation["embeddingModelVersion"]},
              "provisional": True}
    def finalized():
        config = {"method": result["method"], "tau": result["tau"], "bonus": result["bonus"],
                  "maxTags": result["maxTags"], "modelVersion": generation["embeddingModelVersion"],
                  "taxonomyVersion": contract["taxonomyVersion"], "taxonomyHash": contract["taxonomyHash"],
                  "decisionVersion": result["version"]}
        return {**result, "selectedConfig": config, "selectedConfigHash": _json_hash(config)}
    if (not judge_report["complete"] or not contract["pairs"] or human_report is None
            or not human_report["complete"] or human_report["pairCount"] < contract["rules"]["minimumHumanPairs"]):
        return finalized()
    human = human_report["methods"]
    if any(human[method]["byCutoff"][str(k)]["pairCount"] == 0 for method in METHODS for k in (5, 10)):
        return finalized()
    gain = judge_report["overall"]["deltaM3MinusM2"]["strictPrecision@5"]
    eligible = gain >= contract["rules"]["minimumMeanGain"] - 1e-12 and all(
        human[M3_METHOD]["byCutoff"][str(k)]["binaryFitRate"] >= human[M2_METHOD]["byCutoff"][str(k)]["binaryFitRate"]
        and human[M3_METHOD]["byCutoff"][str(k)]["noiseRate"] <= human[M2_METHOD]["byCutoff"][str(k)]["noiseRate"]
        for k in (5, 10))
    if eligible:
        result.update(conclusion="M3 채택", reason="사전 +0.03 기준과 사람 적합·잡음 비악화 기준 충족",
                      method=M3_METHOD, bonus=generation["m3Bonus"], provisional=False)
    else:
        result.update(conclusion="추가 실험", reason="사전 기준 미충족: M2를 유지하고 제한된 tau·bonus 후보를 별도 계약으로 비교")
    return finalized()
