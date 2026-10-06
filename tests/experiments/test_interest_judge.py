from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.interest_judge import cli
from src.experiments.interest_judge.contract import (
    JudgeConfig, decision_rules, load_contract, make_request, prepare_contract, read_json,
    save_contract, validate_request, validate_response,
)
from src.experiments.interest_judge.human import FIELDS, create_human_sheet, score_human
from src.experiments.interest_judge.runner import (
    JudgeReply, OpenAIInterestJudge, cache_path, cost_summary, load_scores, run_judge,
)
from src.experiments.interest_judge.scoring import aggregate, choose_decision, metrics
from src.recommendation.batch import _json_hash, _write_json_atomic
from src.recommendation.interest import (
    METHODS, M2_METHOD, M3_METHOD, InterestRecommendationResult, InterestRecommender,
)
from src.recommendation.interest_evaluation import write_interest_evaluation
from src.recommendation.manifest import build_manifest, write_manifest
from src.recommendation.models import CreatorProfile
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories


class NoEmbedding:
    dim = 1
    def embed(self, _):
        raise AssertionError("evaluation must never call embedding")


def source_files(root, *, changed=False, count=25):
    root.mkdir(parents=True, exist_ok=True)
    profiles = [CreatorProfile(cid, f"PRIVATE_CREATOR_{cid}" + (" changed" if changed and cid == 1 else ""))
                for cid in range(1, count + 1)]
    manifest = build_manifest(profiles)
    write_manifest(root / "manifest.json", manifest)
    service = InterestRecommender(manifest, load_service_categories(DEFAULT_CATEGORIES_CSV), NoEmbedding(), None)
    bundles = {}
    for category in service.categories:
        bundles[category.code] = {}
        for method in METHODS:
            ids = list(range(1, count + 1)) if method == M2_METHOD else list(range(6, count + 1)) + list(range(1, min(6, count + 1)))
            bundles[category.code][method] = InterestRecommendationResult(
                category.code, method, service.config.embedding_model_version,
                service.input_hash(category.code, method), service.config,
                tuple((cid, round(.9 - rank * .01, 8)) for rank, cid in enumerate(ids[:20])),
            ).to_backend_payload()
    write_interest_evaluation(root / "evaluation", service, bundles)
    return root / "evaluation", root / "manifest.json"


@pytest.fixture
def prepared(tmp_path):
    evaluation, manifest = source_files(tmp_path / "source")
    contract = prepare_contract(evaluation, manifest, JudgeConfig(model="fake-judge-v1"))
    work, cache = tmp_path / "work", tmp_path / "cache"
    save_contract(work / "contract.json", contract)
    return contract, work, cache


class FakeJudge:
    def __init__(self, *, fail_calls=(), invalid_calls=(), stop_call=None):
        self.calls = []
        self.fail_calls = fail_calls
        self.invalid_calls = invalid_calls
        self.stop_call = stop_call

    def evaluate(self, request, config):
        self.calls.append(deepcopy(request))
        number = len(self.calls)
        if number == self.stop_call:
            raise KeyboardInterrupt()
        if number in self.fail_calls:
            raise RuntimeError("SECRET or PRIVATE_CREATOR must not appear in ledger")
        if number in self.invalid_calls:
            return JudgeReply("not JSON", 100, 30)
        rows = [{"candidateId": row["candidateId"], "relevance": 2,
                 "isNoise": False, "noiseType": "none", "reason": "직접 관련"}
                for row in request["candidates"]]
        return JudgeReply(json.dumps({"judgments": rows}), 100, 30, f"reply-{number}", "fake-snapshot")


def completed(prepared):
    contract, work, cache = prepared
    judge = FakeJudge()
    run_judge(contract, work, cache, judge)
    return load_scores(contract, cache)


def test_input_is_blind_deterministic_union_and_shared_once(prepared, tmp_path):
    contract, work, cache = prepared
    assert len(contract["pairs"]) == 255  # 17 * union(1..10, 6..15)
    assert len({p["pairId"] for p in contract["pairs"]}) == 255
    again = prepare_contract(tmp_path / "source/evaluation", tmp_path / "source/manifest.json", JudgeConfig(model="fake-judge-v1"))
    assert contract == again == load_contract(work / "contract.json")
    fake = FakeJudge()
    summary = run_judge(contract, work, cache, fake)
    assert summary["complete"] and summary["completedPairCount"] == 255
    for request in fake.calls:
        assert set(request) == {"interest", "candidates"}
        assert all(set(row) == {"candidateId", "introduction"} for row in request["candidates"])
        text = json.dumps(request)
        assert all(word not in text for word in ("creatorId", "INTEREST_M2", "INTEREST_M3", '"rank"', '"score"', '"method"'))
    first_code = contract["interests"][0]["code"]
    order = [p["creatorId"] for p in contract["pairs"] if p["interestCode"] == first_code]
    assert order != list(range(1, 16))
    assert len(fake.calls) == 17
    fake.calls.clear()
    assert run_judge(contract, work, cache, fake)["requestedNow"] == 0
    assert not fake.calls


@pytest.mark.parametrize("change", ["manifest", "taxonomy", "method", "candidate", "score", "duplicate", "input", "description"])
def test_source_contract_rejects_mismatch_before_api(tmp_path, change):
    evaluation, manifest = source_files(tmp_path / change)
    provenance = read_json(evaluation / "provenance.json")
    code = next(iter(provenance["rankings"]))
    if change == "manifest":
        provenance["manifestHash"] = "0" * 64
    elif change == "taxonomy":
        provenance["generationConfig"]["taxonomyHash"] = "0" * 64
    elif change == "method":
        del provenance["rankings"][code][M2_METHOD]
    elif change in ("candidate", "score", "duplicate", "input"):
        candidates = provenance["rankings"][code][M2_METHOD]["candidates"]
        if change == "candidate": candidates[0]["creatorId"] = 9999
        if change == "score": candidates[0]["score"] = True
        if change == "duplicate": candidates[1]["creatorId"] = candidates[0]["creatorId"]
        if change == "input": candidates[0]["inputHash"] = "0" * 64
    else:
        source = read_json(evaluation / "judge-input.json")
        source["pairs"][0]["interestDescription"] = "different"
        _write_json_atomic(evaluation / "judge-input.json", source)
        provenance["judgeInputHash"] = _json_hash(source)
    _write_json_atomic(evaluation / "provenance.json", provenance)
    with pytest.raises(ValueError):
        prepare_contract(evaluation, manifest, JudgeConfig())


def test_cache_invalidates_changed_intro_model_prompt_and_description(prepared, tmp_path):
    contract, work, cache = prepared
    completed(prepared)
    evaluation, manifest = source_files(tmp_path / "changed", changed=True)
    changed = prepare_contract(evaluation, manifest, JudgeConfig(model="fake-judge-v1"))
    assert len(load_scores(changed, cache)) == 238  # only creator 1's 17 pairs invalidated
    assert changed["contractHash"] != contract["contractHash"]
    for config in (JudgeConfig(model="fake-v2"), replace(JudgeConfig(model="fake-judge-v1"), prompt_version="v2"),
                   replace(JudgeConfig(model="fake-judge-v1"), prompt="다른 기준의 본문")):
        updated = prepare_contract(tmp_path / "source/evaluation", tmp_path / "source/manifest.json", config)
        assert not load_scores(updated, cache)
    edited = deepcopy(contract)
    edited["pairs"][0]["interestDescription"] += "changed"
    _write_json_atomic(work / "edited.json", edited)
    with pytest.raises(ValueError): load_contract(work / "edited.json")
    # 설명 지문이 달라지면 pair ID 자체도 달라져 단순 구 pair ID로 재사용할 수 없다.
    from src.experiments.interest_judge.contract import pair_identity
    pair = edited["pairs"][0]
    assert _json_hash(pair_identity(pair, edited["taxonomyHash"], edited["judge"])) != pair["pairId"]
    with pytest.raises(ValueError): save_contract(work / "contract.json", changed)


def test_failures_partial_completion_resume_and_corrupt_cache(prepared):
    contract, work, cache = prepared
    fake = FakeJudge(fail_calls=(1,), invalid_calls=(2,))
    summary = run_judge(contract, work, cache, fake, max_requests=3)
    assert not summary["complete"] and summary["completedPairCount"] == 15
    ledger = read_json(work / "attempts.json")
    assert [a["status"] for a in ledger["attempts"]] == ["failed", "failed", "success"]
    assert "SECRET" not in json.dumps(ledger)
    assert not aggregate(contract, load_scores(contract, cache))["complete"]
    assert aggregate(contract, load_scores(contract, cache))["overall"] is None
    retry = FakeJudge()
    assert run_judge(contract, work, cache, retry)["complete"]
    assert len(retry.calls) == 16
    pair = contract["pairs"][0]
    cache_path(cache, pair).write_text("broken", encoding="utf-8")
    retry.calls.clear()
    run_judge(contract, work, cache, retry)
    assert len(retry.calls) == 1 and len(retry.calls[0]["candidates"]) == 1
    assert not list(cache.glob("*.tmp"))


def test_interrupt_keeps_successful_pairs_and_records_unknown_cost(prepared):
    contract, work, cache = prepared
    with pytest.raises(KeyboardInterrupt):
        run_judge(contract, work, cache, FakeJudge(stop_call=2))
    assert len(load_scores(contract, cache)) == 15
    assert run_judge(contract, work, cache, FakeJudge())["complete"]
    cost = cost_summary(work, contract, input_price=1., output_price=2.)
    assert cost["interruptedCount"] == cost["unknownUsageAttemptCount"] == 1
    assert cost["totalCostUsd"] is None
    assert cost["estimatedKnownUsageUsd"] == pytest.approx(17 * (100 + 30 * 2) / 1e6)


def test_interrupt_during_cache_write_retries_only_unsaved_pairs(prepared, monkeypatch):
    from src.experiments.interest_judge import runner
    contract, work, cache = prepared
    original = runner._write_json_atomic
    saved = []
    def interrupt(path, payload):
        if path.parent == cache:
            if len(saved) == 4:
                raise KeyboardInterrupt()
            saved.append(path)
        original(path, payload)
    monkeypatch.setattr(runner, "_write_json_atomic", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_judge(contract, work, cache, FakeJudge())
    assert len(load_scores(contract, cache)) == 4
    monkeypatch.setattr(runner, "_write_json_atomic", original)
    fake = FakeJudge()
    assert run_judge(contract, work, cache, fake)["complete"]
    assert len(fake.calls[0]["candidates"]) == 11
    assert not list(cache.glob("*.tmp"))


def valid_response(request):
    return {"judgments": [{"candidateId": row["candidateId"], "relevance": 2,
                           "isNoise": False, "noiseType": "none", "reason": "근거"}
                          for row in request["candidates"]]}


@pytest.mark.parametrize("bad", ["boolean", "score", "missing", "unknown", "duplicate", "noise", "reason", "extra", "nan", "duplicate_key"])
def test_response_schema_validation(prepared, bad):
    contract, _, _ = prepared
    interest = contract["interests"][0]
    pairs = [p for p in contract["pairs"] if p["interestCode"] == interest["code"]][:2]
    request = make_request(interest, pairs)
    response = valid_response(request)
    row = response["judgments"][0]
    if bad == "boolean": row["relevance"] = True
    if bad == "score": row["relevance"] = 3
    if bad == "missing": response["judgments"].pop()
    if bad == "unknown": row["candidateId"] = "unknown"
    if bad == "duplicate": response["judgments"][1] = dict(row)
    if bad == "noise": row["noiseType"] = "ad"
    if bad == "reason": row["reason"] = " "
    if bad == "extra": row["rank"] = 1
    if bad == "nan": row["relevance"] = float("nan")
    content = json.dumps(response)
    if bad == "duplicate_key": content = '{"judgments": [], "judgments": []}'
    with pytest.raises(ValueError): validate_response(content, request)


def test_request_schema_and_fake_openai_wiring(prepared):
    contract, _, _ = prepared
    interest = contract["interests"][0]
    request = make_request(interest, [p for p in contract["pairs"] if p["interestCode"] == interest["code"]])
    broken = deepcopy(request)
    broken["candidates"].append(broken["candidates"][0])
    with pytest.raises(ValueError): validate_request(broken)
    broken = deepcopy(request)
    broken["method"] = "M3"
    with pytest.raises(ValueError): validate_request(broken)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(valid_response(request))))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5), id="response-id", model="snapshot")
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reply = OpenAIInterestJudge(sdk).evaluate(request, contract["judge"])
    assert reply.input_tokens == 10 and reply.returned_model == "snapshot"
    assert calls[0]["response_format"]["json_schema"]["strict"]
    assert json.loads(calls[0]["messages"][1]["content"]) == request


def test_metrics_known_values_shortage_ndcg_and_paired_bootstrap(prepared):
    rows = [{"relevance": 2, "isNoise": False}, {"relevance": 1, "isNoise": True}]
    result = metrics(rows, rows, 5)
    assert result["strictPrecision@5"] == .2 and result["lenientPrecision@5"] == .4
    assert result["noiseRate@5"] == .2 and result["gradedNdcg@5"] == 1.
    reversed_result = metrics(rows[::-1], rows, 5)
    assert reversed_result["gradedNdcg@5"] == pytest.approx((1 + 3 / 1.584962500721156) / (3 + 1 / 1.584962500721156))
    contract, _, _ = prepared
    scores = completed(prepared)
    for pair in contract["pairs"]:
        scores[pair["pairId"]]["relevance"] = 0 if pair["creatorId"] <= 5 else 2
    report = aggregate(contract, scores)
    assert report["overall"]["deltaM3MinusM2"]["strictPrecision@5"] == 1.
    interval = report["pairedBootstrap"]["intervals"]["strictPrecision@5"]
    assert interval == {"mean": 1., "lower": 1., "upper": 1.}
    assert report["winLossTie"] == {"win": 17, "loss": 0, "tie": 0}
    assert report == aggregate(contract, scores)


def fill_sheet(work, score="2", noise="0"):
    with (work / "human-blank.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.update(relevance=score, isNoise=noise, noiseType="none" if noise == "0" else "ad", reason="사람 근거")
    target = work / "human-filled.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target, rows


def test_stratified_blank_human_sheet_and_independent_agreement(prepared):
    contract, work, _ = prepared
    scores = completed(prepared)
    for pair in contract["pairs"]:
        score = scores[pair["pairId"]]
        score["relevance"] = 0 if pair["creatorId"] <= 5 else 2
        if pair["creatorId"] == 1:
            score.update(isNoise=True, noiseType="giveaway")
    sampling = create_human_sheet(contract, scores, work)
    assert sampling["sampleSize"] == 50
    assert sampling["strataSelectedCounts"]["noise"] > 0
    assert any(tag.startswith("judgeDisagreement") for tag in sampling["strataSelectedCounts"])
    for interest in contract["interests"]:
        assert sampling["strataSelectedCounts"]["interest:" + interest["code"]] >= 1
    with (work / "human-blank.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["sampleId"] for row in rows}) == 50
    assert all(row[field] == "" for row in rows for field in ("relevance", "isNoise", "noiseType", "reason"))
    assert all(word not in rows[0] for word in ("judgeScore", "rank", "score", "method", "strata"))
    before = (work / "human-blank.csv").read_bytes()
    assert create_human_sheet(contract, scores, work) == sampling
    assert (work / "human-blank.csv").read_bytes() == before
    filled, rows = fill_sheet(work)
    human = score_human(contract, scores, work, filled)
    assert human["complete"] and human["pairCount"] == human["comparedPairCount"] == 50
    expected = sum(scores[row["sampleId"]]["relevance"] == 2 for row in rows) / 50
    assert human["threeLevelExactAgreement"] == human["binaryAgreement"] == expected
    assert human["fullRankingMetrics"] is None
    assert human["methods"][M2_METHOD]["pairCount"] > 0
    assert (work / "human-blank.csv").read_bytes() == before


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "text", "score", "missing", "partial", "blank-path"])
def test_human_sheet_validation_and_partial_progress(prepared, kind):
    contract, work, _ = prepared
    scores = completed(prepared)
    create_human_sheet(contract, scores, work)
    filled, rows = fill_sheet(work)
    if kind == "duplicate": rows[1] = dict(rows[0])
    if kind == "unknown": rows[0]["sampleId"] = "unknown"
    if kind == "text": rows[0]["introduction"] = "changed"
    if kind == "score": rows[0]["relevance"] = "3"
    if kind == "missing": rows.pop()
    if kind == "partial": rows[0].update(relevance="", isNoise="", noiseType="", reason="")
    if kind == "blank-path":
        with pytest.raises(ValueError): score_human(contract, scores, work, work / "human-blank.csv")
        return
    with filled.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if kind == "partial":
        report = score_human(contract, scores, work, filled)
        assert not report["complete"] and report["pairCount"] == 49
    else:
        with pytest.raises(ValueError): score_human(contract, scores, work, filled)


def test_human_sheet_cannot_claim_fifty_when_input_small(tmp_path):
    evaluation, manifest = source_files(tmp_path / "small", count=2)
    contract = prepare_contract(evaluation, manifest, JudgeConfig())
    work, cache = tmp_path / "work", tmp_path / "cache"
    run_judge(contract, work, cache, FakeJudge())
    with pytest.raises(ValueError, match="부족"):
        create_human_sheet(contract, load_scores(contract, cache), work)


def test_empty_manifest_is_complete_but_cannot_adopt_or_make_human_sample(tmp_path):
    evaluation, manifest = source_files(tmp_path / "empty", count=0)
    contract = prepare_contract(evaluation, manifest, JudgeConfig())
    work, cache = tmp_path / "work", tmp_path / "cache"
    fake = FakeJudge()
    result = run_judge(contract, work, cache, fake)
    assert result["complete"] and not fake.calls
    judge = aggregate(contract, {})
    assert judge["overall"]["methods"][M2_METHOD]["strictPrecision@5"] == 0
    assert choose_decision(contract, judge, None)["method"] == M2_METHOD
    with pytest.raises(ValueError, match="부족"): create_human_sheet(contract, {}, work)


def test_decision_gates_and_public_report_has_no_private_cases(prepared):
    contract, work, cache = prepared
    scores = completed(prepared)
    judge = aggregate(contract, scores)
    assert choose_decision(contract, judge, None)["method"] == M2_METHOD
    human = {"complete": True, "pairCount": 50, "methods": {
        method: {"byCutoff": {str(k): {"pairCount": 30, "binaryFitRate": .8, "noiseRate": .1}
                              for k in (5, 10)}} for method in METHODS}}
    judge["overall"]["deltaM3MinusM2"]["strictPrecision@5"] = .03
    assert choose_decision(contract, judge, human)["conclusion"] == "M3 채택"
    human["methods"][M3_METHOD]["byCutoff"]["5"]["binaryFitRate"] = .79
    assert choose_decision(contract, judge, human)["conclusion"] == "추가 실험"
    human["methods"][M3_METHOD]["byCutoff"]["5"]["binaryFitRate"] = .8
    human["methods"][M3_METHOD]["byCutoff"]["5"]["noiseRate"] = .11
    assert choose_decision(contract, judge, human)["method"] == M2_METHOD
    report = cli.write_report(contract, work, cache)
    public = read_json(work / "public-summary.json")
    assert report["decision"]["conclusion"] == "M2 유지"
    text = json.dumps(public)
    assert "PRIVATE_CREATOR" not in text and '"pairId"' not in text and '"candidateId"' not in text
    assert "직접 관련" not in text and '"reason"' not in json.dumps(public["judge"])
    assert public["judge"]["complete"]


def test_cost_counts_invalid_json_and_separates_invoice(prepared):
    contract, work, cache = prepared
    run_judge(contract, work, cache, FakeJudge(invalid_calls=(1,)), max_requests=1)
    cost = cost_summary(work, contract, input_price=1., output_price=2.)
    assert cost["failureCount"] == 1 and cost["inputTokens"] == 100
    assert cost["estimatedKnownUsageUsd"] == pytest.approx(.00016)
    assert not cost["actualBillingVerified"]
    with pytest.raises(ValueError): cost_summary(work, contract, billed_usd=.2)
    with pytest.raises(ValueError): cost_summary(work, contract, input_price=float("nan"), output_price=1)
    cost = cost_summary(work, contract, billed_usd=.2, billing_reference="invoice checked")
    assert cost["actualBillingVerified"] and cost["totalCostUsd"] == .2


def test_cli_approval_guard_and_prepare_report_without_api(tmp_path, monkeypatch):
    evaluation, manifest = source_files(tmp_path / "source")
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path)
    work = tmp_path / "work"
    assert cli.main(["prepare", "--work-dir", str(work), "--evaluation-dir", str(evaluation), "--manifest", str(manifest)]) == 0
    calls = []
    monkeypatch.setattr(cli, "OpenAIInterestJudge", lambda: calls.append(1))
    with pytest.raises(ValueError, match="별도"):
        cli.main(["run", "--work-dir", str(work), "--cache-dir", str(tmp_path / "cache")])
    assert not calls
    assert cli.main(["report", "--work-dir", str(work), "--cache-dir", str(tmp_path / "cache")]) == 1
    public = read_json(work / "public-summary.json")
    assert not public["judge"]["complete"] and public["decision"]["method"] == M2_METHOD
    with pytest.raises(ValueError, match="results"):
        cli.main(["prepare", "--work-dir", str(tmp_path.parent / "tracked"), "--evaluation-dir", str(evaluation), "--manifest", str(manifest)])


def test_rules_cannot_change_after_prepare(prepared):
    contract, work, _ = prepared
    contract["rules"]["minimumMeanGain"] = 0
    contract["contractHash"] = _json_hash({k: v for k, v in contract.items() if k != "contractHash"})
    _write_json_atomic(work / "contract.json", contract)
    with pytest.raises(ValueError): load_contract(work / "contract.json")
    assert decision_rules()["minimumMeanGain"] == .03
