"""성공한 쌍만 원자 저장하고 미완료 후보만 요청하는 Judge 실행기."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.recommendation.batch import _json_hash, _write_json_atomic
from .contract import (
    RESPONSE_SCHEMA, finite_nonnegative, make_request, read_json,
    validate_judgment, validate_request, validate_response,
)


@dataclass(frozen=True)
class JudgeReply:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    response_id: str = ""
    returned_model: str = ""


class Judge(Protocol):
    def evaluate(self, request: dict, config: dict) -> JudgeReply: ...


class OpenAIInterestJudge:
    def __init__(self, client=None):
        from openai import OpenAI
        # SDK 내부 재시도를 끄고 모든 요청 시도를 자체 원장에 기록한다.
        self.client = client if client is not None else OpenAI(max_retries=0, timeout=120)

    def evaluate(self, request, config):
        import json
        validate_request(request)
        response = self.client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "system", "content": config["prompt"]},
                      {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "interest_relevance", "strict": True, "schema": RESPONSE_SCHEMA}},
        )
        usage = response.usage
        return JudgeReply(response.choices[0].message.content,
                          usage.prompt_tokens if usage else None,
                          usage.completion_tokens if usage else None,
                          response.id, response.model)


def cache_path(cache_dir: Path, pair):
    return cache_dir / (pair["pairId"] + ".json")


def cached_score(cache_dir: Path, pair):
    path = cache_path(cache_dir, pair)
    if not path.exists():
        return None
    try:
        record = read_json(path)
        if (record["identity"] != pair["identity"] or record["pairId"] != pair["pairId"]
                or record["status"] != "success" or record["schemaVersion"] != 1):
            return None
        validate_judgment(record["judgment"])
        if record["judgment"]["candidateId"] != pair["candidateId"]:
            return None
        return record["judgment"]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_scores(contract, cache_dir: Path):
    return {pair["pairId"]: score for pair in contract["pairs"]
            if (score := cached_score(cache_dir, pair)) is not None}


def _tokens_valid(value):
    return value is None or (type(value) is int and value >= 0)


def run_judge(contract, work_dir: Path, cache_dir: Path, judge: Judge, *, max_requests: int | None = None):
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError("max_requests는 양의 정수여야 합니다.")
    ledger_path = work_dir / "attempts.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        "schemaVersion": 1, "contractHash": contract["contractHash"], "attempts": []}
    if ledger.get("contractHash") != contract["contractHash"] or ledger.get("schemaVersion") != 1:
        raise ValueError("요청 원장과 평가 계약이 다릅니다.")
    # 이전 실행이 도중에 종료된 호출은 과금 여부를 모르는 시도로 남긴다.
    for attempt in ledger["attempts"]:
        if attempt["status"] == "started":
            attempt["status"] = "interrupted"
    _write_json_atomic(ledger_path, ledger)
    requested_now = 0
    for interest in contract["interests"]:
        pending = [pair for pair in contract["pairs"] if pair["interestCode"] == interest["code"]
                   and cached_score(cache_dir, pair) is None]
        if not pending:
            continue
        if max_requests is not None and requested_now >= max_requests:
            break
        request = make_request(interest, pending)
        number = len(ledger["attempts"]) + 1
        attempt = {"attemptId": number, "interestCode": interest["code"],
                   "requestHash": _json_hash(request), "candidateCount": len(pending),
                   "status": "started", "inputTokens": None, "outputTokens": None}
        ledger["attempts"].append(attempt)
        # 요청을 보내기 전에 기록해 강제 종료와 네트워크 결과 불명을 구분한다.
        _write_json_atomic(work_dir / "requests" / f"{number}.json", request)
        _write_json_atomic(ledger_path, ledger)
        requested_now += 1
        try:
            reply = judge.evaluate(request, contract["judge"])
            if not _tokens_valid(reply.input_tokens) or not _tokens_valid(reply.output_tokens):
                raise ValueError("Judge 토큰 사용량이 올바르지 않습니다.")
            attempt.update(inputTokens=reply.input_tokens, outputTokens=reply.output_tokens,
                           responseId=reply.response_id, returnedModel=reply.returned_model)
            # 잘못된 JSON에도 발생한 토큰은 실패 비용에 포함한다.
            _write_json_atomic(work_dir / "responses" / f"{number}.json", {"content": reply.content})
            _write_json_atomic(ledger_path, ledger)
            scores = validate_response(reply.content, request)
            for pair in pending:
                _write_json_atomic(cache_path(cache_dir, pair), {
                    "schemaVersion": 1, "status": "success", "pairId": pair["pairId"],
                    "identity": pair["identity"], "judgment": scores[pair["candidateId"]],
                    "responseId": reply.response_id, "returnedModel": reply.returned_model,
                })
            attempt["status"] = "success"
        except Exception as error:
            # 예외 메시지에는 인증 키나 원문이 들어갈 수 있어 타입만 기록한다.
            attempt.update(status="failed", errorType=type(error).__name__)
        _write_json_atomic(ledger_path, ledger)
    scores = load_scores(contract, cache_dir)
    summary = {"contractHash": contract["contractHash"], "pairCount": len(contract["pairs"]),
               "completedPairCount": len(scores), "pendingPairCount": len(contract["pairs"]) - len(scores),
               "complete": len(scores) == len(contract["pairs"]), "requestedNow": requested_now,
               "attemptCount": len(ledger["attempts"]),
               "failureCount": sum(a["status"] in ("failed", "interrupted") for a in ledger["attempts"])}
    _write_json_atomic(work_dir / "run-summary.json", summary)
    return summary


def cost_summary(work_dir: Path, contract, *, input_price=None, output_price=None,
                 billed_usd=None, billing_reference: str | None = None):
    for value in (input_price, output_price, billed_usd):
        if value is not None and not finite_nonnegative(value):
            raise ValueError("비용·단가는 0 이상의 유한한 수여야 합니다.")
    if (input_price is None) != (output_price is None):
        raise ValueError("입력·출력 토큰 단가를 함께 지정해야 합니다.")
    if billed_usd is not None and (not isinstance(billing_reference, str) or not billing_reference.strip()):
        raise ValueError("실제 청구 비용에는 대시보드/청구서 확인 근거가 필요합니다.")
    path = work_dir / "attempts.json"
    ledger = read_json(path) if path.exists() else {"contractHash": contract["contractHash"], "attempts": []}
    if ledger["contractHash"] != contract["contractHash"]:
        raise ValueError("비용 원장과 평가 계약이 다릅니다.")
    attempts = ledger["attempts"]
    inputs = sum(a["inputTokens"] or 0 for a in attempts)
    outputs = sum(a["outputTokens"] or 0 for a in attempts)
    unknown = sum(a["inputTokens"] is None or a["outputTokens"] is None for a in attempts)
    estimate = (inputs * input_price + outputs * output_price) / 1e6 if input_price is not None else None
    return {"scope": "this work directory; shared cache's earlier API costs excluded",
            "attemptCount": len(attempts), "failureCount": sum(a["status"] == "failed" for a in attempts),
            "interruptedCount": sum(a["status"] in ("started", "interrupted") for a in attempts),
            "inputTokens": inputs, "outputTokens": outputs, "unknownUsageAttemptCount": unknown,
            "returnedModels": sorted({a["returnedModel"] for a in attempts if a.get("returnedModel")}),
            "inputPricePerMillionUsd": input_price, "outputPricePerMillionUsd": output_price,
            "estimatedKnownUsageUsd": estimate, "totalCostUsd": billed_usd if billed_usd is not None else (estimate if unknown == 0 else None),
            "costBasis": "user verified invoice" if billed_usd is not None else "token estimate, not actual invoice",
            "billedUsd": billed_usd, "billingReference": billing_reference,
            "actualBillingVerified": billed_usd is not None}
