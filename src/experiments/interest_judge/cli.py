"""유료 호출을 명시적으로 분리한 prepare/run/human-sheet/report 명령."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import OPENAI_JUDGE_MODEL, RESULTS_DIR
from src.recommendation.batch import _write_json_atomic
from .contract import JudgeConfig, load_contract, prepare_contract, save_contract
from .human import create_human_sheet, score_human
from .runner import OpenAIInterestJudge, cost_summary, load_scores, run_judge
from .scoring import aggregate, choose_decision


def _private_path(path):
    if not path.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("원문·개별 판정·캐시는 Git 제외 results/ 아래에 저장해야 합니다.")


def build_parser():
    parser = argparse.ArgumentParser(description="#44 관심 분야 M2/M3 블라인드 오프라인 Judge 비교")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "human-sheet", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--work-dir", type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--evaluation-dir", type=Path, required=True)
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--judge-model", default=OPENAI_JUDGE_MODEL)
        else:
            sub.add_argument("--cache-dir", type=Path, default=RESULTS_DIR / "recommendation" / "interest-judge-cache")
        if name == "run":
            sub.add_argument("--approve-data-transfer-and-cost", action="store_true",
                             help="실제 데이터 전송·비용에 대해 별도 승인을 받은 경우에만 지정")
            sub.add_argument("--max-requests", type=int)
        if name == "human-sheet":
            sub.add_argument("--sample-size", type=int, default=50)
        if name == "report":
            sub.add_argument("--human-filled", type=Path)
            sub.add_argument("--input-price-per-million-usd", type=float)
            sub.add_argument("--output-price-per-million-usd", type=float)
            sub.add_argument("--billed-usd", type=float)
            sub.add_argument("--billing-reference")
    return parser


def write_report(contract, work_dir, cache_dir, *, human_filled=None, input_price=None,
                 output_price=None, billed_usd=None, billing_reference=None):
    scores = load_scores(contract, cache_dir)
    judge = aggregate(contract, scores)
    human = score_human(contract, scores, work_dir, human_filled) if human_filled is not None else None
    cost = cost_summary(work_dir, contract, input_price=input_price, output_price=output_price,
                        billed_usd=billed_usd, billing_reference=billing_reference)
    decision = choose_decision(contract, judge, human)
    report = {"schemaVersion": 1, "contractHash": contract["contractHash"],
              "taxonomyHash": contract["taxonomyHash"], "manifestHash": contract["manifestHash"],
              "judgeConfig": contract["judge"], "rules": contract["rules"],
              "pairCount": len(contract["pairs"]), "completedPairCount": len(scores),
              "pendingPairCount": len(contract["pairs"]) - len(scores),
              "judge": judge, "human": human, "cost": cost, "decision": decision,
              "limitations": ["LLM Judge는 정답이 아니며 사람 표본·실제 반응 검증이 필요함",
                              "17개 분야 bootstrap은 독립 회원·Creator 표본의 신뢰구간이 아님",
                              "사람 표본은 사례 층화로 선택되어 전체 적합 비율의 불편 추정이 아님",
                              "nDCG의 이상 순위는 두 방법 Top-10 합집합에 한정됨",
                              "공유 캐시의 이전 실행 비용과 응답 없는 실패 비용은 확인 범위 밖"]}
    _write_json_atomic(work_dir / "report.json", report)
    # 공개 가능한 집계는 allowlist로 만든다. 개별 ID·판정 근거·원문·청구 근거를 빼낸다.
    public = {key: report[key] for key in ("schemaVersion", "contractHash", "taxonomyHash", "manifestHash",
                                         "rules", "pairCount", "completedPairCount", "pendingPairCount",
                                         "decision", "limitations")}
    public["judgeConfig"] = {key: contract["judge"][key] for key in (
        "model", "promptVersion", "promptHash", "responseSchemaHash", "shuffleSeed", "temperature")}
    public["judge"] = {key: judge[key] for key in ("complete", "completedInterestCount", "perInterest",
                                                   "overall", "pairedBootstrap", "winLossTie")}
    public["human"] = ({key: human[key] for key in ("complete", "pairCount", "requestedPairCount", "comparedPairCount",
                                                    "threeLevelExactAgreement", "binaryAgreement", "methods", "metricScope")}
                        if human is not None else None)
    public["cost"] = {key: value for key, value in cost.items() if key != "billingReference"}
    _write_json_atomic(work_dir / "public-summary.json", public)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    _private_path(args.work_dir)
    if args.command == "prepare":
        contract = prepare_contract(args.evaluation_dir, args.manifest, JudgeConfig(model=args.judge_model))
        save_contract(args.work_dir / "contract.json", contract)
        _write_json_atomic(args.work_dir / "decision-plan.json", {
            "contractHash": contract["contractHash"], "rules": contract["rules"],
            "judge": contract["judge"], "generationConfig": contract["generationConfig"]})
        print(f"[interest-judge prepare] pairs={len(contract['pairs'])}; API calls=0")
        return 0
    _private_path(args.cache_dir)
    contract = load_contract(args.work_dir / "contract.json")
    if args.command == "run":
        if not args.approve_data_transfer_and_cost:
            raise ValueError("별도 데이터 전송·비용 승인 후 --approve-data-transfer-and-cost를 지정하세요.")
        # 검증을 통과하기 전에 외부 클라이언트를 만들지 않는다.
        if args.max_requests is not None and args.max_requests < 1:
            raise ValueError("max_requests는 양의 정수여야 합니다.")
        result = run_judge(contract, args.work_dir, args.cache_dir, OpenAIInterestJudge(), max_requests=args.max_requests)
        print(f"[interest-judge run] completed={result['completedPairCount']}/{result['pairCount']}; failures={result['failureCount']}")
        return 0 if result["complete"] else 1
    if args.command == "human-sheet":
        result = create_human_sheet(contract, load_scores(contract, args.cache_dir), args.work_dir, sample_size=args.sample_size)
        print(f"[interest-judge human-sheet] pairs={result['sampleSize']}; labels=blank")
        return 0
    if args.human_filled is not None:
        _private_path(args.human_filled)
    report = write_report(contract, args.work_dir, args.cache_dir, human_filled=args.human_filled,
                          input_price=args.input_price_per_million_usd, output_price=args.output_price_per_million_usd,
                          billed_usd=args.billed_usd, billing_reference=args.billing_reference)
    print(f"[interest-judge report] complete={report['judge']['complete']}; conclusion={report['decision']['conclusion']}")
    return 0 if report["judge"]["complete"] and (report["human"] is None or report["human"]["complete"]) else 1
