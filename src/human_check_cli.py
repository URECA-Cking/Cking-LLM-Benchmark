"""Review frozen subtopic assignments or blind recommendation pairs locally."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.flow_eval import digest
from src.real_eval import load_real_data


def save(path, value):
    """Replace a checkpoint atomically after every explicit human answer."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def run_sheet(rows, path, read=input, emit=print):
    """Keep human decisions separate and reject stale evidence on resume."""
    contract = digest(rows)
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"contract_hash": contract, "source": "human", "scores": {}}
    )
    if state["contract_hash"] != contract:
        raise ValueError("Review evidence or sample changed; use a new output file")
    pending = [r for r in rows if r["id"] not in state["scores"]]
    emit(
        f"전체 {len(rows)}개 / 완료 {len(rows)-len(pending)}개 / 남음 {len(pending)}개"
    )
    save(path, state)
    for position, row in enumerate(pending, 1):
        started = time.monotonic()
        emit(
            f"\n{'=' * 60}\n[{position}/{len(pending)}] {row['title']}\n{row['display']}\n"
        )
        while True:
            try:
                answer = (
                    read("1=적절함 0=부적절함 u=판단 어려움 s=건너뛰기 q=종료 > ")
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                emit("\n저장된 상태로 종료합니다.")
                return
            if answer == "q":
                emit("저장된 상태로 종료합니다. 같은 명령으로 이어서 할 수 있습니다.")
                return
            if answer == "s":
                break
            if answer in ("0", "1", "u"):
                try:
                    note = read("이유 메모 (없으면 Enter) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    note = ""
                state["scores"][row["id"]] = {
                    "score": None if answer == "u" else int(answer),
                    "status": "uncertain" if answer == "u" else "rated",
                    "note": note,
                    "evidence_hash": digest(row),
                    "elapsed_seconds": time.monotonic() - started,
                }
                save(path, state)
                break
            emit("0, 1, u, s, q 중 하나를 입력하세요.")
    emit(
        f"이번 점검 종료. 평가 파일: {path}\n건너뛴 항목은 다음 실행에서 다시 표시됩니다."
    )


def prepare(args):
    """Build a small diagnostic sample, never a final representative test."""
    rng = random.Random(42)
    if args.mode == "recommendations":
        pairs = json.loads((args.experiment_dir / "evaluation_pairs.json").read_text())
        rows = [
            {
                "id": key,
                "title": {
                    "creators": "크리에이터 기반 추천",
                    "favorites": "즐겨찾기 기반 추천",
                    "tags": "관심 태그 기반 추천",
                }[r["route"]],
                "display": f"[좋아하는 크리에이터 / 관심 입력]\n{r['input']}\n\n[추천 후보]\n{r['candidate']}\n\n이 입력을 좋아하는 사람에게 이 후보를 추천할 만한가요?",
            }
            for key, r in sorted(pairs.items())
        ]
    else:
        data = load_real_data(args.data_dir)
        pack = json.loads((args.subtopic_dir / "subtopics_llm.json").read_text())
        taxonomy = json.loads(args.taxonomy.read_text())
        topics = {t["code"]: t for t in taxonomy["subtopics"]}
        rows = []
        for channel in data.pool:
            record = pack["tags"].get(channel.id)
            if record is None:
                continue
            if record["text_hash"] != digest(channel.text()):
                raise ValueError("Subtopic source text changed")
            assigned = record["topics"]
            details = (
                "\n\n".join(
                    f"태그: {topics[t['code']]['name']}\n정의: {topics[t['code']]['definition']}\n선택 인용: {t['evidence']}"
                    for t in assigned
                )
                or "(세부 태그 미분류)"
            )
            rows.append(
                {
                    "id": channel.id,
                    "title": "세부 태그 점검",
                    "display": f"[소개글]\n{channel.text()}\n\n[분류 결과]\n{details}\n\n소개글에 비춰 태그 묶음이 적절한가요? 미분류라면 미분류가 적절한가요?",
                }
            )
    rng.shuffle(rows)
    return rows[: args.limit]


def parse_args(argv=None):
    """Require an external data location only for the tag review mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tags", "recommendations"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--data-dir",
        type=lambda value: Path(value).expanduser(),
    )
    parser.add_argument(
        "--subtopic-dir", type=Path, default=Path("results/subtopics_v1")
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("results/bonus_ablation_v2")
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("data/creator-subtopics.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.mode == "tags" and args.data_dir is None:
        parser.error("tags mode requires --data-dir")
    return args


def main():
    """Open a local review session without calling any provider."""
    args = parse_args()
    path = args.output or Path(f"results/human_check_{args.mode}_{args.limit}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        args.mode == "recommendations"
        and (args.experiment_dir / "final_plan.json").exists()
    ):
        print(
            "최종 입력 비교: 1인 평가·같은 후보 풀의 미평가 입력입니다. 실제 사용자 일반화와 구분합니다. API 호출·비용 없음."
        )
    else:
        print(
            "진단용 표본입니다. 최종 성능을 추정하는 표본은 아닙니다. API 호출·비용 없음."
        )
    print(
        "추천 점검은 방법명·기존 LLM 점수를 숨깁니다. 태그 점검은 태그 묶음 전체를 판단합니다."
    )
    run_sheet(prepare(args), path)


if __name__ == "__main__":
    main()
