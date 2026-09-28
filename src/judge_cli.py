"""판정 시트를 터미널에서 하나씩 채우는 대화형 도구다.

`python3 -m src.judge_cli`로 실행하면 results/judge_sheet.csv를, `--file results/spot_check.csv`처럼
지정하면 다른 판정 시트(같은 4열 형식)를 채운다. 답할 때마다 즉시 파일에 저장하므로 언제든
Ctrl+C나 q로 멈추고 나중에 다시 실행해 이어서 할 수 있다 (비어 있는 score만 물어본다).
어떤 방식이 이 후보를 뽑았는지는 보여주지 않는다 — 그건 judge_provenance.csv에만 있고
채점 단계에서만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import RESULTS_DIR
from src.data import Creator, load_creators

VALID_SCORES = {"0", "1", "2"}
FIELDNAMES = ["pair_id", "query_id", "candidate_id", "score"]


def load_rows(path: Path) -> list[dict[str, str]]:
    """판정 시트를 읽는다. 쿼리별로 묶어 한 사람을 기준으로 후보를 몰아서 볼 수 있게 정렬한다.

    이 정렬은 순전히 보기 편하라고 하는 것이며, 어떤 방식이 뽑았는지는 여전히 감춰진다.
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: (r["query_id"], r["candidate_id"]))


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    """판정 시트를 그대로 다시 쓴다. 답을 받을 때마다 호출해 진행 상황을 잃지 않게 한다."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def format_prompt(row: dict[str, str], creators_by_id: dict[str, Creator], index: int, total: int) -> str:
    """한 쌍을 보여줄 문구를 만든다. 소개가 없는 크리에이터는 이벤트 제목으로 대체해 보여준다."""
    query = creators_by_id[row["query_id"]]
    candidate = creators_by_id[row["candidate_id"]]
    return (
        f"\n[{index}/{total}] 쿼리 {query.id} {query.name}\n"
        f"  소개: {query.input_text()}\n"
        f"  후보 {candidate.id} {candidate.name}\n"
        f"  소개: {candidate.input_text()}\n"
        "점수 (0=무관 1=관련 2=매우관련, s=건너뛰기, q=저장 후 종료): "
    )


def main() -> None:
    """비어 있는 score만 골라 하나씩 물어보고, 답할 때마다 바로 저장한다."""
    parser = argparse.ArgumentParser(description="판정 시트를 터미널에서 채우는 도구")
    parser.add_argument("--file", default=None, help="채울 판정 시트 경로 (기본: results/judge_sheet.csv)")
    args = parser.parse_args()
    path = Path(args.file) if args.file else RESULTS_DIR / "judge_sheet.csv"

    rows = load_rows(path)
    creators_by_id = {c.id: c for c in load_creators()}

    pending = [row for row in rows if not row["score"].strip()]
    total_all = len(rows)
    done_before = total_all - len(pending)
    print(f"전체 {total_all}쌍 중 {done_before}쌍 완료, {len(pending)}쌍 남음")

    for offset, row in enumerate(pending):
        prompt = format_prompt(row, creators_by_id, done_before + offset + 1, total_all)
        while True:
            answer = input(prompt).strip().lower()
            if answer == "q":
                write_rows(rows, path)
                print("저장하고 종료합니다. 다시 실행하면 이어서 물어봅니다.")
                return
            if answer == "s":
                break
            if answer in VALID_SCORES:
                row["score"] = answer
                write_rows(rows, path)
                break
            print("0, 1, 2, s(건너뛰기), q(종료) 중 하나를 입력하세요.")

    write_rows(rows, path)
    if path.name == "spot_check.csv":
        print("모든 쌍의 판정이 끝났습니다. python3 -m src.pipeline spot-check-report 를 실행하세요.")
    else:
        print("모든 쌍의 판정이 끝났습니다. python3 -m src.pipeline score-judgments 를 실행하세요.")


if __name__ == "__main__":
    main()
