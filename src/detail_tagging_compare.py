"""Compare a private v3 summary with the frozen v2 run without publishing text."""
import argparse
from pathlib import Path

from src.detail_tagging_batch import read, write_new


def compare(summary, baseline):
    baseline = Path(baseline)
    index = read(baseline / "revision-index.json")
    first_failures = {r["channel_id"] for r in read(baseline / "first-new-key-summary.json")["failures"]}
    rows = []
    new_store = Path(summary).parent / "revisions"
    report = read(summary)
    for row in report["rows"]:
        cid = row["channel_id"]
        item = {**row, "v2_final_failure": cid not in index,
                "v2_retry_success": cid in index and cid in first_failures}
        if cid in index and row["revision"]:
            old = read(baseline / index[cid])
            new = read(new_store / (row["revision"] + ".json"))
            for key in ("input_hash", "criteria_hash", "model_version", "policy_version", "output_schema_hash"):
                if old[key] != new[key]:
                    raise ValueError("Comparison conditions differ: " + key)
            for axis in ("parents", "topics"):
                before = sorted(r["code"] for r in old["result"][axis])
                after = sorted(r["code"] for r in new["result"][axis])
                item[axis] = {"before": before, "after": after, "equal": before == after}
        rows.append(item)
    failures = [r for r in rows if r["v2_final_failure"]]
    return {"total": len(rows), "v2_first_pass": 36, "v3_first_pass": report["first_pass"],
            "v3_after_retries": report["passed_after_retries"],
            "v2_final_failure_count": len(failures),
            "v2_final_failure_v3_first_pass": sum(r["first_pass"] for r in failures),
            "v2_final_failure_v3_after_retries": sum(r["passed"] for r in failures),
            "rows": rows, "human_review": "pending",
            "limitations": "Successful revisions are selected outcomes; all attempts must also be inspected. Format validity and tag agreement do not prove semantic accuracy or recommendation improvement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, compare(args.summary, args.baseline))


if __name__ == "__main__":
    main()
