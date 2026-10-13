"""Compare a private v3 summary with the frozen v2 run without publishing text."""
import argparse
from pathlib import Path

from src.detail_tagging_batch import read, write_new
from src.detail_tagging import RevisionStore


def load_summary(path):
    value = read(path)
    rows = value["rows"]
    ids = [r["channel_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate comparison channel")
    if any(type(r["first_pass"]) is not bool or type(r["passed"]) is not bool
           or (r["first_pass"] and not r["passed"])
           or bool(r["revision"]) != r["passed"] for r in rows):
        raise ValueError("Invalid comparison row")
    return value, {r["channel_id"]: r for r in rows}


def compare(summary, baseline):
    baseline = Path(baseline)
    old_report, old_rows = load_summary(baseline)
    report, new_rows = load_summary(summary)
    if set(old_rows) != set(new_rows):
        raise ValueError("Comparison input channel sets differ")
    for key in ("inputs_hash", "criteria_hash", "model_version", "policy_version",
                "output_schema_hash", "temperature", "sdk_max_retries", "timeout_seconds"):
        if key not in old_report["conditions"] or key not in report["conditions"]:
            raise ValueError("Unverified comparison condition: " + key)
        if old_report["conditions"][key] != report["conditions"][key]:
            raise ValueError("Comparison conditions differ: " + key)
    old_store = RevisionStore(baseline.parent / "revisions")
    new_store = RevisionStore(Path(summary).parent / "revisions")
    rows = []
    for row in report["rows"]:
        cid = row["channel_id"]
        previous = old_rows[cid]
        item = {**row, "v2_final_failure": not previous["passed"],
                "v2_retry_success": previous["passed"] and not previous["first_pass"]}
        if previous["revision"] and row["revision"]:
            old = old_store.read(previous["revision"])
            new = new_store.read(row["revision"])
            for record, source in ((old, old_report), (new, report)):
                if record["channel_id"] != cid or record["kind"] != "automatic":
                    raise ValueError("Invalid comparison revision")
                for key in ("criteria_hash", "model_version", "policy_version", "output_schema_hash",
                            "prompt_version", "prompt_hash"):
                    if record[key] != source["conditions"][key]:
                        raise ValueError("Revision conditions differ: " + key)
            for key in ("input_hash", "criteria_hash", "model_version", "policy_version", "output_schema_hash"):
                if old[key] != new[key]:
                    raise ValueError("Comparison conditions differ: " + key)
            for axis in ("parents", "topics"):
                before = sorted(r["code"] for r in old["result"][axis])
                after = sorted(r["code"] for r in new["result"][axis])
                item[axis] = {"before": before, "after": after, "equal": before == after}
        rows.append(item)
    failures = [r for r in rows if r["v2_final_failure"]]
    return {"total": len(rows), "v2_first_pass": sum(r["first_pass"] for r in old_rows.values()),
            "v3_first_pass": sum(r["first_pass"] for r in new_rows.values()),
            "v3_after_retries": sum(r["passed"] for r in new_rows.values()),
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
