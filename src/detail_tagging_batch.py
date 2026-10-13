"""Private, immutable per-attempt detail tagging regression runs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from src.detail_tagging import (POLICY_VERSION, PROMPT_VERSION, RevisionStore,
                               build_prompt, candidate_schema, classify_with_model, digest)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def conditions(inputs, taxonomy, model):
    return {"inputs_hash": digest(inputs), "criteria_hash": digest(taxonomy),
            "model_version": model, "policy_version": POLICY_VERSION,
            "prompt_version": PROMPT_VERSION, "prompt_hash": digest(build_prompt(taxonomy)),
            "output_schema_hash": digest(candidate_schema(taxonomy)),
            "temperature": 0, "sdk_max_retries": 0}


def run_batch(client, inputs, taxonomy, model, output, max_attempts=3):
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    ids = [r["channel_id"] for r in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input channel")
    output = Path(output)
    contract = {**conditions(inputs, taxonomy, model), "max_attempts": max_attempts}
    manifest = output / "run-conditions.json"
    if manifest.exists():
        if read(manifest) != contract:
            raise ValueError("Run conditions mismatch; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Unversioned output directory; use a new directory")
        write_new(manifest, contract)
    store = RevisionStore(output / "revisions")
    rows = []
    for row in inputs:
        channel = row["channel_id"]
        # Hash filenames instead of trusting external channel IDs as path components.
        folder = output / "attempts" / digest(channel)
        previous = sorted(folder.glob("*.json"))
        history = [read(p) for p in previous]
        for entry in history:
            if entry["conditions"] != contract or entry["input_hash"] != digest(row["text"]):
                raise ValueError("Attempt provenance mismatch")
            if entry.get("revision"):
                record = store.read(entry["revision"])
                for key in ("criteria_hash", "model_version", "policy_version",
                            "prompt_version", "prompt_hash", "output_schema_hash"):
                    if record[key] != contract[key]:
                        raise ValueError("Revision provenance mismatch")
                if record["kind"] != "automatic" or record["input_hash"] != digest(row["text"]):
                    raise ValueError("Invalid automatic revision")
        if not any(e["status"] == "passed" for e in history):
            for number in range(len(history) + 1, max_attempts + 1):
                entry = {"channel_id": channel, "input_hash": digest(row["text"]),
                         "attempt": number, "conditions": contract,
                         "started_at": datetime.now(timezone.utc).isoformat()}
                def capture(**kwargs):
                    entry["request"] = kwargs
                    response = client.chat.completions.create(**kwargs)
                    entry["response"] = response.model_dump()
                    return response
                proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
                provider_failure = False
                try:
                    record = classify_with_model(proxy, channel, row["text"], taxonomy, model)
                    entry.update(status="passed", revision=store.save(record)["revision"])
                except ValueError as error:
                    entry.update(status="validation_error", error=str(error))
                except Exception as error:
                    provider_failure = True
                    entry.update(status="provider_error", error_type=type(error).__name__,
                                 http_status=getattr(error, "status_code", None))
                write_new(folder / f"{number:03d}-{uuid4().hex}.json", entry)
                history.append(entry)
                if provider_failure:
                    raise RuntimeError("Provider failure; attempt preserved, run stopped")
                if entry["status"] == "passed":
                    break
        rows.append({"channel_id": channel, "first_pass": history[0]["status"] == "passed",
                     "passed": any(e["status"] == "passed" for e in history),
                     "attempts": len(history),
                     "revision": next((e["revision"] for e in history if e["status"] == "passed"), None)})
    summary = {"conditions": contract, "total": len(rows),
               "first_pass": sum(r["first_pass"] for r in rows),
               "passed_after_retries": sum(r["passed"] for r in rows),
               "rows": rows, "human_review": "pending"}
    write_new(output / f"summary-{uuid4().hex}.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    parser.add_argument("--model", default="gpt-5.4-nano-2026-03-17")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--tag-llm", action="store_true", required=True)
    args = parser.parse_args(argv)
    load_dotenv()
    result = run_batch(OpenAI(timeout=60, max_retries=0), read(args.inputs),
                       read(args.taxonomy), args.model, args.output, args.max_attempts)
    print(json.dumps({k: result[k] for k in ("total", "first_pass", "passed_after_retries")}))
    return 0 if result["passed_after_retries"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
