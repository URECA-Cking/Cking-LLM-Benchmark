"""Private, immutable per-attempt detail tagging regression runs."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from src.detail_tagging import (POLICY_VERSION, PROMPT_VERSION, PROMPT_VERSIONS, V4_PROMPT_VERSION,
                               DEDUP_PROMPT_VERSION, V5_PROMPT_VERSION, RevisionStore,
                               build_prompt, candidate_schema, classify_with_model, digest)


class HarnessConfigurationError(RuntimeError):
    """Local request configuration drift detected before any provider call."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _publish(temporary, path):
    if os.name == "nt":
        # Windows rename fails if the destination exists; POSIX rename replaces it.
        os.rename(temporary, path)
    else:
        os.link(temporary, path)


def write_new(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def conditions(inputs, taxonomy, model, client, prompt_version, temperature):
    timeout = client.timeout
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("Batch client timeout must be positive numeric seconds")
    if not isinstance(client.max_retries, int) or client.max_retries < 0:
        raise ValueError("Invalid SDK retry count")
    return {"inputs_hash": digest(inputs), "criteria_hash": digest(taxonomy),
            "model_version": model, "policy_version": POLICY_VERSION,
            "prompt_version": prompt_version, "prompt_hash": digest(build_prompt(taxonomy, prompt_version)),
            "output_schema_hash": digest(candidate_schema(taxonomy)),
            "temperature": temperature, "sdk_max_retries": client.max_retries,
            "timeout_seconds": timeout}


def run_batch(client, inputs, taxonomy, model, output, max_attempts=3,
              prompt_version=PROMPT_VERSION, temperature=0, human_criteria=None, ai_proxy_criteria=None):
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    ids = [r["channel_id"] for r in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input channel")
    output = Path(output)
    contract = {**conditions(inputs, taxonomy, model, client, prompt_version, temperature),
                "max_attempts": max_attempts}
    if human_criteria is not None and ai_proxy_criteria is not None:
        raise ValueError("Human and AI proxy criteria must not be mixed")
    if human_criteria is not None or ai_proxy_criteria is not None:
        from src.detail_tagging_human_eval import validated_criteria
        criteria = human_criteria if human_criteria is not None else ai_proxy_criteria
        origin = "human_declared" if human_criteria is not None else "ai_proxy"
        pack, _ = validated_criteria(criteria, origin)
        if (pack["manifest"]["inputs_hash"] != digest(inputs)
                or pack["manifest"]["criteria_hash"] != digest(taxonomy)):
            raise ValueError("Human criteria input/taxonomy differs")
        if human_criteria is not None:
            contract["human_criteria_hash"] = criteria["hash"]
        else:
            contract.update(reference_origin="ai_proxy", reference_criteria_hash=criteria["hash"])
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
            request = entry.get("request", {})
            if (request.get("temperature") != contract["temperature"]
                    or request.get("model") != model
                    or request.get("messages") != [
                        {"role": "system", "content": build_prompt(taxonomy, prompt_version)},
                        {"role": "user", "content": json.dumps({"creator": row["text"]}, ensure_ascii=False)}]
                    or request.get("response_format", {}).get("json_schema", {}).get("schema") != candidate_schema(taxonomy)):
                raise ValueError("Attempt request provenance mismatch")
            if entry.get("revision"):
                record = store.read(entry["revision"])
                for key in ("criteria_hash", "model_version", "policy_version",
                            "prompt_version", "prompt_hash", "output_schema_hash"):
                    if record[key] != contract[key]:
                        raise ValueError("Revision provenance mismatch")
                if (record["kind"] != "automatic" or record["input_hash"] != digest(row["text"])
                        or record["channel_id"] != channel):
                    raise ValueError("Invalid automatic revision")
        if not any(e["status"] == "passed" for e in history):
            for number in range(len(history) + 1, max_attempts + 1):
                entry = {"channel_id": channel, "input_hash": digest(row["text"]),
                         "attempt": number, "conditions": contract,
                         "started_at": datetime.now(timezone.utc).isoformat()}
                def capture(**kwargs):
                    entry["request"] = kwargs
                    if (kwargs["temperature"] != contract["temperature"]
                            or client.max_retries != contract["sdk_max_retries"]
                            or client.timeout != contract["timeout_seconds"]):
                        raise HarnessConfigurationError("Actual call conditions differ from manifest")
                    response = client.chat.completions.create(**kwargs)
                    entry["response"] = response.model_dump()
                    return response
                proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
                provider_failure = False
                try:
                    record = classify_with_model(proxy, channel, row["text"], taxonomy, model,
                                                 prompt_version, temperature)
                    entry.update(status="passed", revision=store.save(record)["revision"])
                except HarnessConfigurationError as error:
                    # No provider call occurred: keep diagnostics outside the attempt budget.
                    entry.update(status="harness_error", error=str(error), provider_called=False)
                    entry["next_attempt"] = entry.pop("attempt")
                    write_new(output / "harness-errors" / f"{uuid4().hex}.json", entry)
                    raise
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
               "rows": rows, "human_review": "not_performed" if ai_proxy_criteria is not None else "pending"}
    write_new(output / f"summary-{uuid4().hex}.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    parser.add_argument("--model", default="gpt-5.4-nano-2026-03-17")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS,
                        default=PROMPT_VERSION)
    parser.add_argument("--tag-llm", action="store_true", required=True)
    criteria = parser.add_mutually_exclusive_group()
    criteria.add_argument("--human-criteria", type=Path, help="Frozen human criteria JSON")
    criteria.add_argument("--ai-proxy-criteria", type=Path, help="User-authorized AI proxy diagnostic; not human ground truth")
    args = parser.parse_args(argv)
    if args.prompt_version in (V4_PROMPT_VERSION, DEDUP_PROMPT_VERSION, V5_PROMPT_VERSION) and args.human_criteria is None and args.ai_proxy_criteria is None:
        parser.error("v4/v5 requires --human-criteria fixed by human reviewers or explicit --ai-proxy-criteria")
    load_dotenv()
    result = run_batch(OpenAI(timeout=60, max_retries=0), read(args.inputs),
                       read(args.taxonomy), args.model, args.output, args.max_attempts,
                       prompt_version=args.prompt_version,
                       human_criteria=read(args.human_criteria) if args.human_criteria else None,
                       ai_proxy_criteria=read(args.ai_proxy_criteria) if args.ai_proxy_criteria else None)
    print(json.dumps({k: result[k] for k in ("total", "first_pass", "passed_after_retries")}))
    return 0 if result["passed_after_retries"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
