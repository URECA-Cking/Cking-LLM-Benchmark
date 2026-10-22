"""Synthetic handoff tests do not establish real-model semantic improvement."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.detail_tagging import (PROMPT_VERSION, PROMPT_VERSIONS, V4_PROMPT_VERSION,
                                V5_PROMPT_VERSION, DEDUP_PROMPT_VERSION, automatic_record,
                                build_prompt, digest)
from src.detail_tagging_batch import read, run_batch, main as batch_main
from src.detail_tagging_experiment import compare_experiments, reference_metrics
from src.detail_tagging_human_eval import prepare, freeze
from tests.test_detail_tagging_human_eval import judged, write_csv


@pytest.fixture
def taxonomy():
    return read("data/creator-subtopics.json")


def candidate(code, text, parent=None):
    row = dict(code=code, evidence=text, core_evidence="", repetition_evidence=[])
    if parent:
        row["parent"] = parent
    return row


def client(outputs):
    def create(**kwargs):
        data = outputs.pop(0)
        message = dict(content=json.dumps(data), refusal=None)
        response = dict(model=kwargs["model"], choices=[dict(finish_reason="stop", message=message)],
                        usage=dict(prompt_tokens=8, completion_tokens=2, total_tokens=10,
                                   prompt_tokens_details=dict(cached_tokens=0)))
        return SimpleNamespace(model_dump=lambda: response,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(**message))])
    return SimpleNamespace(timeout=60, max_retries=0,
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_frozen_old_prompt_bytes_and_distinct_ablations(taxonomy):
    expected = ["75c0d36b571201cf8b7245441f652f50ab28c689476181a8ab1a222a73337386",
                "85cea67bca072080a1b53378a2374cdcb71ad9741477c6190756cd0311c7a19a",
                "6d112d57826a0031320e69ddb117bf4d3410e6c124617067dd33556b3d4c63ac"]
    assert [digest(build_prompt(taxonomy, v)) for v in PROMPT_VERSIONS[:3]] == expected
    assert PROMPT_VERSION == "grounded-detail-candidates-v2"
    dedup = build_prompt(taxonomy, DEDUP_PROMPT_VERSION)
    v5 = build_prompt(taxonomy, V5_PROMPT_VERSION)
    assert "출력 단위는 근거 개수가 아니라 코드" in dedup and "의미 판단과 인용 형식" not in dedup
    assert "출력 단위는 근거 개수가 아니라 코드" in v5 and "의미 판단과 인용 형식" in v5
    assert "core_evidence가 없다는 이유" in v5 and "사람 합의나 독립 검증 결과가 아니다" in v5


@pytest.mark.parametrize("text,parent,topics", [
    ("퇴근 후 내 하루를 기록해요", "LIFE", ["LIFE_PERSONAL"]),
    ("살면서 겪은 사연을 들려줘요", "LIFE", ["LIFE_STORIES"]),
    ("오늘의 옷 조합을 소개해요", "FASHION", ["FASHION_CLOTHING"]),
    ("취미로 천을 재단하고 바느질해요", "HOBBY", ["HOBBY_CRAFTS"]),
    ("내가 만든 단편 영화 작품을 소개합니다", "ENTERTAIN", ["ENTERTAIN_FILM"]),
    ("#노래커버", "MUSIC", ["MUSIC_VOCAL"]),
    ("패션과 뷰티 채널", "FASHION", []),
])
def test_preserved_candidates_do_not_require_core_or_literal_tag_name(taxonomy, text, parent, topics):
    data = {"parents": [candidate(parent, text)],
            "topics": [candidate(t, text, parent) for t in topics]}
    record = automatic_record("synthetic", text, data, taxonomy, "model", V5_PROMPT_VERSION)
    assert [t["code"] for t in record["result"]["topics"]] == topics
    assert all(t["priority"][0] == 0 for t in record["result"]["topics"])


def test_multiple_quotes_one_code_and_independent_rows(taxonomy):
    text = "오늘 피아노 연주\n내일 기타 연주"
    parent = candidate("MUSIC", "피아노 연주")
    topic = candidate("MUSIC_INSTRUMENT", "피아노 연주", "MUSIC")
    topic["repetition_evidence"] = ["피아노 연주", "기타 연주"]
    record = automatic_record("synthetic", text, dict(parents=[parent], topics=[topic]), taxonomy, "m", V5_PROMPT_VERSION)
    assert record["result"]["topics"][0]["priority"][1] == 2
    for axis in ("parents", "topics"):
        data = dict(parents=[parent], topics=[topic])
        data[axis] *= 2
        with pytest.raises(ValueError, match="Duplicate"):
            automatic_record("synthetic", text, data, taxonomy, "m", V5_PROMPT_VERSION)
    record = automatic_record("synthetic", text.replace("\n", " "), dict(parents=[parent], topics=[topic]), taxonomy, "m", V5_PROMPT_VERSION)
    assert record["result"]["topics"][0]["priority"][1] == 0
    assert record["result"]["topics"][0]["repetition_warnings"]


@pytest.mark.parametrize("quote", ["", "개인 일상", "나의...기록", '"나의 하루 기록"'])
def test_v5_does_not_repair_evidence(taxonomy, quote):
    with pytest.raises(ValueError, match="exact source quote"):
        automatic_record("synthetic", "나의 하루 기록", dict(parents=[candidate("LIFE", quote)], topics=[]),
                         taxonomy, "m", V5_PROMPT_VERSION)


def test_boundary_counterexamples_are_explicit(taxonomy):
    prompt = build_prompt(taxonomy, V5_PROMPT_VERSION)
    for instruction in ("단순 자기소개", "의류 소개만으로 직접 제작", "일반 영상 편집", "배경 음악만으로 작곡", "동물 소재만으로 직접 돌봄"):
        assert instruction in prompt
    # Validator does not invent semantic labels from an empty candidate response.
    record = automatic_record("synthetic", "회사원입니다. 배경 음악과 동물 영상 편집", dict(parents=[], topics=[]),
                              taxonomy, "m", V5_PROMPT_VERSION)
    assert record["result"]["unclassified"]


@pytest.mark.parametrize("version", [DEDUP_PROMPT_VERSION, V5_PROMPT_VERSION])
def test_cli_requires_frozen_reference(version, capsys):
    with pytest.raises(SystemExit) as error:
        batch_main(["--inputs", "unused", "--output", "unused", "--tag-llm", "--prompt-version", version])
    assert error.value.code == 2
    assert "requires --human-criteria" in capsys.readouterr().err


@pytest.fixture
def runs(tmp_path, taxonomy):
    pack = prepare([dict(channel_id="a", text="노래 커버"), dict(channel_id="b", text="노래 커버")], taxonomy)
    rows = judged(pack)
    for row in rows:
        row["reviewer"] = "AI:synthetic"
    csv = tmp_path / "review.csv"
    write_csv(csv, rows)
    frozen = freeze(pack, csv, "ai_proxy")
    good = dict(parents=[candidate("MUSIC", "노래 커버")], topics=[candidate("MUSIC_VOCAL", "노래 커버", "MUSIC")])
    bad = copy.deepcopy(good)
    bad["parents"] *= 2
    paths = []
    for version, outputs in [(PROMPT_VERSION, [good, bad]), (V4_PROMPT_VERSION, [good, good]),
                             (V5_PROMPT_VERSION, [bad, good])]:
        folder = tmp_path / version
        run_batch(client(outputs), pack["manifest"]["inputs"], taxonomy, "model", folder,
                  max_attempts=1, prompt_version=version, ai_proxy_criteria=frozen)
        paths.append(next(folder.glob("summary-*.json")))
    return frozen, paths


def test_pairwise_denominators_and_raw_response_audit(runs):
    frozen, paths = runs
    snapshots = {p: p.read_bytes() for path in paths for p in path.parent.rglob("*.json")}
    report = compare_experiments(frozen, paths)
    assert [pair["channels"] for pair in report["paired_success"]] == [1, 0, 1]
    assert report["human_review"] == "not_performed"
    assert report["ablation_effect"] == "not_established"
    audit = report["runs"][2]["audit"]
    assert audit["errors"]["Duplicate candidate code"] == dict(calls=1, channels=1)
    assert len(audit["final_failures"]) == 1
    assert audit["tokens"]["total_tokens"] == 20
    assert audit["usage_missing_attempts"] == 0
    assert all(p.read_bytes() == value for p, value in snapshots.items())


@pytest.mark.parametrize("fault", ["raw", "sequence", "reference", "duplicate_version"])
def test_forged_comparison_rejected(runs, fault):
    frozen, paths = runs
    path = paths[0]
    if fault == "duplicate_version":
        paths = [path, path]
    elif fault == "reference":
        frozen = copy.deepcopy(frozen)
        frozen["hash"] = "forged"
    else:
        entry_path = next((path.parent / "attempts").rglob("*.json"))
        entry = read(entry_path)
        if fault == "raw":
            # Pick the passed attempt so raw candidate/revision equality must be checked.
            entry_path = next(p for p in (path.parent / "attempts").rglob("*.json") if read(p)["status"] == "passed")
            entry = read(entry_path)
            entry["response"]["choices"][0]["message"]["content"] = '{"parents":[],"topics":[]}'
        else:
            entry["attempt"] = 2
        entry_path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ValueError):
        compare_experiments(frozen, paths)


@pytest.mark.parametrize("old", [PROMPT_VERSION, V4_PROMPT_VERSION, DEDUP_PROMPT_VERSION])
def test_v5_resume_cannot_mix_prompts(tmp_path, taxonomy, old):
    inputs = [dict(channel_id="a", text="")]
    run_batch(client([dict(parents=[], topics=[])]), inputs, taxonomy, "m", tmp_path, prompt_version=old)
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), inputs, taxonomy, "m", tmp_path, prompt_version=V5_PROMPT_VERSION)
