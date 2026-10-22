import csv
import json
from types import SimpleNamespace

import pytest

from src import generate_taste as gt


def _result(**overrides):
    result = {"keyword": "홈트, 맨몸운동, 스트레칭", "sentence": "집에서 짧게 할 수 있는 운동 영상을 즐겨 봅니다.", "long": "퇴근 후 집에서 하는 운동 영상을 자주 봅니다. " * 6}
    result.update(overrides)
    return result


def _fake_client(responses):
    calls = {"n": 0}

    def create(**_kwargs):
        content = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_build_profiles_makes_two_single_and_one_cross_per_category() -> None:
    profiles = gt.build_profiles(["FITNESS", "FOOD"])

    assert [p.profile_id for p in profiles] == [f"P0{i}" for i in range(1, 7)]
    assert [p.variant for p in profiles[:3]] == ["single_a", "single_b", "cross"]
    assert profiles[0].gold == ("FITNESS",) and profiles[2].gold[0] == "FITNESS" and len(profiles[2].gold) == 2


def test_validate_flags_empty_hard_and_length_soft() -> None:
    assert gt.validate(_result()) == []
    assert gt.validate(_result(sentence=" "))[0].endswith("비어 있음")
    soft = gt.validate(_result(keyword="가" * 100))
    assert soft and all(e.startswith(gt.SOFT_PREFIX) for e in soft)


def test_to_rows_takes_gold_from_profile_and_writes_lf(tmp_path) -> None:
    profile = gt.build_profiles(["FITNESS"])[2]
    rows = gt.to_rows(profile, _result())
    path = tmp_path / "t.csv"
    gt.write_csv(rows, path)

    assert [r["style"] for r in rows] == list(gt.STYLES) and rows[0]["gold"] == "|".join(profile.gold)
    assert b"\r\n" not in path.read_bytes()
    assert list(csv.DictReader(path.open(encoding="utf-8")))[0]["id"] == "P03_keyword"


def test_generate_profile_retries_then_accepts_soft_only_and_rejects_hard() -> None:
    good = json.dumps(_result(), ensure_ascii=False)
    client, calls = _fake_client([json.dumps(_result(sentence="")), good])
    result, in_tokens, out_tokens = gt._generate_profile(client, "m", "p")
    assert calls["n"] == 2 and (in_tokens, out_tokens) == (20, 10) and result["sentence"]

    soft_client, soft_calls = _fake_client([json.dumps(_result(keyword="가" * 100), ensure_ascii=False)])
    gt._generate_profile(soft_client, "m", "p")
    assert soft_calls["n"] == gt.MAX_ATTEMPTS

    hard_client, _ = _fake_client([json.dumps(_result(sentence=""))])
    with pytest.raises(ValueError):
        gt._generate_profile(hard_client, "m", "p")


def test_estimate_cost_scales_with_calls() -> None:
    assert gt.estimate_cost("gpt-5.4-mini-2026-03-17", 30) > gt.estimate_cost("gpt-5.4-nano-2026-03-17", 30) > 0
    assert gt.estimate_cost("gpt-5.4-mini-2026-03-17", 0) == 0


def test_main_rejects_cache_made_with_another_model(tmp_path, monkeypatch) -> None:
    import sys

    monkeypatch.setattr(gt, "CACHE_DIR", tmp_path)
    (tmp_path / "taste_gen").mkdir()
    (tmp_path / "taste_gen" / "P01.json").write_text(json.dumps({"model": "gpt-5.4-nano-2026-03-17", "result": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_taste", "--model", "gpt-5.4-mini-2026-03-17", "--dry-run"])

    with pytest.raises(ValueError, match="캐시"):
        gt.main()


def test_load_taste_queries_reads_rows_and_checks_hash(tmp_path) -> None:
    from src.data import load_taste_queries

    path = tmp_path / "taste.csv"
    gt.write_csv(gt.to_rows(gt.build_profiles(["FITNESS"])[2], _result()), path)

    queries = load_taste_queries(path, expected_sha256=None)

    assert len(queries) == 3 and queries[0].gold == ("FITNESS", "FOOD")
    with pytest.raises(ValueError):
        load_taste_queries(path, expected_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        load_taste_queries(tmp_path / "none.csv")


def test_main_does_not_overwrite_existing_csv_without_force(tmp_path, monkeypatch, capsys) -> None:
    import sys

    existing = tmp_path / "taste.csv"
    existing.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(gt, "TASTE_QUERIES_CSV", existing)
    monkeypatch.setattr(gt, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_taste"])

    gt.main()

    assert existing.read_text(encoding="utf-8") == "keep" and "이미 있어" in capsys.readouterr().out


def _write_cache(path, prompt, result=None):
    path.write_text(json.dumps({"model": "m", "prompt_hash": gt._prompt_hash(prompt), "result": result or _result()}, ensure_ascii=False), encoding="utf-8")


def test_plan_reuses_cache_only_when_prompt_hash_matches(tmp_path) -> None:
    names = {"FITNESS": "운동", "FOOD": "요리"}
    profiles = gt.build_profiles(["FITNESS"])
    _write_cache(tmp_path / "P01.json", gt.build_prompt(names, profiles[0], ""))
    _write_cache(tmp_path / "P03.json", gt.build_prompt(names, profiles[2], ""))

    plan = gt._plan(profiles, names, tmp_path)

    assert plan[0][2] is not None  # 프롬프트가 같아 재사용
    assert plan[1][2] is None  # 캐시 없음
    assert plan[2][2] is not None  # 교차 프로필은 앞선 결과와 무관하게 재사용


def test_plan_regenerates_when_prompt_changed_or_hash_missing(tmp_path, monkeypatch) -> None:
    names = {"FITNESS": "운동", "FOOD": "요리"}
    profiles = gt.build_profiles(["FITNESS"])
    _write_cache(tmp_path / "P01.json", gt.build_prompt(names, profiles[0], ""))
    (tmp_path / "P03.json").write_text(json.dumps({"model": "m", "result": _result()}), encoding="utf-8")  # 해시 없는 예전 캐시

    monkeypatch.setitem(gt.STYLE_GUIDE, "sentence", "완전히 다른 길이 요구")  # 서비스 형식에 맞게 프롬프트를 바꾼 상황
    plan = gt._plan(profiles, names, tmp_path)

    assert plan[0][2] is None and plan[2][2] is None


def test_plan_regenerates_dependent_profile_when_previous_is_regenerated(tmp_path) -> None:
    names = {"FITNESS": "운동", "FOOD": "요리"}
    profiles = gt.build_profiles(["FITNESS"])
    first = _result(sentence="첫 사용자 문장이다.")
    _write_cache(tmp_path / "P02.json", gt.build_prompt(names, profiles[1], first["sentence"]))  # P01의 옛 결과를 avoid로 쓴 캐시

    plan = gt._plan(profiles, names, tmp_path)  # P01 캐시가 없어 새로 만들어지므로 P02의 avoid가 달라진다

    assert plan[0][2] is None and plan[1][2] is None
