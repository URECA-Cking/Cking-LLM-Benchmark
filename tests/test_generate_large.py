import csv
from types import SimpleNamespace

import pytest

from src import generate_large as gl


def _items(slots, **overrides):
    """슬롯 요구를 지키는 가짜 모델 응답을 만든다."""
    items = []
    for i, slot in enumerate(slots):
        bio = {"short_casual": "짧은 소개 ㅎㅎ", "medium": "한두 문장짜리 소개입니다. " * 3, "long_polished": "길게 쓴 소개 문장입니다. " * 12, "events_only": ""}[slot.style]
        events = "이벤트 제목 / 다른 이벤트" if slot.style == "events_only" else ""
        items.append({"name": f"닉네임{i}", "bio": bio, "events": events, "subtopic": f"주제{i}"})
    for index, patch in overrides.items():
        items[index].update(patch)
    return items


def test_build_slots_marks_cross_and_misdeclared_with_related_categories() -> None:
    slots = gl.build_slots("FITNESS", 0)

    assert len(slots) == gl.BATCH_SIZE
    assert slots[gl.CROSS_SLOT].gold == ("FITNESS", "FOOD") and slots[gl.CROSS_SLOT].declared == ()
    assert slots[gl.MISDECLARED_SLOT].gold == ("FITNESS",) and slots[gl.MISDECLARED_SLOT].declared == ("TRAVEL",)
    assert sum(1 for s in slots if s.kind == "plain") == gl.BATCH_SIZE - 2


def test_build_slots_never_declares_the_gold_category() -> None:
    for code, partners in gl.PARTNERS.items():
        assert code not in partners
        for batch in range(gl.BATCHES_PER_CATEGORY):
            for slot in gl.build_slots(code, batch):
                assert not (set(slot.declared) & set(slot.gold))


def test_validate_batch_accepts_conforming_items() -> None:
    slots = gl.build_slots("GAME", 1)

    assert gl.validate_batch(_items(slots), slots, set()) == []


def test_validate_batch_rejects_wrong_count_duplicate_name_and_style_violations() -> None:
    slots = gl.build_slots("GAME", 0)

    assert gl.validate_batch(_items(slots)[:-1], slots, set())  # 개수 부족
    assert gl.validate_batch(_items(slots), slots, {"닉네임0"})  # 이미 쓴 이름
    events_only = gl.STYLE_BY_SLOT.index("events_only")
    bad = _items(slots)
    bad[events_only]["bio"] = "이벤트만이어야 하는데 소개가 있음"
    assert gl.validate_batch(bad, slots, set())
    too_long = _items(slots)
    too_long[gl.STYLE_BY_SLOT.index("short_casual")]["bio"] = "가" * 200
    assert gl.validate_batch(too_long, slots, set())


def test_to_csv_row_takes_gold_and_declared_from_slot_not_model() -> None:
    slot = gl.build_slots("BEAUTY", 0)[gl.MISDECLARED_SLOT]
    row = gl.to_csv_row("L001", {"name": " 뷰티러 ", "bio": "소개", "events": "", "subtopic": "스킨케어"}, slot)

    assert row["gold"] == "BEAUTY" and row["declared"] == "FITNESS" and row["name"] == "뷰티러"
    assert row["written_by"] == "synthetic-large"


def test_write_csv_uses_lf_line_endings(tmp_path) -> None:
    path = tmp_path / "large.csv"
    gl.write_csv([gl.to_csv_row("L001", {"name": "a", "bio": "b", "events": "", "subtopic": "c"}, gl.build_slots("PET", 0)[0])], path)

    data = path.read_bytes()
    assert b"\r\n" not in data
    assert list(csv.DictReader(path.open(encoding="utf-8")))[0]["id"] == "L001"


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


def test_generate_batch_retries_on_invalid_response_and_sums_tokens() -> None:
    import json

    slots = gl.build_slots("TECH", 0)
    good = json.dumps({"creators": _items(slots)}, ensure_ascii=False)
    bad = json.dumps({"creators": _items(slots)[:3]}, ensure_ascii=False)
    client, calls = _fake_client([bad, good])

    items, in_tokens, out_tokens = gl._generate_batch(client, "m", "prompt", slots, set())

    assert len(items) == gl.BATCH_SIZE and calls["n"] == 2
    assert (in_tokens, out_tokens) == (20, 10)


def test_generate_batch_gives_up_after_max_attempts() -> None:
    import json

    slots = gl.build_slots("TECH", 0)
    client, calls = _fake_client([json.dumps({"creators": []})])

    with pytest.raises(ValueError):
        gl._generate_batch(client, "m", "prompt", slots, set())
    assert calls["n"] == gl.MAX_ATTEMPTS


def test_estimate_cost_scales_with_calls_and_model() -> None:
    assert gl.estimate_cost("gpt-5.4-mini-2026-03-17", 30) > gl.estimate_cost("gpt-5.4-nano-2026-03-17", 30) > 0
    assert gl.estimate_cost("gpt-5.4-mini-2026-03-17", 0) == 0


def test_generate_batch_accepts_length_only_violations_after_max_attempts() -> None:
    import json

    slots = gl.build_slots("TECH", 0)
    items = _items(slots)
    items[gl.STYLE_BY_SLOT.index("long_polished")]["bio"] = "짧은 긴글"  # 길이만 어김
    client, calls = _fake_client([json.dumps({"creators": items}, ensure_ascii=False)])

    result, _, _ = gl._generate_batch(client, "m", "prompt", slots, set())

    assert len(result) == gl.BATCH_SIZE and calls["n"] == gl.MAX_ATTEMPTS


def test_main_rejects_cache_made_with_another_model(tmp_path, monkeypatch) -> None:
    import json
    import sys

    monkeypatch.setattr(gl, "CACHE_DIR", tmp_path)
    (tmp_path / "large_gen").mkdir()
    (tmp_path / "large_gen" / "GAME_0.json").write_text(json.dumps({"model": "gpt-5.4-nano-2026-03-17", "items": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_large", "--model", "gpt-5.4-mini-2026-03-17", "--dry-run"])

    with pytest.raises(ValueError, match="캐시"):
        gl.main()


def test_validate_batch_rejects_subtopic_duplicated_within_category() -> None:
    slots = gl.build_slots("MUSIC", 0)
    items = _items(slots)
    items[1]["subtopic"] = " 주제 0 "  # 같은 배치 안 0번과 공백·대소문자만 다름

    assert any("subtopic" in e for e in gl.validate_batch(items, slots, set()))
    assert gl.validate_batch(_items(slots), slots, set(), {"다른주제"}) == []
    assert any("subtopic" in e for e in gl.validate_batch(_items(slots), slots, set(), {"주제0"}))


def test_main_reuses_cache_with_soft_length_violation_but_rejects_duplicate_subtopic(tmp_path, monkeypatch) -> None:
    import json
    import sys

    from src.data import load_categories

    monkeypatch.setattr(gl, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(gl, "CREATORS_LARGE_CSV", tmp_path / "large.csv")
    monkeypatch.setattr(gl, "OpenAI", lambda: None)
    monkeypatch.setattr(sys, "argv", ["generate_large", "--model", "gpt-5.4-mini-2026-03-17"])
    (tmp_path / "large_gen").mkdir()
    for c_index, category in enumerate(load_categories()):
        for batch in range(gl.BATCHES_PER_CATEGORY):
            items = _items(gl.build_slots(category.code, batch))
            for k, item in enumerate(items):
                item["name"] = f"이름{c_index}_{batch}_{k}"
                item["subtopic"] = f"주제{c_index}_{batch}_{k}"
            if (c_index, batch) == (0, 0):
                items[gl.STYLE_BY_SLOT.index("long_polished")]["bio"] = "짧은 긴글"  # 길이(소프트)만 위반
            (tmp_path / "large_gen" / f"{category.code}_{batch}.json").write_text(
                json.dumps({"model": "gpt-5.4-mini-2026-03-17", "items": items}, ensure_ascii=False), encoding="utf-8"
            )

    gl.main()  # 소프트 위반 캐시도 거부하지 않고 CSV까지 만든다
    assert (tmp_path / "large.csv").exists()

    dup = tmp_path / "large_gen" / f"{load_categories()[0].code}_1.json"
    data = json.loads(dup.read_text(encoding="utf-8"))
    data["items"][0]["subtopic"] = "주제0_0_0"
    dup.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="subtopic"):
        gl.main()
