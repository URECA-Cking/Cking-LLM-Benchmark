from src.data import dev_creators, load_categories, load_creators, query_creators
from src.data import test_creators as get_test_split  # pytest가 test_로 시작하는 임포트를 테스트로 오인하지 않게 alias


def test_load_categories_returns_ten_with_descriptions() -> None:
    categories = load_categories()

    assert len(categories) == 10
    assert all(c.description.strip() for c in categories)
    assert {c.code for c in categories} == {
        "FITNESS", "FOOD", "GAME", "BEAUTY", "FASHION", "TRAVEL", "MUSIC", "PET", "TECH", "KNOWLEDGE",
    }


def test_load_creators_matches_confirmed_split_sizes() -> None:
    creators = load_creators()

    assert len(creators) == 100
    assert len(dev_creators(creators)) == 30
    assert len(get_test_split(creators)) == 70
    assert len(query_creators(creators)) == 30
    assert all(c.split == "test" for c in query_creators(creators))


def test_creator_input_text_falls_back_to_events_when_bio_empty() -> None:
    creators = {c.id: c for c in load_creators()}

    x10 = creators["X10"]
    assert x10.bio.strip() == ""
    assert "캠핑" in x10.input_text() or "차박" in x10.input_text()


def test_declared_defaults_to_gold_when_not_overridden() -> None:
    creators = {c.id: c for c in load_creators()}

    assert creators["F01"].declared == creators["F01"].gold
    assert creators["X03"].declared == ("BEAUTY",)
    assert creators["X03"].gold == ("FITNESS",)


def test_load_large_creators_reads_pinned_file_as_dev_pool() -> None:
    from src.data import load_large_creators

    large = load_large_creators()

    assert len(large) == 300
    assert {c.split for c in large} == {"dev"} and not any(c.is_query for c in large)


def test_load_large_creators_rejects_missing_file_and_hash_mismatch(tmp_path) -> None:
    import pytest

    from src.data import load_large_creators

    with pytest.raises(FileNotFoundError):
        load_large_creators(tmp_path / "none.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("id,name,bio,events,subtopic,gold,written_by,note,declared\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_large_creators(bad, expected_sha256="0" * 64)
