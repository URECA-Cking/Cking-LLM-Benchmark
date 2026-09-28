from src.judge import build_judge_pairs, shuffle_rows


def test_build_judge_pairs_deduplicates_across_methods() -> None:
    top5 = {
        "q1": {
            "M1": [("a", 0.9), ("b", 0.5)],
            "M2": [("a", 0.8), ("c", 0.4)],
        },
        "q2": {
            "M1": [("d", 0.7)],
        },
    }

    rows, provenance = build_judge_pairs(top5)

    pair_ids = {row.pair_id for row in rows}
    assert pair_ids == {"q1::a", "q1::b", "q1::c", "q2::d"}
    assert provenance["q1::a"] == ["M1", "M2"]
    assert provenance["q1::b"] == ["M1"]
    assert all(row.score == "" for row in rows)


def test_shuffle_rows_is_deterministic_for_same_seed() -> None:
    top5 = {"q1": {"M1": [(f"c{i}", 1.0) for i in range(5)]}}
    rows, _ = build_judge_pairs(top5)

    first = shuffle_rows(rows, seed=1)
    second = shuffle_rows(rows, seed=1)
    different = shuffle_rows(rows, seed=2)

    assert [r.pair_id for r in first] == [r.pair_id for r in second]
    assert {r.pair_id for r in first} == {r.pair_id for r in rows}
    assert [r.pair_id for r in first] != [r.pair_id for r in different] or len(rows) < 2
