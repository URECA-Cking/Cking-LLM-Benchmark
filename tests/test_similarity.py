import numpy as np
import pytest

from src.similarity import (
    cosine_matrix,
    cosine_with_tag_bonus,
    jaccard,
    jaccard_matrix,
    precision_at_k_by_gold,
    select_bonus,
    top_n,
)


def test_jaccard_basic_cases() -> None:
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(frozenset({"A"}), frozenset({"A"})) == 1.0
    assert jaccard(frozenset({"A", "B"}), frozenset({"B", "C"})) == 1 / 3


def test_jaccard_matrix_masks_diagonal() -> None:
    tags = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"})]
    matrix = jaccard_matrix(tags)

    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0
    assert not np.isfinite(matrix[0, 0])


def test_cosine_matrix_of_normalized_vectors() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix = cosine_matrix(vectors)

    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0
    assert not np.isfinite(matrix[1, 1])


def test_cosine_with_tag_bonus_adds_only_when_sharing_tag() -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    cosine = cosine_matrix(vectors)
    tags = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"})]

    combined = cosine_with_tag_bonus(cosine, tags, bonus=0.5)

    assert combined[0, 1] == cosine[0, 1] + 0.5
    assert combined[0, 2] == cosine[0, 2]
    assert not np.isfinite(combined[0, 0])


def test_top_n_excludes_self_and_breaks_ties_by_id() -> None:
    ids = ["b", "a", "c"]
    matrix = np.array(
        [
            [-np.inf, 0.5, 0.5],
            [0.5, -np.inf, 0.1],
            [0.5, 0.1, -np.inf],
        ],
        dtype=np.float32,
    )

    result = top_n(matrix, ids, row_index=0, n=2)

    assert result == [("a", 0.5), ("c", 0.5)]


def _four_creator_setup() -> tuple[np.ndarray, list[str], dict[str, frozenset[str]]]:
    # a,b는 임베딩이 가깝고 실제 gold도 같은 분야다. c는 임베딩이 멀지만 tag는 a와 겹친다(잡음).
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    ids = ["a", "b", "c", "d"]
    gold = {"a": frozenset({"FITNESS"}), "b": frozenset({"FITNESS"}), "c": frozenset({"GAME"}), "d": frozenset({"FOOD"})}
    return vectors, ids, gold


def test_precision_at_k_by_gold_counts_shared_gold_in_top_k() -> None:
    vectors, ids, gold = _four_creator_setup()
    cosine = cosine_matrix(vectors)

    # a 기준 top-3: b(gold 공유), d, c 순 -> 3개 중 1개만 gold 공유
    score = precision_at_k_by_gold(cosine, ids, gold, query_ids=["a"], k=3)

    assert score == pytest.approx(1 / 3)


def test_select_bonus_prefers_bonus_that_improves_gold_precision() -> None:
    vectors, ids, gold = _four_creator_setup()
    cosine = cosine_matrix(vectors)
    # c의 잡음 태그가 a와 겹쳐 순위를 갉아먹는 상황. bonus는 gold와 무관한 태그이므로
    # 여기서는 bonus를 키워도 precision이 나빠지거나 그대로여야 한다 (0을 골라야 함).
    tag_sets = [frozenset({"X"}), frozenset(), frozenset({"X"}), frozenset()]

    bonus = select_bonus(cosine, ids, tag_sets, gold, dev_ids=["a"], candidate_bonuses=[0.0, 0.5, 1.0], k=1)

    assert bonus == 0.0

