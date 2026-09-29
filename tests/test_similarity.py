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



def test_restricting_candidate_pool_to_dev_prevents_test_label_leakage() -> None:
    """cmd_select_params 수정 검증(리뷰 P1): 후보 pool에 test가 섞이면 test 크리에이터의
    gold만 바꿔도 선택되는 bonus가 달라진다(누수). pool을 dev만으로 제한하면 영향이 없다.

    d1(쿼리, dev)·d2(dev, d1과 태그 미공유)·t1(test, d1과 태그 공유). d1-d2 코사인이 더 높지만
    bonus=0.5면 d1-t1이 역전한다. d2는 gold를 절대 공유하지 않게 해, "bonus가 올려주는 후보(t1)의
    gold가 쿼리와 맞는가"만으로 승부가 갈리게 만든다.
    """
    vectors = np.array([[1.0, 0.0], [0.99, 0.0], [0.8, 0.6]], dtype=np.float32)  # d1, d2, t1
    tag_sets = [frozenset({"X"}), frozenset(), frozenset({"X"})]  # d1-t1만 태그 공유
    query_ids = ["d1"]
    candidate_bonuses = [0.0, 0.5]

    def leaky_selected_bonus(t1_gold: frozenset[str]) -> float:
        ids = ["d1", "d2", "t1"]
        gold_by_id = {"d1": frozenset({"A"}), "d2": frozenset({"C"}), "t1": t1_gold}
        cosine = cosine_matrix(vectors)
        return select_bonus(cosine, ids, tag_sets, gold_by_id, query_ids, candidate_bonuses, k=1)

    def dev_only_selected_bonus(t1_gold: frozenset[str]) -> float:
        # t1_gold는 dev-only pool에 아예 등장하지 않으므로 결과에 영향을 줄 수 없다.
        del t1_gold
        dev_index = [0, 1]
        ids = ["d1", "d2"]
        gold_by_id = {"d1": frozenset({"A"}), "d2": frozenset({"C"})}
        cosine = cosine_matrix(vectors[dev_index])
        dev_tag_sets = [tag_sets[i] for i in dev_index]
        return select_bonus(cosine, ids, dev_tag_sets, gold_by_id, query_ids, candidate_bonuses, k=1)

    leaky_when_t1_unrelated = leaky_selected_bonus(frozenset({"B"}))
    leaky_when_t1_relevant = leaky_selected_bonus(frozenset({"A"}))
    assert leaky_when_t1_unrelated != leaky_when_t1_relevant  # 누수 재현

    fixed_when_t1_unrelated = dev_only_selected_bonus(frozenset({"B"}))
    fixed_when_t1_relevant = dev_only_selected_bonus(frozenset({"A"}))
    assert fixed_when_t1_unrelated == fixed_when_t1_relevant  # 수정 후 test gold와 무관


def test_cosine_matrix_with_query_vectors_keeps_candidate_side_original() -> None:
    """쿼리 프롬프트 벡터는 행(쿼리 쪽)에만 쓰이고, 쿼리끼리 후보가 될 때도 열은 원래 벡터여야 한다."""
    vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    query_vectors = vectors.copy()
    query_vectors[0] = [0.0, 1.0]  # 0번 쿼리만 프롬프트 적용으로 방향이 바뀐 상황

    matrix = cosine_matrix(vectors, query_vectors=query_vectors)

    assert matrix[0, 1] == pytest.approx(0.8)  # 행 0은 쿼리 벡터로
    assert matrix[1, 0] == pytest.approx(0.6)  # 열 0(후보)은 원래 벡터 [1, 0]으로 — 덮어쓰면 0.8이 된다
    assert np.array_equal(cosine_matrix(vectors), cosine_matrix(vectors, query_vectors=None), equal_nan=True)


def test_select_cutoff_drops_unrelated_and_keeps_related() -> None:
    from src.similarity import select_cutoff

    ids = ["q", "a", "b", "c"]
    gold = {"q": frozenset({"X"}), "a": frozenset({"X"}), "b": frozenset({"Y"}), "c": frozenset({"Y"})}
    matrix = np.array(
        [[-np.inf, 0.9, 0.3, 0.2], [0.9, -np.inf, 0.1, 0.1], [0.3, 0.1, -np.inf, 0.5], [0.2, 0.1, 0.5, -np.inf]], dtype=np.float32
    )

    cutoff = select_cutoff(matrix, ids, gold, ["q"], k=3)

    assert cutoff == pytest.approx(0.9)  # 관련(a=0.9)만 남기고 무관(b=0.3, c=0.2)은 거른다


def test_select_cutoff_prefers_lowest_cutoff_on_tie() -> None:
    from src.similarity import select_cutoff

    ids = ["q", "a", "b"]
    gold = {"q": frozenset({"X"}), "a": frozenset({"X"}), "b": frozenset({"X"})}
    matrix = np.array([[-np.inf, 0.9, 0.4], [0.9, -np.inf, 0.1], [0.4, 0.1, -np.inf]], dtype=np.float32)

    assert select_cutoff(matrix, ids, gold, ["q"], k=2) == pytest.approx(0.4)  # 전부 관련이면 아무것도 안 거른다


def test_select_cutoff_can_drop_everything_when_all_candidates_are_unrelated() -> None:
    from src.similarity import select_cutoff

    ids = ["q", "a", "b"]
    gold = {"q": frozenset({"X"}), "a": frozenset({"Y"}), "b": frozenset({"Y"})}
    matrix = np.array([[-np.inf, 0.9, 0.8], [0.9, -np.inf, 0.1], [0.8, 0.1, -np.inf]], dtype=np.float32)

    cutoff = select_cutoff(matrix, ids, gold, ["q"], k=2)

    assert cutoff > float(np.float32(0.9))  # 관측된 최고 점수(float32 0.9)보다 커서 무관 후보 둘 다 거른다
