"""실데이터 M2·M3·M4 비교 도구(src/real_eval.py)의 순수 로직과 단계 흐름을 가짜 데이터·가짜 클라이언트로 검증한다."""

import argparse
import csv
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from src import real_eval as re_
from src.clients.openai_judge import BINARY_SCHEMA, OpenAIJudge
from src.clients.openai_tagger import OpenAITagger
from src.real_eval import Channel
from src.similarity import cosine_matrix, cosine_with_tag_bonus, top_n

CODES = ["C0", "C1", "C2", "C3"]


def ch(i: int, bio: str = "긴 소개글입니다. 열다섯 글자를 넘습니다.", name: str = "이름") -> Channel:
    """테스트용 채널을 만든다."""
    return Channel(f"id{i:03d}", f"{name}{i}", bio)


# ---- 분할·쿼리 ---------------------------------------------------------------------------------------------------------


def test_split_dev_test_is_stratified_disjoint_and_reproducible():
    gold = {f"a{i}": "X" for i in range(10)} | {f"b{i}": "Y" for i in range(20)}
    dev1, test1 = re_.split_dev_test(gold, 0.3, seed=1)
    dev2, test2 = re_.split_dev_test(gold, 0.3, seed=1)
    assert (dev1, test1) == (dev2, test2)
    assert not set(dev1) & set(test1) and set(dev1) | set(test1) == set(gold)
    assert sum(1 for c in dev1 if gold[c] == "X") == 3 and sum(1 for c in dev1 if gold[c] == "Y") == 6


def test_split_keeps_at_least_one_test_per_category():
    gold = {"a1": "X", "a2": "X", "solo": "Y"}
    dev, test = re_.split_dev_test(gold, 0.9, seed=1)
    assert len([c for c in dev if gold[c] == "X"]) == 1 and "a1" in dev + test and "a2" in dev + test
    assert "solo" in test and "solo" not in dev


def test_select_queries_excludes_dev_and_separates_short_bios():
    pool = [ch(i) for i in range(20)] + [Channel(f"s{i}", f"짧은{i}", "") for i in range(10)]
    queries = re_.select_queries(pool, {"id000", "s0"}, n_regular=5, n_short=4, seed=7)
    assert len(queries["regular"]) == 5 and len(queries["short"]) == 4
    assert "id000" not in queries["regular"] and "s0" not in queries["short"]
    assert all(q.startswith("id") for q in queries["regular"]) and all(q.startswith("s") for q in queries["short"])
    assert queries == re_.select_queries(pool, {"id000", "s0"}, 5, 4, 7)


def test_select_queries_caps_at_available():
    pool = [ch(i) for i in range(3)]
    assert len(re_.select_queries(pool, set(), 100, 100, 1)["regular"]) == 3


def test_channel_text_falls_back_to_name_when_bio_empty():
    assert Channel("x", "채널명", "  ").text() == "채널명" and Channel("x", "채널명", "소개").text() == "소개"
    assert Channel("x", "n", "짧음").is_short() and not ch(1).is_short()


# ---- 후보 계산 ---------------------------------------------------------------------------------------------------------


def _random_setup(n: int = 30, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    tag_sets = [frozenset(c for c in CODES if rng.random() < 0.4) for _ in range(n)]
    return vectors, tag_sets, [f"id{i:03d}" for i in range(n)]


def test_tag_matrix_marks_tags_and_ignores_unknown():
    matrix = re_.tag_matrix([frozenset({"C1"}), frozenset(), frozenset({"C0", "ZZZ"})], CODES)
    assert matrix.tolist() == [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]


def test_top_candidates_matches_slow_reference_implementation():
    vectors, tag_sets, ids = _random_setup()
    bonus, k = 0.2, 5
    reference = cosine_with_tag_bonus(cosine_matrix(vectors), tag_sets, bonus)
    query_index = [0, 7, 19]
    got = re_.top_candidates(vectors, re_.tag_matrix(tag_sets, CODES), query_index, bonus, k, ids)
    for qi in query_index:
        expected = top_n(reference, ids, qi, k)
        assert [c for c, _ in got[ids[qi]]] == [c for c, _ in expected]
        assert [s for _, s in got[ids[qi]]] == pytest.approx([s for _, s in expected], abs=1e-5)


def test_top_candidates_without_bonus_is_plain_cosine_and_excludes_self():
    vectors, _, ids = _random_setup()
    got = re_.top_candidates(vectors, None, [3], 0.0, 4, ids)
    assert ids[3] not in [c for c, _ in got[ids[3]]]
    expected = top_n(cosine_matrix(vectors), ids, 3, 4)
    assert [c for c, _ in got[ids[3]]] == [c for c, _ in expected]


def test_top_candidates_ties_are_stable_by_pool_order():
    vectors = np.ones((5, 2), dtype=np.float32)
    ids = ["a", "b", "c", "d", "e"]
    got = re_.top_candidates(vectors, None, [2], 0.0, 3, ids)
    assert [c for c, _ in got["c"]] == ["a", "b", "d"]


def test_tag_system_prompt_lists_every_category_with_description():
    prompt = re_.tag_system_prompt([("FOOD", "요리·푸드", "레시피, 먹방"), ("GAME", "게임", "공략, 리뷰")], 3)
    assert "FOOD(요리·푸드): 레시피, 먹방" in prompt and "GAME(게임): 공략, 리뷰" in prompt and "최대 3개" in prompt


def test_compute_candidates_uses_llm_tags_for_m4_and_zero_shot_tags_for_m3():
    """q와 코사인이 같은 후보 c1·c2 중 zero-shot 태그는 둘 다 q와 겹치고, LLM 태그는 c2만 겹친다. M3는 동점(풀 순서 c1), M4는 c2가 1위여야 한다."""
    pool = [Channel(i, i, "소개글 열다섯 글자를 넘게 씁니다") for i in ("q", "c1", "c2", "c3")]
    vectors = np.array([[1, 0], [0.8, 0.6], [0.8, -0.6], [0, 1]], dtype=np.float32)
    category_vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    llm_tags = {"q": frozenset({"C2"}), "c1": frozenset({"C3"}), "c2": frozenset({"C2"}), "c3": frozenset()}
    params = {"tau": 0.7, "bonus_m3": 0.5, "bonus_m4": 0.3}
    got = re_.compute_candidates(pool, vectors, category_vectors, CODES, params, llm_tags, ["q"], k=3, max_tags=1)
    assert got["M2"]["q"][0][0] == "c1" and got["M3"]["q"][0][0] == "c1"
    assert got["M3"]["q"][0][1] == pytest.approx(0.8 + 0.5, abs=1e-5)  # M3는 bonus_m3
    assert got["M4"]["q"][0][0] == "c2" and got["M4"]["q"][0][1] == pytest.approx(0.8 + 0.3, abs=1e-5)  # M4는 bonus_m4


def test_precision_at_k_averages_over_available_candidates_when_fewer_than_k():
    assert re_.precision_at_k(["a", "b"], "q", {("q", "a"): 1, ("q", "b"): 0}, 5) == pytest.approx(0.5)


def test_select_parameters_ignores_test_channels():
    """tau·bonus는 dev만으로 정해져야 한다. test 채널의 벡터·라벨을 바꿔도 결과가 같아야 한다(과거 dev/test 라벨 누수 회귀)."""
    rng = np.random.default_rng(5)
    pool = [Channel(f"id{i:03d}", "n", "소개글 열다섯 글자를 넘게 씁니다") for i in range(40)]
    category_vectors = np.eye(4, 8, dtype=np.float32)
    base = np.array([category_vectors[i % 4] * 2 + 0.5 * rng.normal(size=8) for i in range(40)], dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    gold = {c.id: CODES[i % 4] for i, c in enumerate(pool)}
    dev_ids = [c.id for c in pool[:20]]
    llm = {c.id: frozenset({CODES[i % 4]}) for i, c in enumerate(pool)}
    first = re_.select_parameters(pool, base, category_vectors, CODES, gold, dev_ids, llm, 1, [0.0, 0.1, 0.3])

    other = base.copy()
    other[20:] = rng.normal(size=(20, 8))
    other[20:] /= np.linalg.norm(other[20:], axis=1, keepdims=True)
    scrambled = {**gold, **{c.id: CODES[(i + 1) % 4] for i, c in enumerate(pool[20:], start=20)}}
    llm_scrambled = {**llm, **{c.id: frozenset({"C3"}) for c in pool[20:]}}
    assert re_.select_parameters(pool, other, category_vectors, CODES, scrambled, dev_ids, llm_scrambled, 1, [0.0, 0.1, 0.3]) == first
    assert set(first) == {"tau", "bonus_m3", "bonus_m4"} and first["bonus_m3"] in (0.0, 0.1, 0.3)


# ---- 동시 실행·재개 ----------------------------------------------------------------------------------------------------


def test_run_concurrent_preserves_order_and_retries_transient_errors():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if x == 2 and calls["n"] < 4:
            raise RuntimeError("일시 오류")
        return x * 10

    assert re_.run_concurrent(flaky, [1, 2, 3], workers=1, retries=3, backoff=0) == [10, 20, 30]


def test_run_concurrent_raises_after_retries():
    with pytest.raises(RuntimeError):
        re_.run_concurrent(lambda x: (_ for _ in ()).throw(RuntimeError("항상 실패")), [1], workers=1, retries=2, backoff=0)


def test_run_concurrent_spaces_call_starts_by_min_interval_across_threads():
    starts: list[float] = []
    lock = __import__("threading").Lock()

    def fn(x):
        with lock:
            starts.append(__import__("time").monotonic())
        return x

    assert re_.run_concurrent(fn, list(range(8)), workers=4, min_interval=0.05) == list(range(8))
    starts.sort()
    assert all(b - a >= 0.04 for a, b in zip(starts, starts[1:]))  # 스레드가 여럿이어도 시작 간격이 지켜진다


def test_run_concurrent_default_retries_survive_five_consecutive_failures(monkeypatch):
    monkeypatch.setattr(re_.time, "sleep", lambda s: None)  # 대기 시간만 건너뛴다
    calls = {"n": 0}

    def rate_limited(x):
        calls["n"] += 1
        if calls["n"] <= 5:
            raise RuntimeError("429 호출 한도 초과")
        return x

    assert re_.run_concurrent(rate_limited, [7], workers=1) == [7] and calls["n"] == 6


def test_run_concurrent_backoff_grows_and_is_capped(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(re_.time, "sleep", waits.append)
    with pytest.raises(RuntimeError):
        re_.run_concurrent(lambda x: (_ for _ in ()).throw(RuntimeError("실패")), [1], workers=1, retries=8, backoff=2.0)
    assert waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]  # 2^n배로 늘다가 60초에서 멈춘다


def test_tag_pool_calls_only_missing_and_resets_on_config_change():
    channels = [ch(i, bio=f"소개글 열다섯 글자 넘게 {i}") for i in range(4)]
    seen: list[str] = []

    def tag_fn(text):
        seen.append(text)
        return ("C1",)

    cache = re_.tag_pool(channels[:2], tag_fn, {}, "cfg", workers=1)
    assert len(seen) == 2
    saves: list[dict] = []
    cache = re_.tag_pool(channels, tag_fn, cache, "cfg", workers=1, save=saves.append, chunk=1)
    assert len(seen) == 4 and len(saves) == 2  # 새 2명만 호출, 1개씩 저장
    re_.tag_pool(channels, tag_fn, cache, "cfg2", workers=1)
    assert len(seen) == 8  # 설정이 바뀌면 전부 다시 태깅


def test_tag_pool_retags_when_text_changes():
    channels = [ch(1, bio="원래 소개글 열다섯 글자 넘게")]
    calls: list[str] = []
    cache = re_.tag_pool(channels, lambda t: calls.append(t) or ("C0",), {}, "cfg", workers=1)
    changed = [Channel(channels[0].id, "n", "바뀐 소개글 열다섯 글자 넘게 쓴다")]
    re_.tag_pool(changed, lambda t: calls.append(t) or ("C0",), cache, "cfg", workers=1)
    assert len(calls) == 2


def test_judge_pairs_resumes_and_rejudges_when_text_or_condition_changes():
    text_of = {"q": "쿼리 소개", "c1": "후보1 소개", "c2": "후보2 소개"}
    calls: list[tuple[str, str]] = []

    def judge_fn(q, c):
        calls.append((q, c))
        return 1

    saved = re_.judge_pairs([("q", "c1")], text_of, judge_fn, {}, "j1", workers=1)
    assert len(calls) == 1 and saved["q::c1"]["score"] == 1
    saved = re_.judge_pairs([("q", "c1"), ("q", "c2")], text_of, judge_fn, saved, "j1", workers=1)
    assert len(calls) == 2  # 이미 한 쌍은 건너뜀
    re_.judge_pairs([("q", "c1")], {**text_of, "c1": "바뀐 후보 소개"}, judge_fn, saved, "j1", workers=1)
    assert len(calls) == 3  # 텍스트가 바뀌면 다시
    re_.judge_pairs([("q", "c1")], text_of, judge_fn, saved, "j2", workers=1)
    assert len(calls) == 4  # 판정 조건이 바뀌면 다시


# ---- 쌍·지표·결정 ------------------------------------------------------------------------------------------------------


def test_build_pairs_is_union_with_provenance():
    candidates = {
        "M2": {"q1": [("a", 0.9), ("b", 0.8), ("z", 0.1)]},
        "M3": {"q1": [("a", 0.9), ("c", 0.7)]},
        "M4": {"q1": [("c", 0.7), ("d", 0.6)]},
    }
    pairs, provenance = re_.build_pairs(candidates, k=2)
    assert pairs == [("q1", "a"), ("q1", "b"), ("q1", "c"), ("q1", "d")]
    assert provenance["q1::a"] == ["M2", "M3"] and provenance["q1::c"] == ["M3", "M4"] and ("q1", "z") not in pairs


def test_precision_at_k_uses_top_k_only():
    judged = {("q", "a"): 1, ("q", "b"): 0, ("q", "c"): 1, ("q", "d"): 1}
    assert re_.precision_at_k(["a", "b", "c", "d"], "q", judged, 3) == pytest.approx(2 / 3)
    assert re_.precision_at_k([], "q", judged, 3) == 0.0


@pytest.mark.parametrize(
    ("mean", "lower", "expected"),
    [(0.05, 0.001, "M4"), (0.049, 0.02, "M3"), (0.10, 0.0, "M3"), (0.10, -0.01, "M3"), (-0.02, -0.05, "M3"), (0.20, 0.10, "M4")],
)
def test_decide_follows_preregistered_rule(mean, lower, expected):
    assert re_.decide(mean, lower) == expected


def _scored_fixture(m4_better: bool):
    """일반 30개 쿼리, 짧은 소개글 10개 쿼리의 가짜 후보와 판정을 만든다. m4_better면 M4만 모든 쿼리에서 후보 5개 중 4개가 통과한다."""
    candidates, judged = {"M2": {}, "M3": {}, "M4": {}}, {}
    groups = {"regular": [f"r{i}" for i in range(30)], "short": [f"s{i}" for i in range(10)]}
    for q in groups["regular"] + groups["short"]:
        for method in candidates:
            good = 4 if (method == "M4" and m4_better) else 2
            ranked = [(f"{method}-{q}-{j}", 1.0) for j in range(5)]
            candidates[method][q] = ranked
            for j, (cid, _) in enumerate(ranked):
                judged[(q, cid)] = 1 if j < good else 0
    return candidates, judged, groups


def test_summarize_scores_adopts_m4_only_when_gain_is_large_and_consistent():
    candidates, judged, groups = _scored_fixture(m4_better=True)
    summary = re_.summarize_scores(candidates, judged, groups, k=5)
    regular = summary["groups"]["regular"]
    assert regular["n"] == 30 and regular["precision"]["M3"] == pytest.approx(0.4) and regular["precision"]["M4"] == pytest.approx(0.8)
    assert regular["m4_minus_m3"]["wins"] == 30 and regular["m4_minus_m3"]["ci_lower"] > 0
    assert summary["adopted"] == "M4"


def test_summarize_scores_keeps_m3_when_methods_tie():
    candidates, judged, groups = _scored_fixture(m4_better=False)
    summary = re_.summarize_scores(candidates, judged, groups, k=5)
    assert summary["groups"]["regular"]["m4_minus_m3"]["ties"] == 30 and summary["adopted"] == "M3"


def test_summarize_scores_decision_uses_regular_group_only():
    candidates, judged, groups = _scored_fixture(m4_better=False)
    for q in groups["short"]:  # 짧은 소개글에서만 M4가 좋아도 채택 결정은 일반 채널 기준
        for j in range(5):
            judged[(q, f"M4-{q}-{j}")] = 1
    summary = re_.summarize_scores(candidates, judged, groups, k=5)
    assert summary["groups"]["short"]["decision"] == "M4" and summary["adopted"] == "M3"


# ---- 사람 채점 일치율 --------------------------------------------------------------------------------------------------


def test_agreement_stats_perfect_and_disagreement():
    perfect = re_.agreement_stats({"a": 1, "b": 0, "c": 1}, {"a": 1, "b": 0, "c": 1})
    assert perfect["agree_rate"] == 1.0 and perfect["kappa"] == pytest.approx(1.0)
    mixed = re_.agreement_stats({"a": 1, "b": 0, "c": 1, "d": 0}, {"a": 1, "b": 1, "c": 1, "d": 0})
    assert mixed["agree_rate"] == 0.75 and mixed["human_positive_rate"] == 0.5 and mixed["auto_positive_rate"] == 0.75
    assert re_.agreement_stats({}, {"a": 1})["n"] == 0


def test_sample_human_pairs_is_deterministic_and_capped():
    ids = [f"q{i}::c{i}" for i in range(50)]
    a = re_.sample_human_pairs(ids, 10, seed=3)
    assert a == re_.sample_human_pairs(list(reversed(ids)), 10, seed=3) and len(a) == 10
    assert len(re_.sample_human_pairs(ids, 999, seed=3)) == 50


# ---- 클라이언트 확장 ---------------------------------------------------------------------------------------------------


class _FakeChat:
    """chat.completions.create 호출 인자를 기록하고 정해진 JSON을 돌려주는 가짜 OpenAI 클라이언트다."""

    def __init__(self, content: str):
        self.kwargs: dict = {}
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )


def test_binary_judge_sends_binary_schema_and_custom_prompt():
    fake = _FakeChat('{"score": 1}')
    judge = OpenAIJudge(model="m", client=fake, system_prompt=re_.BINARY_SYSTEM_PROMPT, schema=BINARY_SCHEMA)
    assert judge.judge("쿼리", "후보").score == 1
    sent = fake.kwargs["response_format"]["json_schema"]["schema"]
    assert sent["properties"]["score"]["enum"] == [0, 1]
    assert fake.kwargs["messages"][0]["content"] == re_.BINARY_SYSTEM_PROMPT


def test_default_judge_schema_is_unchanged():
    fake = _FakeChat('{"score": 2}')
    assert OpenAIJudge(model="m", client=fake).judge("q", "c").score == 2
    assert fake.kwargs["response_format"]["json_schema"]["schema"]["properties"]["score"]["enum"] == [0, 1, 2]


def test_tagger_uses_custom_system_prompt_and_keeps_default():
    fake = _FakeChat('{"tags": ["C1", "UNCLASSIFIED"]}')
    assert OpenAITagger(model="m", category_codes=CODES, client=fake, system_prompt="맞춤 프롬프트").tag("소개").tags == ("C1",)
    assert fake.kwargs["messages"][0]["content"] == "맞춤 프롬프트"
    OpenAITagger(model="m", category_codes=CODES, client=fake).tag("소개")
    assert "분류기" in fake.kwargs["messages"][0]["content"] and "맞춤" not in fake.kwargs["messages"][0]["content"]


# ---- 데이터 로더 -------------------------------------------------------------------------------------------------------


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_data(root, n_per_cat: int = 12, bad: str | None = None):
    """카테고리 4개 x n_per_cat개 정답 채널 + 정답 없는 채널·짧은 소개글 채널이 섞인 가짜 외부 데이터 폴더를 만든다."""
    pool, ref, gold = [], [], []
    number = 0
    for cat in range(4):
        for i in range(n_per_cat):
            number += 1
            cid = f"c{cat}_{i}"
            pool.append([cid, f"이름{cid}", f"cat{cat} 소개글 열다섯 글자를 넘는 내용 {i}", "q"])
            ref.append([f"C{number:03d}", cid, "", "", "", ""])
            gold.append([f"C{number:03d}", "", f"C{cat}", "audit", 0, 1])
    for i in range(20):  # 정답 없는 후보 풀 채널
        pool.append([f"u{i}", f"무라벨{i}", f"cat{i % 4} 소개글 열다섯 글자를 넘는 내용 u{i}", "q"])
    for i in range(6):  # 짧은 소개글 채널
        pool.append([f"s{i}", f"cat{i % 4} 짧은이름{i}", "", "q"])
    if bad == "missing_pool":
        gold.append(["C999", "", "C0", "audit", 0, 1])
        ref.append(["C999", "not_in_pool", "", "", "", ""])
    if bad == "unknown_category":
        gold[0][2] = "ZZZ"
    gold.append(["C998", "", "C0", "audit", 1, 0])  # use_gold=0은 읽지 않는다
    _write_csv(root / "raw" / "pool.csv", ["channel_id", "name", "bio", "source_query"], pool)
    _write_csv(root / "labels" / "reference_hidden.csv", ["번호", "channel_id", "handle", "source_query", "wiki_topics", "subscribers"], ref)
    _write_csv(root / "labels" / "gold_v2.csv", ["번호", "label_v1", "gold_v2", "verified_by", "low_evidence", "use_gold"], gold)
    _write_csv(root / "labels" / "categories_v2.csv", ["code", "name", "description"], [[f"C{i}", f"분야{i}", f"cat{i} 설명"] for i in range(4)])


def test_load_real_data_reads_pool_gold_and_categories(tmp_path):
    _write_data(tmp_path)
    data = re_.load_real_data(tmp_path)
    assert len(data.pool) == 48 + 20 + 6 and len(data.gold) == 48 and data.codes == CODES
    assert data.gold["c2_3"] == "C2"


@pytest.mark.parametrize("bad", ["missing_pool", "unknown_category"])
def test_load_real_data_rejects_inconsistent_labels(tmp_path, bad):
    _write_data(tmp_path, bad=bad)
    with pytest.raises(ValueError):
        re_.load_real_data(tmp_path)


def test_resolve_data_dir_uses_arg_then_env_and_rejects_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(re_.REAL_DATA_ENV, raising=False)
    with pytest.raises(ValueError):
        re_.resolve_data_dir(None)
    monkeypatch.setenv(re_.REAL_DATA_ENV, str(tmp_path))
    assert re_.resolve_data_dir(None) == tmp_path
    with pytest.raises(FileNotFoundError):
        re_.resolve_data_dir(str(tmp_path / "없는폴더"))


def test_loader_strips_nul_characters(tmp_path):
    _write_data(tmp_path)
    path = tmp_path / "raw" / "pool.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace("cat0 소개글", "cat0 소\x00개글", 1), encoding="utf-8-sig")
    assert all("\x00" not in c.bio for c in re_.load_real_data(tmp_path).pool)


# ---- 단계 전체 흐름 ----------------------------------------------------------------------------------------------------


class _FakeEmbedding:
    """텍스트 앞의 "cat<N>" 표시를 분야 신호로 삼는 결정적 가짜 임베딩이다."""

    dim = 8
    calls = 0

    def embed(self, texts):
        type(self).calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16))
            out[i] = 0.3 * rng.normal(size=self.dim)
            for cat in range(4):
                if f"cat{cat}" in text:
                    out[i, cat] += 3.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class _FakeTagger:
    """텍스트의 cat<N> 표시를 그대로 태그로 돌려주는 가짜 태거다."""

    def __init__(self, model, category_codes, system_prompt=None, **_):
        assert "C0(분야0): cat0 설명" in system_prompt

    def tag(self, text):
        tags = tuple(f"C{c}" for c in range(4) if f"cat{c}" in text)
        if re.search(r" u\d+$", text):  # 정답 없는 채널은 다른 분야로 태깅해 M3와 M4의 후보가 달라지게 한다
            tags = tuple(f"C{(int(tag[1]) + 1) % 4}" for tag in tags)
        return SimpleNamespace(tags=tags)


class _FakeJudge:
    """쿼리와 후보에 같은 cat<N> 표시가 있으면 1을 돌려주는 가짜 판정기다."""

    def __init__(self, model, system_prompt=None, schema=None, **_):
        assert schema == BINARY_SCHEMA and system_prompt == re_.BINARY_SYSTEM_PROMPT

    def judge(self, query, candidate):
        same = any(f"cat{c}" in query and f"cat{c}" in candidate for c in range(4))
        return SimpleNamespace(score=int(same))


def _args(data_dir, **kw):
    return argparse.Namespace(data_dir=str(data_dir), api=False, force=False, file=None, source="human", targeted=False, calibrated=False, rejudged=False, groups=None, out=None, **kw)


def test_full_stage_flow_with_fake_clients(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr(re_, "REAL_CONCURRENCY", 2)
    monkeypatch.setattr(re_, "REAL_TAG_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_JUDGE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "OpenAITagger", _FakeTagger)
    monkeypatch.setattr(re_, "OpenAIJudge", _FakeJudge)
    monkeypatch.setattr("src.pipeline._embedding_client", lambda key: _FakeEmbedding())
    args = _args(data_dir)

    re_.cmd_embed(args)
    first_calls = _FakeEmbedding.calls
    re_.cmd_embed(args)  # 캐시가 같으면 다시 임베딩하지 않는다
    assert _FakeEmbedding.calls == first_calls and "캐시" in capsys.readouterr().out

    re_.cmd_tag_llm(args)
    tags = json.loads((tmp_path / "real" / "tags_llm.json").read_text(encoding="utf-8"))["tags"]
    assert len(tags) == 74 and tags["s3"]["tags"] == ["C3"] and tags["c1_0"]["tags"] == ["C1"]

    re_.cmd_select_params(args)
    params = json.loads((tmp_path / "real" / "params.json").read_text(encoding="utf-8"))
    assert set(params["dev_ids"]) & set(params["test_ids"]) == set() and len(params["dev_ids"]) + len(params["test_ids"]) == 48
    assert params["tag_accuracy"]["zero_shot_top1"] == 1.0 and params["tag_accuracy"]["llm_hit_rate"] == 1.0
    assert not set(params["queries"]["regular"]) & set(params["dev_ids"])
    assert all(q.startswith("s") for q in params["queries"]["short"])

    re_.cmd_candidates(args)
    re_.cmd_judge_sheet(args)
    pairs = json.loads((tmp_path / "real" / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    assert pairs and all(q != c for q, c in pairs)

    re_.cmd_auto_judge(args)
    re_.cmd_human_sheet(args)
    sheet = tmp_path / "real" / "human_sheet.csv"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert 0 < len(rows) <= re_.REAL_HUMAN_SAMPLE and all("쿼리 소개" in r for r in rows)
    judgments = json.loads((tmp_path / "real" / "judgments.json").read_text(encoding="utf-8"))
    header = "채점(1=추천에 넣을 만함, 0=아님)"
    for r in rows:
        r[header] = str(judgments[r["pair_id"]]["score"])  # 사람이 LLM과 똑같이 채점했다고 가정
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    re_.cmd_human_agree(args)
    assert json.loads((tmp_path / "real" / "human_agree.json").read_text(encoding="utf-8"))["passed"] is True

    re_.cmd_score(args)
    score = json.loads((tmp_path / "real" / "score.json").read_text(encoding="utf-8"))
    assert score["adopted"] in ("M3", "M4") and set(score["groups"]) == {"regular", "short"}
    assert score["groups"]["regular"]["precision"]["M3"] > 0.9  # 분야 신호가 분명한 가짜 데이터라 태그 보정이 잘 맞는다
    assert "채택" in capsys.readouterr().out


def test_load_embeddings_rejects_stale_cache(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr("src.pipeline._embedding_client", lambda key: _FakeEmbedding())
    re_.cmd_embed(_args(data_dir))
    data = re_.load_real_data(data_dir)
    assert re_.load_embeddings(data)[0].shape == (74, 8)
    changed = re_.RealData([Channel(data.pool[0].id, "n", "바뀐 소개글 열다섯 글자 넘게 쓴다")] + data.pool[1:], data.gold, data.categories)
    with pytest.raises(RuntimeError, match="다시"):
        re_.load_embeddings(changed)


def test_score_refuses_when_judgments_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    (tmp_path / "candidates.json").write_text(json.dumps({m: {"q": [["a", 1.0]]} for m in re_.METHODS}), encoding="utf-8")
    (tmp_path / "judgments.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pairs.json").write_text(json.dumps({"pairs": [["q", "a"]], "provenance": {"q::a": ["M2", "M3", "M4"]}}), encoding="utf-8")
    (tmp_path / "params.json").write_text(json.dumps({"queries": {"regular": ["q"], "short": []}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="판정이 없는 쌍"):
        re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))


def test_human_agree_records_source_and_score_warns_when_not_human(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    (tmp_path / "judgments.json").write_text(json.dumps({"q::a": {"score": 1, "hash": "h1"}, "q::b": {"score": 0, "hash": "h2"}}), encoding="utf-8")
    (tmp_path / "pairs.json").write_text(json.dumps({"pairs": [["q", "a"], ["q", "b"]], "provenance": {"q::a": ["M4"], "q::b": ["M3"]}}), encoding="utf-8")
    header = "채점(1=추천에 넣을 만함, 0=아님)"
    sheet = tmp_path / "sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "쿼리 소개", "후보 소개", header, "메모"])
        writer.writerows([["q::a", "x", "y", "1", ""], ["q::b", "x", "y", "0", ""]])
    re_.cmd_human_agree(argparse.Namespace(file=str(sheet), source="claude", targeted=False))
    saved = json.loads((tmp_path / "human_agree.json").read_text(encoding="utf-8"))
    assert saved["source"] == "claude" and saved["passed"] is True
    (tmp_path / "candidates.json").write_text(json.dumps({m: {"q": [["a", 1.0], ["b", 0.5]]} for m in re_.METHODS}), encoding="utf-8")
    (tmp_path / "params.json").write_text(json.dumps({"queries": {"regular": ["q"], "short": []}}), encoding="utf-8")
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert "사람 채점이 아니라 'claude'" in capsys.readouterr().out


# ---- 결정을 가르는 쌍 · 보정 -------------------------------------------------------------------------------------------


def _provenance():
    """M4만(a1~a6), M3만(b1~b6), 셋 다(c1~c6) 뽑은 쌍의 가짜 기록이다."""
    prov = {}
    for i in range(1, 7):
        prov[f"q::a{i}"], prov[f"q::b{i}"], prov[f"q::c{i}"] = ["M4"], ["M3"], ["M2", "M3", "M4"]
    return prov


def test_sample_targeted_pairs_covers_every_group_that_changes_m4_minus_m3_and_shuffles():
    prov = _provenance() | {f"q::d{i}": ["M2", "M4"] for i in range(1, 4)} | {f"q::e{i}": ["M2", "M3"] for i in range(1, 4)}
    chosen = re_.sample_targeted_pairs(prov, 2, seed=1)
    assert len(chosen) == 8 and len(set(chosen)) == 8
    assert sorted(re_.group_of(prov[k]) for k in chosen) == ["M2+M3"] * 2 + ["M2+M4"] * 2 + ["M3"] * 2 + ["M4"] * 2
    assert not any(prov[k] == ["M2", "M3", "M4"] for k in chosen)  # 세 방식이 모두 뽑은 쌍은 M4 − M3와 무관
    assert set(re_.differing_pairs(prov)) >= set(chosen)  # 표본이 재판정 대상 범위 안에 있다
    assert chosen == re_.sample_targeted_pairs(prov, 2, seed=1)
    assert chosen != sorted(chosen, key=lambda k: prov[k])  # 조합별로 묶여 있지 않다(섞임)
    assert len(re_.sample_targeted_pairs(prov, 99, seed=1)) == 6 + 6 + 3 + 3  # 가진 만큼만
    assert {re_.group_of(prov[k]) for k in re_.sample_targeted_pairs(prov, 2, seed=1, groups=("M4",))} == {"M4"}


def test_group_offsets_computes_offset_and_standard_error():
    prov = _provenance()
    human = {"q::a1": 1, "q::a2": 1, "q::a3": 0, "q::a4": 1, "q::b1": 0, "q::b2": 0}
    auto = {"q::a1": 0, "q::a2": 1, "q::a3": 0, "q::a4": 0, "q::b1": 0, "q::b2": 1}
    offsets = re_.group_offsets(human, auto, prov)
    assert offsets["M4"]["n"] == 4 and offsets["M4"]["human_positive"] == 0.75 and offsets["M4"]["auto_positive"] == 0.25
    assert offsets["M4"]["offset"] == pytest.approx(0.5) and offsets["M4"]["se"] == pytest.approx(0.2887, abs=1e-3)
    assert offsets["M3"]["offset"] == pytest.approx(-0.5)


def test_adjusted_judgments_uses_human_rate_given_llm_verdict_and_leaves_other_groups():
    prov = _provenance()
    auto = {"q::a1": 0, "q::a2": 0, "q::a3": 1, "q::b1": 1, "q::c1": 1, "q::c2": 0}
    human = {"q::a1": 1}
    offsets = {
        "M4": {"rate_if_auto_0": 0.4, "rate_if_auto_1": 0.9, "human_positive": 0.6},
        "M3": {"rate_if_auto_0": None, "rate_if_auto_1": 0.7, "human_positive": 0.3},
    }
    adjusted = re_.adjusted_judgments(auto, human, prov, offsets)
    assert adjusted["q::a1"] == 1.0  # 사람이 채점한 쌍은 그 값
    assert adjusted["q::a2"] == pytest.approx(0.4) and adjusted["q::a3"] == pytest.approx(0.9)  # LLM이 같은 판정을 낸 표본에서 사람이 준 긍정률
    assert adjusted["q::b1"] == pytest.approx(0.7)
    assert adjusted["q::c1"] == 1.0 and adjusted["q::c2"] == 0.0  # 표본이 없는 조합은 그대로
    assert re_.adjusted_judgments({"q::b2": 0}, {}, prov, offsets)["q::b2"] == pytest.approx(0.3)  # 그 칸에 표본이 없으면 조합 전체 사람 긍정률
    assert re_.adjusted_judgments(auto, human, prov, offsets, shift={"M4": 0.1})["q::a2"] == pytest.approx(0.5)
    assert re_.adjusted_judgments({"q::a3": 1}, {}, prov, {"M4": {"rate_if_auto_1": 0.95, "human_positive": 0.6}}, shift={"M4": 0.2})["q::a3"] == 1.0  # 1을 넘지 않는다


def test_group_offsets_reports_human_rate_by_llm_verdict():
    prov = _provenance()
    human = {"q::a1": 1, "q::a2": 1, "q::a3": 0, "q::a4": 1}
    auto = {"q::a1": 0, "q::a2": 1, "q::a3": 0, "q::a4": 1}
    o = re_.group_offsets(human, auto, prov)["M4"]
    assert o["rate_if_auto_0"] == pytest.approx(0.5) and o["rate_if_auto_1"] == pytest.approx(1.0)
    assert re_.group_offsets({"q::a1": 1}, {"q::a1": 1}, prov)["M4"]["rate_if_auto_0"] is None  # 표본이 없는 칸은 None


def test_applied_positive_rate_matches_the_reported_human_rate_when_sample_is_the_whole_group():
    """보고하는 채점 긍정률과 실제로 적용된 긍정률이 어긋나지 않아야 한다(가법 offset을 더하고 자르던 때는 어긋났다)."""
    queries = [f"r{i}" for i in range(10)]
    candidates, auto, human, prov = {m: {} for m in re_.METHODS}, {}, {}, {}
    for i, q in enumerate(queries):
        m4, m3 = f"m4-{q}", f"m3-{q}"
        candidates["M2"][q], candidates["M3"][q], candidates["M4"][q] = [(m3, 1.0)], [(m3, 1.0)], [(m4, 1.0)]
        auto[f"{q}::{m4}"], prov[f"{q}::{m4}"] = int(i < 5), ["M4"]  # LLM은 절반만 1
        auto[f"{q}::{m3}"], prov[f"{q}::{m3}"] = 0, ["M3"]
        human[f"{q}::{m4}"] = 1  # 사람은 전부 1 → 채점 긍정률 1.0
        human[f"{q}::{m3}"] = 0
    result = re_.calibrated_summary(candidates, auto, human, prov, {"regular": queries, "short": []}, k=1)
    assert result["offsets"]["M4"]["human_positive"] == 1.0
    assert result["applied"]["M4"]["applied_positive"] == pytest.approx(1.0) and result["applied"]["M4"]["auto_positive"] == pytest.approx(0.5)
    assert result["coverage"] == {"differing": 20, "calibrated": 20}


def test_calibrated_summary_reports_uncovered_groups():
    prov = _provenance() | {"q::d1": ["M2", "M4"], "q::e1": ["M2", "M3"]}
    result = re_.calibrated_summary({m: {"q": [("a1", 1.0)]} for m in re_.METHODS}, {k: 0 for k in prov}, {"q::a1": 1, "q::b1": 0}, prov, {"regular": ["q"], "short": []}, k=1)
    assert result["coverage"]["differing"] == 6 + 6 + 1 + 1 and result["coverage"]["calibrated"] == 12  # M2+M4·M2+M3 조합은 표본이 없어 보정 안 됨


def test_calibrated_summary_can_flip_the_decision_when_human_disagrees_with_llm():
    """LLM은 M4만 뽑은 후보를 전부 0으로 봤지만 사람은 전부 1로 봤다면, 보정 후 M4 − M3가 커져 채택이 M4로 바뀐다."""
    queries = [f"r{i}" for i in range(30)]
    candidates, auto, human, prov = {m: {} for m in re_.METHODS}, {}, {}, {}
    for q in queries:
        shared = [(f"s{j}-{q}", 1.0) for j in range(3)]
        m4_only = [(f"m4-{q}-{j}", 1.0) for j in range(2)]
        m3_only = [(f"m3-{q}-{j}", 1.0) for j in range(2)]
        candidates["M2"][q], candidates["M3"][q], candidates["M4"][q] = shared + m3_only, shared + m3_only, shared + m4_only
        for cid, _ in shared:
            auto[f"{q}::{cid}"], prov[f"{q}::{cid}"] = 1, ["M2", "M3", "M4"]
        for cid, _ in m4_only:
            auto[f"{q}::{cid}"], prov[f"{q}::{cid}"] = 0, ["M4"]
            if q in queries[:10]:
                human[f"{q}::{cid}"] = 1  # 사람은 M4만 뽑은 후보를 전부 좋다고 봄
        for cid, _ in m3_only:
            auto[f"{q}::{cid}"], prov[f"{q}::{cid}"] = 0, ["M3"]
            if q in queries[:10]:
                human[f"{q}::{cid}"] = 0
    groups = {"regular": queries, "short": []}
    plain = re_.summarize_scores(candidates, {tuple(k.split("::")): v for k, v in auto.items()}, groups, k=5)
    assert plain["adopted"] == "M3"  # LLM 판정만으로는 M4가 M3보다 낫지 않음
    result = re_.calibrated_summary(candidates, auto, human, prov, groups, k=5)
    assert result["offsets"]["M4"]["offset"] == pytest.approx(1.0) and result["summary"]["adopted"] == "M4"
    assert result["m4_minus_m3_if_m4_unfavorable"] <= result["summary"]["groups"]["regular"]["m4_minus_m3"]["mean"] <= result["m4_minus_m3_if_m4_favorable"]


def test_targeted_sheet_and_calibrated_score_flow(tmp_path, monkeypatch, capsys):
    """targeted 시트 생성 → 채점 → human-agree --targeted → score --calibrated 흐름이 이어진다."""
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr(re_, "REAL_CONCURRENCY", 2)
    monkeypatch.setattr(re_, "REAL_TAG_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_JUDGE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_TARGETED_EACH", 5)
    monkeypatch.setattr(re_, "REAL_BONUS_GRID", [0.5])  # dev에서 bonus가 전부 동점이면 0.0이 뽑혀 M3·M4가 M2와 같아진다. 후보가 갈리도록 고정한다
    monkeypatch.setattr(re_, "OpenAITagger", _FakeTagger)
    monkeypatch.setattr(re_, "OpenAIJudge", _FakeJudge)
    monkeypatch.setattr("src.pipeline._embedding_client", lambda key: _FakeEmbedding())
    args = _args(data_dir)
    for stage in (re_.cmd_embed, re_.cmd_tag_llm, re_.cmd_select_params, re_.cmd_candidates, re_.cmd_judge_sheet, re_.cmd_auto_judge):
        stage(args)
    prov = json.loads((tmp_path / "real" / "pairs.json").read_text(encoding="utf-8"))["provenance"]
    assert any(v in (["M4"], ["M3"]) for v in prov.values())  # 결정을 가르는 쌍이 실제로 있어야 이 흐름을 검증한 것이다(M3만·M4만 중 한쪽은 가짜 데이터에서 없을 수 있다)
    args.targeted = True
    re_.cmd_human_sheet(args)
    sheet = tmp_path / "real" / "human_sheet_targeted.csv"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert rows and set(r["pair_id"] for r in rows) <= set(re_.differing_pairs(prov))
    header = "채점(1=추천에 넣을 만함, 0=아님)"
    for r in rows:
        r[header] = "1"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    re_.cmd_human_agree(args)
    saved = json.loads((tmp_path / "real" / "human_agree_targeted.json").read_text(encoding="utf-8"))
    assert set(saved["human"]) == {r["pair_id"] for r in rows} and saved["groups"]
    args.calibrated = True
    re_.cmd_score(args)
    score = json.loads((tmp_path / "real" / "score.json").read_text(encoding="utf-8"))
    assert score["calibrated"]["source"] == "human" and "offsets" in score["calibrated"]
    assert "보정한 정밀도" in capsys.readouterr().out


def test_calibrated_sensitivity_range_is_ordered_when_offsets_are_uncertain():
    """사람 채점이 엇갈려 표준오차가 0이 아닐 때, M4에 유리한 경우가 불리한 경우보다 M4 − M3가 커야 한다."""
    queries = [f"r{i}" for i in range(20)]
    candidates, auto, human, prov = {m: {} for m in re_.METHODS}, {}, {}, {}
    for i, q in enumerate(queries):
        m4, m3 = f"m4-{q}", f"m3-{q}"
        candidates["M2"][q], candidates["M3"][q], candidates["M4"][q] = [(m3, 1.0)], [(m3, 1.0)], [(m4, 1.0)]
        auto[f"{q}::{m4}"], prov[f"{q}::{m4}"] = 0, ["M4"]
        auto[f"{q}::{m3}"], prov[f"{q}::{m3}"] = 0, ["M3"]
        if i < 10:  # 사람이 절반은 1, 절반은 0으로 채점 → offset 0.5, 표준오차 > 0
            human[f"{q}::{m4}"], human[f"{q}::{m3}"] = i % 2, i % 2
    result = re_.calibrated_summary(candidates, auto, human, prov, {"regular": queries, "short": []}, k=1)
    assert result["offsets"]["M4"]["se"] > 0
    assert result["m4_minus_m3_if_m4_favorable"] > result["m4_minus_m3_if_m4_unfavorable"]


# ---- 재판정(rejudge) ---------------------------------------------------------------------------------------------------


def test_differing_pairs_are_those_picked_by_exactly_one_of_m3_and_m4():
    prov = {"q::a": ["M4"], "q::b": ["M3"], "q::c": ["M2", "M3", "M4"], "q::d": ["M2", "M4"], "q::e": ["M2", "M3"], "q::f": ["M2"], "q::g": ["M3", "M4"]}
    assert re_.differing_pairs(prov) == ["q::a", "q::b", "q::d", "q::e"]  # M2만·공통 쌍은 M4 − M3에 영향이 없다


def test_mix_judgments_replaces_only_rejudged_pairs():
    mixed = re_.mix_judgments({"q::a": 0, "q::b": 1, "q::c": 1}, {"q::a": 1, "q::b": 0})
    assert mixed == {"q::a": 1, "q::b": 0, "q::c": 1}


def test_m4_minus_m3_depends_only_on_differing_pairs():
    """공통 쌍의 판정을 어떻게 바꿔도 M4 − M3는 그대로다. 그래서 다른 쌍만 재판정해도 정확하다."""
    candidates = {"M2": {"q": [("s", 1.0), ("x", 1.0)]}, "M3": {"q": [("s", 1.0), ("x", 1.0)]}, "M4": {"q": [("s", 1.0), ("y", 1.0)]}}
    groups = {"regular": ["q"], "short": []}
    base = {("q", "s"): 1, ("q", "x"): 0, ("q", "y"): 1}
    changed_shared = {**base, ("q", "s"): 0}
    a = re_.summarize_scores(candidates, base, groups, k=2)["groups"]["regular"]["m4_minus_m3"]["mean"]
    b = re_.summarize_scores(candidates, changed_shared, groups, k=2)["groups"]["regular"]["m4_minus_m3"]["mean"]
    assert a == b == pytest.approx(0.5)


def test_openai_judge_omits_temperature_when_none():
    fake = _FakeChat('{"score": 1}')
    OpenAIJudge(model="m", client=fake, schema=BINARY_SCHEMA, temperature=None).judge("q", "c")
    assert "temperature" not in fake.kwargs
    OpenAIJudge(model="m", client=fake, schema=BINARY_SCHEMA).judge("q", "c")
    assert fake.kwargs["temperature"] == 0


def test_rejudge_stage_and_score_rejudged_flow(tmp_path, monkeypatch, capsys):
    """rejudge는 M3·M4가 다르게 뽑은 쌍만 다시 판정하고, score --rejudged가 그 결과로 M4 − M3를 다시 계산한다."""
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr(re_, "REAL_CONCURRENCY", 2)
    monkeypatch.setattr(re_, "REAL_TAG_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_JUDGE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_BONUS_GRID", [0.5])
    monkeypatch.setattr(re_, "OpenAITagger", _FakeTagger)
    monkeypatch.setattr(re_, "OpenAIJudge", _FakeJudge)
    monkeypatch.setattr("src.pipeline._embedding_client", lambda key: _FakeEmbedding())
    args = _args(data_dir)
    for stage in (re_.cmd_embed, re_.cmd_tag_llm, re_.cmd_select_params, re_.cmd_candidates, re_.cmd_judge_sheet, re_.cmd_auto_judge):
        stage(args)
    prov = json.loads((tmp_path / "real" / "pairs.json").read_text(encoding="utf-8"))["provenance"]
    expected = set(re_.differing_pairs(prov))
    assert expected
    re_.cmd_rejudge(args)
    saved = json.loads((tmp_path / "real" / "judgments_rejudge.json").read_text(encoding="utf-8"))
    assert set(saved) == expected  # 다른 쌍만 다시 판정
    args.rejudged = True
    re_.cmd_score(args)
    score = json.loads((tmp_path / "real" / "score.json").read_text(encoding="utf-8"))
    assert score["rejudged"]["model"] == re_.REAL_REJUDGE_MODEL and score["rejudged"]["summary"]["adopted"] in ("M3", "M4")
    assert "재판정" in capsys.readouterr().out


# ---- 리뷰 반영: 일치율 지문 · 잠정/확정 · 재판정 검증 ------------------------------------------------------------------


PROV = {"q::a": ["M2", "M3", "M4"], "q::b": ["M4"]}


def test_agreement_is_current_detects_changed_judgments_texts_and_missing_fingerprint():
    judg = {"q::a": {"score": 1, "hash": "h"}, "q::b": {"score": 0, "hash": "hb"}}
    agree = {"human": {"q::a": 1}, "fingerprint": re_.agreement_fingerprint(["q::a"], judg, PROV)}
    assert re_.agreement_is_current(agree, judg, PROV)
    assert not re_.agreement_is_current(agree, {**judg, "q::a": {"score": 0, "hash": "h"}}, PROV)  # 표본 쌍의 LLM 판정이 바뀜
    assert not re_.agreement_is_current(agree, {**judg, "q::a": {"score": 1, "hash": "h2"}}, PROV)  # 표본 쌍의 텍스트가 바뀜
    assert not re_.agreement_is_current(agree, {"q::b": judg["q::b"]}, PROV)  # 표본 쌍이 더는 없음
    assert not re_.agreement_is_current({"human": {"q::a": 1}}, judg, PROV)  # 지문이 없는 옛 결과
    assert not re_.agreement_is_current({}, judg, PROV)


def test_agreement_is_stale_when_the_evaluation_set_or_provenance_changes_but_the_sample_does_not():
    """채점 표본은 그대로여도 전체 평가 쌍이나 어느 방식이 뽑았는지가 바뀌면 이전 일치율·확정 상태를 재사용하면 안 된다."""
    judg = {"q::a": {"score": 1, "hash": "h"}, "q::b": {"score": 0, "hash": "hb"}}
    agree = {"human": {"q::a": 1}, "fingerprint": re_.agreement_fingerprint(["q::a"], judg, PROV)}
    assert not re_.agreement_is_current(agree, {**judg, "q::new": {"score": 1, "hash": "hn"}}, {**PROV, "q::new": ["M4"]})  # 새 후보 쌍 추가
    assert not re_.agreement_is_current(agree, {**judg, "q::b": {"score": 1, "hash": "hb"}}, PROV)  # 표본 밖 쌍의 판정이 바뀜
    assert not re_.agreement_is_current(agree, {**judg, "q::b": {"score": 0, "hash": "다른-텍스트"}}, PROV)  # 표본 밖 쌍의 텍스트가 바뀜
    assert not re_.agreement_is_current(agree, judg, {**PROV, "q::b": ["M3"]})  # 같은 쌍인데 뽑은 방식이 바뀜
    assert not re_.agreement_is_current(agree, judg, {**PROV, "q::extra": ["M3"]})  # 판정은 없고 후보 출처만 늘어남


def test_agreement_is_stale_when_the_stored_sample_differs_from_the_fingerprinted_sample():
    """일치율 파일의 채점 표본이 지문을 만든 표본과 다르면(파일이 편집되거나 다른 표본에서 계산됨) 무효다."""
    judg = {"q::a": {"score": 1, "hash": "h"}, "q::b": {"score": 0, "hash": "hb"}}
    agree = {"human": {"q::a": 1, "q::b": 0}, "fingerprint": re_.agreement_fingerprint(["q::a", "q::b"], judg, PROV)}
    assert re_.agreement_is_current(agree, judg, PROV)
    assert not re_.agreement_is_current({**agree, "human": {"q::a": 1}}, judg, PROV)  # 표본에서 한 쌍이 빠짐
    assert not re_.agreement_is_current({**agree, "human": {"q::b": 0}}, judg, PROV)  # 다른 쌍으로 바뀜


def test_adoption_status_is_confirmed_only_when_a_human_check_passed():
    assert re_.adoption_status({"source": "human", "passed": True}) == "확정"
    assert re_.adoption_status({"source": "human", "passed": False}) == "잠정"  # 일치율 미달
    assert re_.adoption_status({"source": "claude", "passed": True}) == "잠정"  # 사람 채점이 아님
    assert re_.adoption_status({}) == "잠정"  # 확인 안 함


def _score_fixture(tmp_path, agree_fingerprint_ok: bool, source: str = "human"):
    """후보 쌍 하나짜리 최소 결과 폴더와, 일치율 결과 파일을 만든다."""
    judg = {"q::a": {"score": 1, "hash": "h"}}
    write = lambda name, obj: (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    write("candidates.json", {m: {"q": [["a", 1.0]]} for m in re_.METHODS})
    write("judgments.json", judg)
    write("params.json", {"queries": {"regular": ["q"], "short": []}})
    write("pairs.json", {"pairs": [["q", "a"]], "provenance": {"q::a": ["M2", "M3", "M4"]}})
    fingerprint = re_.agreement_fingerprint(["q::a"], judg, {"q::a": ["M2", "M3", "M4"]}) if agree_fingerprint_ok else "옛-지문"
    write("human_agree.json", {"source": source, "passed": True, "human": {"q::a": 1}, "fingerprint": fingerprint})


def test_score_ignores_agreement_computed_on_other_inputs_and_marks_provisional(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    _score_fixture(tmp_path, agree_fingerprint_ok=False)
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    out = capsys.readouterr().out
    assert "다른 입력에서 계산됐습니다" in out and "잠정" in out and "확정" not in out.split("채택:")[1].split("\n")[0]
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "잠정"


def test_score_marks_confirmed_only_for_current_passed_human_agreement(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    _score_fixture(tmp_path, agree_fingerprint_ok=True)
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "확정"
    _score_fixture(tmp_path, agree_fingerprint_ok=True, source="claude")
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "잠정"
    assert "사람 채점이 아니라" in capsys.readouterr().out


def test_score_calibrated_refuses_stale_targeted_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    _score_fixture(tmp_path, agree_fingerprint_ok=True)
    (tmp_path / "human_agree_targeted.json").write_text(json.dumps({"source": "human", "human": {"q::a": 1}, "fingerprint": "옛-지문"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="human-agree --targeted"):
        re_.cmd_score(argparse.Namespace(calibrated=True, rejudged=False))


def test_score_rejudged_refuses_missing_or_stale_rejudgments(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    write = lambda name, obj: (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    write("candidates.json", {"M2": {"q": [["b", 1.0]]}, "M3": {"q": [["b", 1.0]]}, "M4": {"q": [["a", 1.0]]}})
    write("judgments.json", {"q::a": {"score": 1, "hash": "ha"}, "q::b": {"score": 0, "hash": "hb"}})
    write("params.json", {"queries": {"regular": ["q"], "short": []}})
    write("pairs.json", {"pairs": [["q", "a"], ["q", "b"]], "provenance": {"q::a": ["M4"], "q::b": ["M3"]}})
    write("judgments_rejudge.json", {"q::a": {"score": 1, "hash": "ha"}})  # q::b 재판정 없음
    with pytest.raises(RuntimeError, match="rejudge"):
        re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=True))
    write("judgments_rejudge.json", {"q::a": {"score": 1, "hash": "ha"}, "q::b": {"score": 0, "hash": "옛-텍스트"}})  # 텍스트 해시가 다름
    with pytest.raises(RuntimeError, match="rejudge"):
        re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=True))
    write("judgments_rejudge.json", {"q::a": {"score": 1, "hash": "ha"}, "q::b": {"score": 1, "hash": "hb"}})
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=True))  # 맞으면 통과
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["rejudged"]["adopted_status"] == "잠정"


def test_read_filled_sheets_merges_multiple_files_and_rejects_conflicts_and_blanks(tmp_path):
    header = ["pair_id", "쿼리 소개", "후보 소개", "채점(1=추천에 넣을 만함, 0=아님)", "메모"]

    def sheet(name, rows):
        path = tmp_path / name
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    a = sheet("a.csv", [["q::1", "x", "y", "1", ""], ["q::2", "x", "y", "0", ""]])
    b = sheet("b.csv", [["q::3", "x", "y", "1", ""], ["q::2", "x", "y", "0", ""]])
    assert re_._read_filled_sheets([a, b]) == {"q::1": 1, "q::2": 0, "q::3": 1}
    with pytest.raises(ValueError, match="다르게"):
        re_._read_filled_sheets([a, sheet("c.csv", [["q::2", "x", "y", "1", ""]])])
    with pytest.raises(ValueError, match="비었거나"):
        re_._read_filled_sheets([sheet("d.csv", [["q::9", "x", "y", "", ""]])])


def test_score_does_not_carry_confirmed_status_to_an_extended_candidate_set(tmp_path, monkeypatch, capsys):
    """표본은 그대로 두고 평가 쌍만 늘려도(후보 집합이 바뀜) 이전의 확정 상태가 붙으면 안 된다."""
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    _score_fixture(tmp_path, agree_fingerprint_ok=True)
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "확정"
    judg = json.loads((tmp_path / "judgments.json").read_text(encoding="utf-8"))
    pairs = json.loads((tmp_path / "pairs.json").read_text(encoding="utf-8"))
    judg["q::z"] = {"score": 1, "hash": "hz"}
    pairs["provenance"]["q::z"] = ["M4"]
    (tmp_path / "judgments.json").write_text(json.dumps(judg), encoding="utf-8")
    (tmp_path / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "잠정"
    assert "다른 입력에서 계산됐습니다" in capsys.readouterr().out


def _four_group_fixture():
    """모든 쿼리가 M4만·M2+M4·M3만·M2+M3 후보를 하나씩 가지고, 앞 절반은 조합마다 1/0이 섞이게 채점된 데이터를 만든다."""
    queries = [f"r{i}" for i in range(20)]
    candidates, auto, human, prov = {m: {} for m in re_.METHODS}, {}, {}, {}
    for i, q in enumerate(queries):
        ids = {"M4": f"a-{q}", "M2+M4": f"b-{q}", "M3": f"c-{q}", "M2+M3": f"d-{q}"}
        candidates["M2"][q] = [(ids["M2+M3"], 1.0), (ids["M2+M4"], 1.0)]
        candidates["M3"][q] = [(ids["M3"], 1.0), (ids["M2+M3"], 1.0)]
        candidates["M4"][q] = [(ids["M4"], 1.0), (ids["M2+M4"], 1.0)]
        for group, cid in ids.items():
            key = f"{q}::{cid}"
            auto[key], prov[key] = 0, group.split("+")
            if i < 10:
                human[key] = i % 2  # 조합마다 채점이 엇갈려 표준오차가 0보다 크다
    return candidates, auto, human, prov, {"regular": queries, "short": []}


def test_sensitivity_range_moves_every_calibrated_group_not_only_m4_and_m3():
    candidates, auto, human, prov, groups = _four_group_fixture()
    result = re_.calibrated_summary(candidates, auto, human, prov, groups, k=2)
    offsets = result["offsets"]
    assert all(offsets[g]["se"] > 0 for g in ("M4", "M2+M4", "M3", "M2+M3"))
    sign = {"M4": 1, "M2+M4": 1, "M3": -1, "M2+M3": -1}

    def mean_diff(shift):
        adjusted = re_.adjusted_judgments(auto, human, prov, offsets, shift)
        summary = re_.summarize_scores(candidates, {tuple(k.split("::")): v for k, v in adjusted.items()}, groups, k=2)
        return summary["groups"]["regular"]["m4_minus_m3"]["mean"]

    assert result["m4_minus_m3_if_m4_favorable"] == pytest.approx(mean_diff({g: sign[g] * offsets[g]["se"] for g in sign}))
    assert result["m4_minus_m3_if_m4_unfavorable"] == pytest.approx(mean_diff({g: -sign[g] * offsets[g]["se"] for g in sign}))
    only_two = mean_diff({"M4": offsets["M4"]["se"], "M3": -offsets["M3"]["se"]})
    assert result["m4_minus_m3_if_m4_favorable"] > only_two  # M2+M4·M2+M3까지 움직이면 범위가 더 넓다


def test_m4_direction_signs():
    assert [re_._m4_direction(g) for g in ("M4", "M2+M4", "M3", "M2+M3", "M3+M4", "M2")] == [1, 1, -1, -1, 0, 0]


def test_stale_cached_judgments_are_ignored_by_sheet_score_and_agreement(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path)
    _score_fixture(tmp_path, agree_fingerprint_ok=True)
    judg = json.loads((tmp_path / "judgments.json").read_text(encoding="utf-8"))
    judg["옛질의::옛후보"] = {"score": 1, "hash": "옛"}  # 후보를 다시 만들기 전에 캐시된 쌍
    (tmp_path / "judgments.json").write_text(json.dumps(judg), encoding="utf-8")
    assert set(re_._current_judgments()) == {"q::a"}
    # 옛 쌍이 있어도 현재 판정만으로 만든 일치율 지문과 같아 확정이 유지된다
    re_.cmd_score(argparse.Namespace(calibrated=False, rejudged=False))
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["adopted_status"] == "확정"


@pytest.fixture(autouse=True)
def _hand_built_results_skip_freshness(request, tmp_path_factory, monkeypatch):
    """손으로 만든 결과 파일(해시 없음)을 쓰는 단위 테스트는 데이터 폴더만 채워 주고 최신성 검사는 건너뛴다.
    전체 흐름(flow)·최신성(freshness) 테스트는 진짜 검사를 그대로 쓴다."""
    if "flow" in request.node.name or "freshness" in request.node.name:
        return
    root = tmp_path_factory.mktemp("hand_built_data")
    _write_data(root)
    monkeypatch.setenv(re_.REAL_DATA_ENV, str(root))
    monkeypatch.setattr(re_, "_assert_judgments_fresh", lambda *a, **k: None)


def _run_flow(tmp_path, monkeypatch, through_rejudge: bool = False):
    """가짜 클라이언트로 auto-judge(와 선택적으로 rejudge)까지 실행하고 (args, data_dir)를 돌려준다."""
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    monkeypatch.setattr(re_, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr(re_, "REAL_CONCURRENCY", 2)
    monkeypatch.setattr(re_, "REAL_TAG_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_JUDGE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(re_, "REAL_TARGETED_EACH", 5)
    monkeypatch.setattr(re_, "REAL_BONUS_GRID", [0.5])
    monkeypatch.setattr(re_, "OpenAITagger", _FakeTagger)
    monkeypatch.setattr(re_, "OpenAIJudge", _FakeJudge)
    monkeypatch.setattr("src.pipeline._embedding_client", lambda key: _FakeEmbedding())
    args = _args(data_dir)
    stages = [re_.cmd_embed, re_.cmd_tag_llm, re_.cmd_select_params, re_.cmd_candidates, re_.cmd_judge_sheet, re_.cmd_auto_judge]
    if through_rejudge:
        stages.append(re_.cmd_rejudge)
    for stage in stages:
        stage(args)
    return args, data_dir


def _change_bio(data_dir, channel_id):
    """풀 CSV에서 해당 채널의 소개글만 바꾼다."""
    pool = data_dir / "raw" / "pool.csv"
    rows = list(csv.reader(pool.open(encoding="utf-8-sig")))
    for row in rows[1:]:
        if row[0] == channel_id:
            row[2] = "완전히 바뀐 소개글 내용입니다 열다섯 글자를 넘김"
    with pool.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def _judged_query(tmp_path):
    """판정한 쌍에 실제로 들어 있는 채널 하나(쿼리 쪽)를 돌려준다."""
    return json.loads((tmp_path / "real" / "pairs.json").read_text(encoding="utf-8"))["pairs"][0][0]


def test_freshness_score_rejects_judgments_when_bio_or_judge_settings_change(tmp_path, monkeypatch):
    args, data_dir = _run_flow(tmp_path, monkeypatch)
    re_.cmd_score(args)  # 그대로면 통과
    original = re_.BINARY_SYSTEM_PROMPT
    monkeypatch.setattr(re_, "BINARY_SYSTEM_PROMPT", original + " 바뀐 판정 기준")
    with pytest.raises(RuntimeError, match="auto-judge"):
        re_.cmd_score(args)
    monkeypatch.setattr(re_, "BINARY_SYSTEM_PROMPT", original)
    re_.cmd_score(args)  # 되돌리면 다시 통과
    _change_bio(data_dir, _judged_query(tmp_path))
    with pytest.raises(RuntimeError, match="auto-judge"):
        re_.cmd_score(_args(data_dir))


def test_freshness_llm_tags_reject_changed_bio_or_tag_prompt(tmp_path, monkeypatch):
    args, data_dir = _run_flow(tmp_path, monkeypatch)
    data = re_.load_real_data(data_dir)
    assert re_._llm_tags(data)  # 그대로면 통과
    _change_bio(data_dir, "c0_0")
    with pytest.raises(RuntimeError, match="tag-llm"):
        re_._llm_tags(re_.load_real_data(data_dir))
    monkeypatch.setattr(re_, "LLM_TAG_MAX", re_.LLM_TAG_MAX + 1)
    with pytest.raises(RuntimeError, match="tag-llm"):
        re_._llm_tags(data)


def test_freshness_rejudged_rejects_results_from_another_model(tmp_path, monkeypatch):
    args, _ = _run_flow(tmp_path, monkeypatch, through_rejudge=True)
    args.rejudged = True
    re_.cmd_score(args)  # 같은 모델이면 통과
    monkeypatch.setattr(re_, "REAL_REJUDGE_MODEL", "다른-모델")
    with pytest.raises(RuntimeError, match="rejudge"):
        re_.cmd_score(args)


def test_freshness_human_sheet_refuses_to_overwrite_filled_sheet_unless_forced(tmp_path, monkeypatch):
    args, _ = _run_flow(tmp_path, monkeypatch)
    sheet = tmp_path / "real" / "human_sheet.csv"
    re_.cmd_human_sheet(args)  # 빈 시트는 다시 만들어도 된다
    re_.cmd_human_sheet(args)
    header = "채점(1=추천에 넣을 만함, 0=아님)"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    rows[0][header] = "1"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(FileExistsError, match="채점"):
        re_.cmd_human_sheet(args)
    assert list(csv.DictReader(sheet.open(encoding="utf-8-sig")))[0][header] == "1"  # 채점이 남아 있다
    args.force = True
    re_.cmd_human_sheet(args)
    assert list(csv.DictReader(sheet.open(encoding="utf-8-sig")))[0][header] == ""


def test_freshness_targeted_sheet_n_each_reproduces_uneven_group_sizes(tmp_path, monkeypatch):
    """보고한 150쌍(M4만 50·M3만 50 + M2+M4 25·M2+M3 25)은 --groups와 --n-each로 다시 만들 수 있다."""
    args, _ = _run_flow(tmp_path, monkeypatch)
    prov = json.loads((tmp_path / "real" / "pairs.json").read_text(encoding="utf-8"))["provenance"]
    args.targeted, args.n_each = True, 2
    args.groups, args.out = "M4,M3", "a.csv"
    re_.cmd_human_sheet(args)
    ids = [r["pair_id"] for r in csv.DictReader((tmp_path / "real" / "a.csv").open(encoding="utf-8-sig"))]
    assert ids == re_.sample_targeted_pairs(prov, 2, re_.REAL_SEED, ("M4", "M3"))
    assert all(re_.group_of(prov[k]) in ("M4", "M3") for k in ids)
