"""Previously evaluated queries and reviewed candidates never become final seeds."""

import json
from src.final_eval_prepare import excluded_seeds, sample_inputs
from src.real_eval import Channel


def test_exclusion_flattens_cases_and_reviews(tmp_path):
    base = {
        "dev_ids": ["dev"],
        "cases": [{"seeds": ["query"]}, {"seeds": ["fav1", "fav2"]}],
    }
    p = tmp_path / "review.json"
    p.write_text(
        json.dumps(
            {
                "scores": {
                    "creator-human::candidate": {},
                    "tag-X::tagcandidate": {},
                    "tagged": {},
                }
            }
        )
    )
    assert excluded_seeds(base, [p]) == {
        "dev",
        "query",
        "fav1",
        "fav2",
        "human",
        "candidate",
        "tagcandidate",
        "tagged",
    }


def test_unseen_sampling_is_disjoint_balanced_and_stable():
    pool = [Channel(str(i), str(i), "a" * 200) for i in range(80)] + [
        Channel("s" + str(i), "s", "brief") for i in range(30)
    ]
    gold = {str(i): "X" if i % 2 else "Y" for i in range(80)}
    result = sample_inputs(pool, {"0", "s0"}, gold, ["X", "Y"])
    assert len(result) == 100 and sum(c["group"] == "regular" for c in result) == 75
    assert not {"0", "s0"} & {c["seeds"][0] for c in result}
    assert result == sample_inputs(list(reversed(pool)), {"0", "s0"}, gold, ["X", "Y"])
