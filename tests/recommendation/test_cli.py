import json

import pytest

from src.recommendation.cli import load_request
from src.recommendation.runtime import ServiceCategory, build_service_tag_prompt, load_service_categories


def test_load_request_reads_service_contract(tmp_path) -> None:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "creatorId": 10,
                "introduction": "홈트레이닝을 소개합니다.",
                "topN": 3,
                "candidateCreators": [
                    {"creatorId": 20, "introduction": "근력 운동을 소개합니다."},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    seed, candidates, top_n = load_request(path)

    assert seed.creator_id == 10
    assert candidates[0].creator_id == 20
    assert top_n == 3


@pytest.mark.parametrize("top_n", [True, 0, -1, 1.5, "5"])
def test_load_request_rejects_invalid_top_n(tmp_path, top_n) -> None:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "creatorId": 10,
                "introduction": "소개",
                "topN": top_n,
                "candidateCreators": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="topN"):
        load_request(path)


def test_load_categories_and_prompt_bind_taxonomy_content(tmp_path) -> None:
    path = tmp_path / "categories.csv"
    path.write_text(
        "code,name,description\nFITNESS,운동,홈트와 러닝\nFOOD,음식,요리와 맛집\n",
        encoding="utf-8",
    )

    categories = load_service_categories(path)
    prompt = build_service_tag_prompt(categories)

    assert categories == [
        ServiceCategory("FITNESS", "운동", "홈트와 러닝"),
        ServiceCategory("FOOD", "음식", "요리와 맛집"),
    ]
    assert "FITNESS(운동): 홈트와 러닝" in prompt
    assert "UNCLASSIFIED" in prompt


def test_load_categories_rejects_duplicate_code(tmp_path) -> None:
    path = tmp_path / "categories.csv"
    path.write_text(
        "code,name,description\nFITNESS,운동,홈트\nFITNESS,피트니스,러닝\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="중복"):
        load_service_categories(path)
