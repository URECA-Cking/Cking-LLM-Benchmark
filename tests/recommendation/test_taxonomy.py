import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from src.recommendation.taxonomy import (
    DEFAULT_CATEGORIES_CSV,
    ServiceCategory,
    canonical_taxonomy_json,
    load_service_categories,
    taxonomy_hash,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "taxonomy"


def test_cross_language_fixture_matches_exact_utf8_bytes_and_hash():
    raw = (FIXTURES / "input.csv").read_bytes()
    assert b"\r\n" in raw and b"\r" in raw and b"\n" in raw and b"\t" in raw
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    categories = [ServiceCategory(**row) for row in rows]
    expected = (FIXTURES / "canonical.json").read_bytes()
    assert not expected.startswith(b"\xef\xbb\xbf")
    assert not expected.endswith(b"\n")
    assert canonical_taxonomy_json(categories).encode("utf-8") == expected
    assert canonical_taxonomy_json(load_service_categories(FIXTURES / "input.csv")).encode("utf-8") == expected
    digest = (FIXTURES / "sha256.txt").read_text(encoding="ascii")
    assert taxonomy_hash(categories) == digest == hashlib.sha256(expected).hexdigest()
    payload = json.loads(expected)
    assert list(payload) == ["categories"]
    assert list(payload["categories"][0]) == ["code", "name", "description"]
    assert payload["categories"][0]["name"] == "운동·건강"
    assert payload["categories"][0]["description"] == "홈트\n러닝\n요가\n내부  공백\t유지"
    assert payload["categories"][1]["name"] == "\u00a0요리·푸드\u00a0"
    assert payload["categories"][1]["description"].startswith("\u2003")
    assert taxonomy_hash(reversed(categories)) != digest


def test_v02_source_matches_frozen_evaluation_taxonomy():
    categories = load_service_categories(DEFAULT_CATEGORIES_CSV)
    assert [row.code for row in categories] == [
        "FITNESS", "FOOD", "GAME", "BEAUTY", "FASHION", "TRAVEL", "MUSIC", "PET", "TECH",
        "KNOWLEDGE", "ENTERTAIN", "HUMOR", "SPORTS", "NEWS", "LIFE", "HOBBY", "LIFETIP",
    ]
    assert canonical_taxonomy_json(categories).encode("utf-8") == (FIXTURES / "v02.canonical.json").read_bytes()
    assert taxonomy_hash(categories) == (FIXTURES / "v02.sha256.txt").read_text(encoding="ascii")


@pytest.mark.parametrize("content", [
    "code,name,description\n",
    "code,name,description\nA, ,설명\n",
    "code,name,description\nA,이름,\t\n",
    "code,name,description\nA,이름\n",
    "code,name,description\nA,이름,설명,추가\n",
    "name,description\n이름,설명\n",
    "code,name,description\n A ,이름,설명\nA,다름,다름\n",
    "code,name,description\n가,이름,설명\n가,다름,다름\n",
])
def test_invalid_csv_is_rejected_after_normalization(tmp_path, content):
    path = tmp_path / "categories.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_service_categories(path)


@pytest.mark.parametrize("field", ["code", "name", "description"])
def test_each_field_change_affects_taxonomy_hash(field):
    original = {"code": "FOOD", "name": "요리", "description": "베이킹"}
    changed = {**original, field: original[field] + "변경"}
    assert taxonomy_hash([ServiceCategory(**original)]) != taxonomy_hash([ServiceCategory(**changed)])
