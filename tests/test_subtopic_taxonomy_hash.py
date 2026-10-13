import json
import hashlib
from pathlib import Path

import pytest

from src.recommendation.subtopic_taxonomy import canonical_subtopic_json, subtopic_hash, main

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/taxonomy"


def test_be_draft_fixture_matches_source_bytes_and_hash():
    source = json.loads((ROOT / "data/creator-subtopics.json").read_text(encoding="utf-8"))
    assert canonical_subtopic_json(source).encode("utf-8") == (FIXTURES / "subtopic-v1.canonical.json").read_bytes()
    assert subtopic_hash(source) == (FIXTURES / "subtopic-v1.sha256.txt").read_text().strip()


def test_normalization_escaping_order_and_excluded_fields():
    row = {"code": " A\r\n", "parent": " P\t", "name": "\u1100\u1161", "definition": ' \r\n\u00a0\u3000"\\\b\f\x00\tX\rY\u00a0 '}
    source = {"subtopics": [row], "version": "draft", "formats": []}
    canonical = canonical_subtopic_json(source)
    assert json.loads(canonical) == {"subtopics": [{"code": "A", "parent": "P", "name": "가", "description": '\u00a0\u3000"\\\b\f\x00\tX\nY\u00a0'}]}
    assert '\\u0000' in canonical and '\\b' in canonical and '\\f' in canonical
    assert not canonical.endswith("\n") and not canonical.startswith("\ufeff")
    source["version"] = "service"
    row["include"] = ["changed"]
    assert canonical_subtopic_json(source) == canonical
    source["subtopics"].append({**row, "code": "B"})
    before = subtopic_hash(source)
    source["subtopics"].reverse()
    assert subtopic_hash(source) != before


@pytest.mark.parametrize("field", ["code", "parent", "name", "definition"])
def test_changed_contract_field_changes_hash(field):
    source = {"subtopics": [{"code": "A", "parent": "P", "name": "이름", "definition": "설명"}]}
    before = subtopic_hash(source)
    source["subtopics"][0][field] += "변경"
    assert subtopic_hash(source) != before


@pytest.mark.parametrize("change", ["add", "remove", "reverse"])
def test_changed_list_is_generated_and_rejected_by_draft_comparison(tmp_path, change):
    source = json.loads((ROOT / "data/creator-subtopics.json").read_text(encoding="utf-8"))
    if change == "add":
        source["subtopics"].append({**source["subtopics"][0], "code": "NEW"})
    elif change == "remove":
        source["subtopics"].pop()
    else:
        source["subtopics"].reverse()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "review"
    draft = (FIXTURES / "subtopic-v1.canonical.json").read_bytes()
    assert main([str(path), "--output-dir", str(output)]) == 0
    canonical = (output / "subtopic-v1.canonical.json").read_bytes()
    assert canonical == canonical_subtopic_json(source).encode("utf-8")
    assert not canonical.startswith(b"\xef\xbb\xbf") and not canonical.endswith(b"\n")
    assert list(json.loads(canonical)) == ["subtopics"]
    assert list(json.loads(canonical)["subtopics"][0]) == ["code", "parent", "name", "description"]
    assert (output / "subtopic-v1.sha256.txt").read_bytes() == hashlib.sha256(canonical).hexdigest().encode("ascii")
    assert main([str(path), "--fixture-dir", str(output)]) == 0
    assert main([str(path), "--fixture-dir", str(FIXTURES)]) == 1
    with pytest.raises(FileExistsError):
        main([str(path), "--output-dir", str(FIXTURES)])
    assert (FIXTURES / "subtopic-v1.canonical.json").read_bytes() == draft


def test_comparison_checks_bytes_and_hash_independently(tmp_path):
    source = ROOT / "data/creator-subtopics.json"
    assert main([str(source), "--output-dir", str(tmp_path / "review")]) == 0
    output = tmp_path / "review"
    digest = output / "subtopic-v1.sha256.txt"
    original = digest.read_bytes()
    digest.write_bytes(b"0" * 64)
    assert main([str(source), "--fixture-dir", str(output)]) == 1
    digest.write_bytes(original)
    canonical = output / "subtopic-v1.canonical.json"
    canonical.write_bytes(canonical.read_bytes() + b"\n")
    assert main([str(source), "--fixture-dir", str(output)]) == 1
