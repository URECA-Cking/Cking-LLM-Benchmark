"""BE InterestSubtopicHash와 공유하는 세부 주제 해시 계약."""

import hashlib
import json
import sys
from pathlib import Path

from src.recommendation.taxonomy import normalize_taxonomy_field


def canonical_subtopic_json(taxonomy: dict) -> str:
    """정본 배열 순서(display_order)를 보존하고 definition을 description으로 매핑한다.

    version/status/include/exclude/formats는 BE 저장 해시에 포함되지 않는다.
    """
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("subtopics"), list) or not taxonomy["subtopics"]:
        raise ValueError("subtopics는 비어 있지 않은 배열이어야 합니다.")
    rows = []
    codes = set()
    for index, row in enumerate(taxonomy["subtopics"]):
        normalized = {}
        for key, source in (
            ("code", "code"), ("parent", "parent"), ("name", "name"),
            ("description", "definition"),
        ):
            if not isinstance(row, dict) or not isinstance(row.get(source), str):
                raise ValueError(f"subtopics[{index}].{source}: 문자열 필드가 필요합니다.")
            value = normalize_taxonomy_field(row[source])
            if not value:
                raise ValueError(f"subtopics[{index}].{source}: 정규화 후 빈 값입니다.")
            normalized[key] = value
        if normalized["code"] in codes:
            raise ValueError(f"subtopics[{index}].code: 정규화 후 중복 코드 {normalized['code']}")
        codes.add(normalized["code"])
        rows.append(normalized)
    return json.dumps({"subtopics": rows}, ensure_ascii=False, separators=(",", ":"))


def subtopic_hash(taxonomy: dict) -> str:
    return hashlib.sha256(canonical_subtopic_json(taxonomy).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except FileExistsError as error:
        print(f"ERROR: 출력 경로가 이미 존재합니다. 새 디렉터리를 지정하세요: {error.filename}", file=sys.stderr)
    except FileNotFoundError as error:
        print(f"ERROR: 파일 또는 디렉터리를 찾을 수 없습니다: {error.filename}", file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
    return 2


def _run(argv: list[str] | None = None) -> int:
    """공유 fixture 비교 또는 별도 디렉터리에 새 검토본을 출력한다."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture-dir", type=Path, help="기존 canonical JSON 및 해시와 비교")
    mode.add_argument("--output-dir", type=Path, help="존재하지 않는 디렉터리에 새 검토본 생성")
    args = parser.parse_args(argv)
    taxonomy = json.loads(args.source.read_text(encoding="utf-8"))
    actual = canonical_subtopic_json(taxonomy).encode("utf-8")
    digest = hashlib.sha256(actual).hexdigest()
    if args.output_dir is not None:
        # 기존 초안이나 이전 검토본을 실수로 교체하지 않는다.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "subtopic-v1.canonical.json").write_bytes(actual)
        (args.output_dir / "subtopic-v1.sha256.txt").write_bytes(digest.encode("ascii"))
        print(f"GENERATED: {args.output_dir} (검토본; 서비스 승인 아님)")
        print(digest)
        return 0
    expected = (args.fixture_dir / "subtopic-v1.canonical.json").read_bytes()
    expected_hash = (args.fixture_dir / "subtopic-v1.sha256.txt").read_text(encoding="ascii").strip()
    if actual != expected or digest != expected_hash:
        print("FAIL: canonical bytes or SHA-256 mismatch")
        return 1
    print(f"PASS: {len(taxonomy['subtopics'])} rows, {len(actual)} bytes")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
