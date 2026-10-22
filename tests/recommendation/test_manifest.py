from __future__ import annotations

import json

import pytest

from src.recommendation.manifest import fetch_manifest, load_manifest, write_manifest


class FakeBackend:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.get_paths: list[str] = []

    def get(self, path: str) -> dict[str, object]:
        self.get_paths.append(path)
        return self.pages.pop(0)

    def put(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("manifest 조회에서 PUT을 호출하면 안 됩니다.")


def page(number: int, items: list[dict[str, object]], *, total: int, pages: int) -> dict[str, object]:
    return {
        "items": items,
        "page": number,
        "size": 100,
        "totalElements": total,
        "totalPages": pages,
        "hasNext": number + 1 < pages,
    }


def test_fetches_all_pages_sorts_and_round_trips_deterministic_manifest(tmp_path) -> None:
    backend = FakeBackend(
        [
            page(0, [{"creatorId": 20, "introText": " 둘째\r\n"}], total=2, pages=2),
            page(1, [{"creatorId": 10, "introText": "첫째"}], total=2, pages=2),
        ]
    )

    manifest = fetch_manifest(backend)
    output = tmp_path / "creator-manifest.json"
    write_manifest(output, manifest)
    loaded = load_manifest(output)

    assert backend.get_paths == ["/api/creators?page=0&size=100", "/api/creators?page=1&size=100"]
    assert [creator.creator_id for creator in manifest.creators] == [10, 20]
    assert manifest.creators[1].text == "둘째"
    assert loaded == manifest
    assert len(manifest.manifest_hash) == 64


@pytest.mark.parametrize(
    "pages, message",
    [
        (
            [page(0, [{"creatorId": 1, "introText": "a"}, {"creatorId": 1, "introText": "b"}], total=2, pages=1)],
            "중복 creatorId",
        ),
        (
            [page(0, [{"creatorId": 0, "introText": "a"}], total=1, pages=1)],
            "양의 정수",
        ),
        (
            [page(0, [], total=1, pages=2)],
            "hasNext=true",
        ),
    ],
)
def test_rejects_duplicate_invalid_and_inconsistent_pages(pages, message) -> None:
    with pytest.raises(ValueError, match=message):
        fetch_manifest(FakeBackend(pages))


def test_load_rejects_changed_content_and_unsorted_order(tmp_path) -> None:
    backend = FakeBackend([page(0, [{"creatorId": 1, "introText": "a"}], total=1, pages=1)])
    output = tmp_path / "manifest.json"
    write_manifest(output, fetch_manifest(backend))
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["creators"][0]["introText"] = "changed"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_manifest(output)
