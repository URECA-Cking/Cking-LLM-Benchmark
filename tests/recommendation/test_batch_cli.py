from __future__ import annotations

from pathlib import Path

import pytest

from src.recommendation.batch_cli import build_parser


def test_manifest_command_requires_base_url_and_supports_operational_settings() -> None:
    args = build_parser().parse_args(
        [
            "manifest",
            "--be-base-url",
            "https://be.example",
            "--output",
            "manifest.json",
            "--timeout",
            "4",
            "--retries",
            "3",
            "--refresh",
        ]
    )
    assert args.command == "manifest"
    assert args.output == Path("manifest.json")
    assert args.timeout == 4
    assert args.retries == 3
    assert args.refresh is True


def test_dry_run_has_no_backend_or_token_option() -> None:
    args = build_parser().parse_args(["dry-run", "--manifest", "manifest.json"])
    assert args.command == "dry-run"
    assert not hasattr(args, "be_base_url")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dry-run", "--be-base-url", "https://be.example"])


def test_apply_exposes_top_n_backend_timeout_and_retry_settings() -> None:
    args = build_parser().parse_args(
        [
            "apply",
            "--manifest",
            "manifest.json",
            "--be-base-url",
            "https://be.example",
            "--top-n",
            "7",
            "--timeout",
            "12",
            "--retries",
            "1",
        ]
    )
    assert args.top_n == 7
    assert args.timeout == 12
    assert args.retries == 1
    assert not hasattr(args, "access_token")
