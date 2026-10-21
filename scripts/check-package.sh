#!/usr/bin/env bash
# 배포 패키지를 서버와 같은 Ubuntu 26.04에서, 서버처럼 읽기 전용 파일 시스템·root가 아닌 사용자로 확인한다.
# 확인: 패키지 안 Python 3.12, 서비스 모듈 import(torch 없음), systemd 실행 경로, BE 없이 기동해 BE 요청 단계까지 가는지.
# 마지막 줄에 이 패키지가 BE에 보낼 생성 설정 해시(configHash)를 출력한다. taxonomy 승인 등록에 쓴다.
# 사용법: scripts/check-package.sh <패키지 파일>
set -euo pipefail

PKG="${1:?사용법: scripts/check-package.sh <패키지 파일>}"
IMAGE="ubuntu:26.04@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
tar -C "$WORK" -xzf "$PKG"

docker run --rm -i --platform linux/arm64 --read-only --tmpfs /tmp --user 65534:65534 \
  -e CKING_CONTENT_TAG_ACCESS_TOKEN=check -e OPENAI_API_KEY=check -e DEEPINFRA_API_KEY=check \
  -v "$WORK/opt/cking-llm:/opt/cking-llm:ro" -w /opt/cking-llm "$IMAGE" \
  /opt/cking-llm/.venv/bin/python - <<'PY'
import importlib
import importlib.util
import json
import pathlib
import platform
import sys


def check(name, ok, detail=""):
    if not ok:
        print(f"실패: {name} {detail}")
        sys.exit(1)
    print(f"통과: {name} {detail}".rstrip())


check("Python 3.12", sys.version_info[:2] == (3, 12), platform.python_version())

for unit in ("cking-content-event.service", "cking-content-daily.service"):
    text = pathlib.Path("deploy/content-worker", unit).read_text()
    check(f"{unit} 실행 경로", "ExecStart=/opt/cking-llm/.venv/bin/python " in text and "WorkingDirectory=/opt/cking-llm" in text)

modules = ["src.recommendation.content_worker", "src.recommendation.daily_cli", "src.recommendation.interest_runtime",
           "src.recommendation.runtime", "src.recommendation.service"]
for module in modules:
    importlib.import_module(module)
heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "sentence_transformers", "transformers"))
check("서비스 모듈 import, 무거운 모듈 없음", not heavy, ",".join(heavy))
check("torch 미설치", importlib.util.find_spec("torch") is None)

import src.recommendation.content_worker as worker

argv = ["--state", "/tmp/state", "--be-base-url", "http://127.0.0.1:9", "--source", "event",
        "--approve-external-transfer", "--max-daily-calls", "1", "--", "apply"]
code = worker.main(argv)
summary = json.loads(pathlib.Path("/tmp/state/summary.json").read_text())
check("BE 없이 기동하면 BE 요청 단계에서 실패", code == 1 and summary.get("code") == "BACKEND_REQUEST_FAILED", summary.get("code", ""))

# 실행 스크립트의 기본값(모델, taxonomy 경로) 그대로 작업자를 만들고, BE에 보내기 전에 생성 설정을 꺼낸다.
captured = {}


def capture(self, source):
    captured["identity"] = self.identity
    return {"failureCount": 0}


worker.ContentWorker.run = capture
worker.main(argv)
identity = captured["identity"]
check("생성 설정", bool(identity), f"model={identity['modelVersion']} taxonomy={identity['taxonomyVersion']}")
print(f"configHash={worker.digest(identity)}")
PY
