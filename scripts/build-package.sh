#!/usr/bin/env bash
# 콘텐츠 작업자 배포 패키지를 만든다. 서버(Ubuntu 26.04, Python 3.14)와 관계없이 검증된 Python 3.12로 돌도록
# 독립 실행형 Python 3.12와 가상환경, 코드(src, data, deploy)를 /opt/cking-llm에 묶는다. 서버는 이 파일을 /에 푼다.
# 사용법: scripts/build-package.sh <릴리스 ID> <출력 폴더>  (Linux arm64, uv 필요, 빈 /opt/cking-llm에 쓸 수 있어야 함)
set -euo pipefail

RELEASE="${1:?사용법: scripts/build-package.sh <릴리스 ID> <출력 폴더>}"
OUT="${2:?사용법: scripts/build-package.sh <릴리스 ID> <출력 폴더>}"
PREFIX=/opt/cking-llm
cd "$(dirname "$0")/.."

if [ "$(uname -sm)" != "Linux aarch64" ]; then
  echo "서버와 같은 Linux arm64에서 빌드한다"
  exit 1
fi
if [ -n "$(ls -A "$PREFIX" 2>/dev/null)" ]; then
  echo "$PREFIX가 비어 있어야 한다"
  exit 1
fi

# 가상환경은 만든 경로의 Python을 가리키므로 Python도 같은 경로 아래에 둔다.
export UV_PYTHON_INSTALL_DIR="$PREFIX/python" UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy
uv python install 3.12
uv venv --python 3.12 "$PREFIX/.venv"
uv pip install --python "$PREFIX/.venv/bin/python" --no-deps --compile-bytecode -r requirements-deploy.txt
uv pip check --python "$PREFIX/.venv/bin/python"

tar -cf - --exclude=__pycache__ src data deploy | tar -C "$PREFIX" -xf -
"$PREFIX/.venv/bin/python" -m compileall -q "$PREFIX/src"
PYTHON_VERSION=$("$PREFIX/.venv/bin/python" -c 'import platform; print(platform.python_version())')
printf 'release=%s\ncommit=%s\npython=%s\n' "$RELEASE" "$(git rev-parse HEAD)" "$PYTHON_VERSION" > "$PREFIX/RELEASE"

mkdir -p "$OUT"
tar -C / -czf "$OUT/cking-llm.tar.gz" "${PREFIX#/}"
(cd "$OUT" && sha256sum cking-llm.tar.gz > cking-llm.tar.gz.sha256)
echo "패키지 $RELEASE: Python $PYTHON_VERSION, $(du -h "$OUT/cking-llm.tar.gz" | cut -f1)"
