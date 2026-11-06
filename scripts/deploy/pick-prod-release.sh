#!/usr/bin/env bash
# CI 운영 배포의 올릴 패키지 정하기 단계.
# 개발 서버 교체에 성공한(verified) 개발 패키지 중 패키지에 들어가는 파일이 이 커밋과 같은 것을 고르고, 여럿이면 가장 최근 빌드를
# 고른다(매월 재빌드의 Python 최신 패치). develop→main은 squash로 합쳐 개발 커밋이 main의 조상이 아니므로 이력이 아니라 내용으로 고른다.
# 없으면 develop의 개발 배포가 끝나기 전에 합친 것이라 멈춘다.
# 환경 변수: INPUT_RELEASE(되돌리기), RELEASE_PREFIX, PROD_RELEASE_PREFIX
set -e

if [ -n "$INPUT_RELEASE" ]; then
  if ! [[ "$INPUT_RELEASE" =~ ^[0-9]{14}-[0-9a-f]{12}$ ]]; then
    echo "::error::버전 ID는 <빌드 시각 14자리>-<커밋 12자리>다: $INPUT_RELEASE"
    exit 1
  fi
  if ! aws s3 ls "$PROD_RELEASE_PREFIX/$INPUT_RELEASE/cking-llm.tar.gz" >/dev/null; then
    echo "::error::운영 배포 버킷에 $INPUT_RELEASE 가 없다"
    exit 1
  fi
  echo "release=$INPUT_RELEASE" >> "$GITHUB_OUTPUT"
  echo "rollback=true" >> "$GITHUB_OUTPUT"
  echo "운영 되돌리기: $INPUT_RELEASE" | tee -a "$GITHUB_STEP_SUMMARY"
  exit 0
fi
INPUTS=(src data deploy requirements-deploy.txt scripts/build-package.sh)
LIST=$(aws s3 ls --recursive "$RELEASE_PREFIX/")
BEST=""
while read -r _ _ _ KEY; do
  [[ "$KEY" =~ /([0-9]{14}-[0-9a-f]{12})/verified$ ]] || continue
  ID=${BASH_REMATCH[1]}
  T=${ID##*-}
  git rev-parse -q --verify "$T^{commit}" >/dev/null || continue
  git diff --quiet "$T" "$GITHUB_SHA" -- "${INPUTS[@]}" || continue
  if [[ -z "$BEST" || "$ID" > "$BEST" ]]; then
    BEST="$ID"
  fi
done <<< "$LIST"
if [ -z "$BEST" ]; then
  echo "::error::이 커밋과 패키지 내용이 같고 개발 서버 교체에 성공한 개발 패키지가 없다. develop의 개발 배포가 끝난 뒤 main에서 다시 실행한다"
  exit 1
fi
echo "release=$BEST" >> "$GITHUB_OUTPUT"
echo "올릴 패키지: $RELEASE_PREFIX/$BEST/" | tee -a "$GITHUB_STEP_SUMMARY"
