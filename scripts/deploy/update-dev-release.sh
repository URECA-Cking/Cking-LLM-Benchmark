#!/usr/bin/env bash
# CI 개발 배포의 배포 버전 갱신 단계.
# 배포된 커밋이 이 커밋의 조상이거나 같으면 바꾸고, 이 커밋이 더 오래됐으면(지난 실행의 재시도) 건너뛴다.
# 둘 다 아니면 이력이 갈라진 것이라 멈춘다. 배포 버전 값이 없으면 첫 배포다.
# 환경 변수: RELEASE_PARAMETER, RELEASE
set -e

CURRENT=""
if CURRENT=$(aws ssm get-parameter --name "$RELEASE_PARAMETER" --query Parameter.Value --output text 2>"$RUNNER_TEMP/ssm.err"); then
  DEPLOYED=${CURRENT##*-}
  if ! git rev-parse -q --verify "$DEPLOYED^{commit}" >/dev/null; then
    echo "::error::배포된 버전 $CURRENT 의 커밋을 확인할 수 없다"
    exit 1
  fi
  if ! git merge-base --is-ancestor "$DEPLOYED" "$GITHUB_SHA"; then
    if git merge-base --is-ancestor "$GITHUB_SHA" "$DEPLOYED"; then
      echo "::warning::더 새 커밋의 버전 $CURRENT 이 배포되어 있어 배포 버전을 바꾸지 않는다"
      echo "더 새 커밋의 버전 $CURRENT 이 배포되어 있어 배포 버전을 바꾸지 않는다" >> "$GITHUB_STEP_SUMMARY"
      echo "updated=false" >> "$GITHUB_OUTPUT"
      exit 0
    fi
    echo "::error::배포된 버전 $CURRENT 과 이 커밋의 이력이 갈라졌다"
    exit 1
  fi
elif ! grep -q ParameterNotFound "$RUNNER_TEMP/ssm.err"; then
  cat "$RUNNER_TEMP/ssm.err"
  exit 1
fi
aws ssm put-parameter --name "$RELEASE_PARAMETER" --type String --value "$RELEASE" --overwrite >/dev/null
echo "updated=true" >> "$GITHUB_OUTPUT"
echo "배포 버전: ${CURRENT:-없음} -> $RELEASE" | tee -a "$GITHUB_STEP_SUMMARY"
