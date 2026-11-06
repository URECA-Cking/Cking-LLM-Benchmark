#!/usr/bin/env bash
# CI 운영 배포의 배포 버전 갱신 단계.
# 개발과 같은 기준을 패키지 커밋에 쓴다. 배포된 패키지의 커밋이 고른 패키지 커밋의 조상이거나 같으면 바꾸고,
# 더 새로우면(지난 실행의 재시도) 건너뛰고, 갈라졌으면 멈춘다. 되돌리기는 순서를 보지 않는다.
# 배포 버전 값은 서버 교체보다 먼저 바뀌므로, 값이 이미 같으면 그 뒤에 시작한 교체가 성공했을 때만 건너뛴다.
# 환경 변수: PROD_RELEASE_PARAMETER, PROD_ASG_NAME, RELEASE, ROLLBACK
set -e

CURRENT=""
if CURRENT=$(aws ssm get-parameter --name "$PROD_RELEASE_PARAMETER" --query Parameter.Value --output text 2>"$RUNNER_TEMP/ssm.err"); then
  if [ "$CURRENT" = "$RELEASE" ]; then
    SET_AT=$(aws ssm get-parameter --name "$PROD_RELEASE_PARAMETER" --query Parameter.LastModifiedDate --output text)
    LAST=$(aws autoscaling describe-instance-refreshes --auto-scaling-group-name "$PROD_ASG_NAME" --max-records 1 \
      --query 'InstanceRefreshes[0].[Status,StartTime]' --output text)
    read -r STATUS STARTED <<< "$LAST"
    if [ "$STATUS" = Successful ] && [ "$(date -d "$STARTED" +%s)" -ge "$(date -d "$SET_AT" +%s)" ]; then
      echo "이미 $RELEASE 이 배포되어 있다" | tee -a "$GITHUB_STEP_SUMMARY"
      echo "updated=false" >> "$GITHUB_OUTPUT"
      exit 0
    fi
    echo "::warning::배포 버전은 이미 $RELEASE 이지만 그 뒤 서버 교체가 성공하지 않아(${STATUS:-없음}) 다시 교체한다"
    echo "updated=true" >> "$GITHUB_OUTPUT"
    exit 0
  fi
  DEPLOYED=${CURRENT##*-}
  T=${RELEASE##*-}
  if [ "$ROLLBACK" != true ]; then
    if ! git rev-parse -q --verify "$DEPLOYED^{commit}" >/dev/null; then
      echo "::error::배포된 버전 $CURRENT 의 커밋을 확인할 수 없다"
      exit 1
    fi
    if ! git merge-base --is-ancestor "$DEPLOYED" "$T"; then
      if git merge-base --is-ancestor "$T" "$DEPLOYED"; then
        echo "::warning::더 새 커밋의 버전 $CURRENT 이 배포되어 있어 배포 버전을 바꾸지 않는다"
        echo "더 새 커밋의 버전 $CURRENT 이 배포되어 있어 배포 버전을 바꾸지 않는다" >> "$GITHUB_STEP_SUMMARY"
        echo "updated=false" >> "$GITHUB_OUTPUT"
        exit 0
      fi
      echo "::error::배포된 버전 $CURRENT 과 이 패키지의 이력이 갈라졌다"
      exit 1
    fi
  fi
elif ! grep -q ParameterNotFound "$RUNNER_TEMP/ssm.err"; then
  cat "$RUNNER_TEMP/ssm.err"
  exit 1
fi
aws ssm put-parameter --name "$PROD_RELEASE_PARAMETER" --type String --value "$RELEASE" --overwrite >/dev/null
echo "updated=true" >> "$GITHUB_OUTPUT"
echo "운영 배포 버전: ${CURRENT:-없음} -> $RELEASE" | tee -a "$GITHUB_STEP_SUMMARY"
