#!/usr/bin/env bash
# CI 개발 배포의 개발 추천 배치 서버 교체 단계.
# 서버 1대가 상태 디스크 하나를 붙여 쓰므로 기존 서버를 먼저 내리고(최소 정상 0%) 새 서버를 띄운다.
# 새 서버는 부팅하며 배포 버전의 패키지를 받는다. 서버 그룹은 URECA-Cking/Cking-Infra#136에서 만들며, 그 전에는 건너뛴다.
# 교체가 성공하면 패키지 옆에 verified를 남긴다. 운영 배포는 이 표시가 있는 패키지만 고른다.
# 환경 변수: ASG_NAME, RELEASE_PREFIX, RELEASE
set -e

COUNT=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG_NAME" \
  --query 'length(AutoScalingGroups)' --output text)
if [ "$COUNT" = 0 ]; then
  echo "개발 추천 배치 서버 그룹이 아직 없어 교체를 건너뛴다" | tee -a "$GITHUB_STEP_SUMMARY"
  exit 0
fi
ID=$(aws autoscaling start-instance-refresh --auto-scaling-group-name "$ASG_NAME" \
  --preferences '{"MinHealthyPercentage":0}' --query InstanceRefreshId --output text)
for _ in $(seq 1 80); do
  STATUS=$(aws autoscaling describe-instance-refreshes --auto-scaling-group-name "$ASG_NAME" \
    --instance-refresh-ids "$ID" --query 'InstanceRefreshes[0].Status' --output text)
  case "$STATUS" in
    Successful)
      printf '%s\n' "$ID" | aws s3 cp - "$RELEASE_PREFIX/$RELEASE/verified" --only-show-errors
      echo "서버 교체 완료: $ID" | tee -a "$GITHUB_STEP_SUMMARY"
      exit 0 ;;
    Failed|Cancelled|RollbackSuccessful|RollbackFailed)
      echo "::error::서버 교체 $STATUS: $ID"
      exit 1 ;;
  esac
  sleep 15
done
echo "::error::서버 교체가 20분 안에 끝나지 않았다: $ID"
exit 1
