#!/usr/bin/env bash
# CI 운영 배포의 운영 추천 배치 서버 교체 단계.
# 개발과 같다. 서버 그룹(URECA-Cking/Cking-Infra#155)이 없으면 건너뛴다.
# 지난 실행이 시작한 교체가 아직 돌고 있으면, 배포 버전 값을 바꾼 뒤 시작한 것은 이어서 지켜보고,
# 바꾸기 전에 시작한 것은 끝나기를 기다린 뒤 새로 교체한다(그 교체의 서버는 이전 패키지를 받았을 수 있다).
# 환경 변수: PROD_ASG_NAME, PROD_RELEASE_PARAMETER
set -e

COUNT=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$PROD_ASG_NAME" \
  --query 'length(AutoScalingGroups)' --output text)
if [ "$COUNT" = 0 ]; then
  echo "운영 추천 배치 서버 그룹이 아직 없어 교체를 건너뛴다" | tee -a "$GITHUB_STEP_SUMMARY"
  exit 0
fi
# 교체가 끝날 때까지(최대 20분) 기다려 마지막 상태를 출력한다. 끝나지 않으면 Timeout, 조회가 실패하면 QueryFailed다.
# 명령 치환 안에서 돌아 set -e가 꺼지므로 조회 실패를 직접 확인한다.
wait_refresh() {
  for _ in $(seq 1 80); do
    STATUS=$(aws autoscaling describe-instance-refreshes --auto-scaling-group-name "$PROD_ASG_NAME" \
      --instance-refresh-ids "$1" --query 'InstanceRefreshes[0].Status' --output text) || { echo QueryFailed; return; }
    case "$STATUS" in
      Successful|Failed|Cancelled|RollbackSuccessful|RollbackFailed) echo "$STATUS"; return ;;
    esac
    sleep 15
  done
  echo Timeout
}
SET_AT=$(aws ssm get-parameter --name "$PROD_RELEASE_PARAMETER" --query Parameter.LastModifiedDate --output text)
# shellcheck disable=SC2016 # 백틱은 JMESPath 문자열이다
ACTIVE=$(aws autoscaling describe-instance-refreshes --auto-scaling-group-name "$PROD_ASG_NAME" \
  --query 'InstanceRefreshes[?Status==`Pending` || Status==`InProgress`] | [0].[InstanceRefreshId,StartTime]' --output text)
read -r ID STARTED <<< "$ACTIVE"
if [ -n "$ID" ] && [ "$ID" != None ] && [ "$(date -d "$STARTED" +%s)" -lt "$(date -d "$SET_AT" +%s)" ]; then
  echo "배포 버전 값을 바꾸기 전에 시작한 교체($ID)가 끝나기를 기다린 뒤 새로 교체한다" | tee -a "$GITHUB_STEP_SUMMARY"
  case "$(wait_refresh "$ID")" in
    Timeout)
      echo "::error::이전 서버 교체가 20분 안에 끝나지 않았다: $ID"
      exit 1 ;;
    QueryFailed)
      echo "::error::이전 서버 교체 상태를 조회하지 못했다: $ID"
      exit 1 ;;
  esac
  ID=None
fi
if [ -z "$ID" ] || [ "$ID" = None ]; then
  ID=$(aws autoscaling start-instance-refresh --auto-scaling-group-name "$PROD_ASG_NAME" \
    --preferences '{"MinHealthyPercentage":0}' --query InstanceRefreshId --output text)
else
  echo "배포 버전 값을 바꾼 뒤 시작한 교체를 지켜본다: $ID" | tee -a "$GITHUB_STEP_SUMMARY"
fi
STATUS=$(wait_refresh "$ID")
case "$STATUS" in
  Successful)
    echo "서버 교체 완료: $ID" | tee -a "$GITHUB_STEP_SUMMARY" ;;
  Timeout)
    echo "::error::서버 교체가 20분 안에 끝나지 않았다: $ID"
    exit 1 ;;
  QueryFailed)
    echo "::error::서버 교체 상태를 조회하지 못했다: $ID"
    exit 1 ;;
  *)
    echo "::error::서버 교체 $STATUS: $ID"
    exit 1 ;;
esac
