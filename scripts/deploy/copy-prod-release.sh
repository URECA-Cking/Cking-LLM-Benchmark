#!/usr/bin/env bash
# CI 운영 배포의 운영 배포 버킷 복사 단계.
# 버전 ID는 개발과 같게 둔다. 이미 복사해 둔 버전이면(다시 실행) 복사하지 않는다. 복사본은 내려받아 sha256으로 확인한다.
# 원본 태그·메타데이터는 복사하지 않는다(기본값은 태그 조회 권한 s3:GetObjectTagging이 필요하다).
# 환경 변수: RELEASE, RELEASE_PREFIX, PROD_RELEASE_PREFIX
set -e

if ! aws s3 ls "$PROD_RELEASE_PREFIX/$RELEASE/cking-llm.tar.gz.sha256" >/dev/null; then
  for f in cking-llm.tar.gz cking-llm.tar.gz.sha256; do
    aws s3 cp "$RELEASE_PREFIX/$RELEASE/$f" "$PROD_RELEASE_PREFIX/$RELEASE/$f" --copy-props none --only-show-errors
  done
fi
mkdir -p "$RUNNER_TEMP/prod"
for f in cking-llm.tar.gz cking-llm.tar.gz.sha256; do
  aws s3 cp "$PROD_RELEASE_PREFIX/$RELEASE/$f" "$RUNNER_TEMP/prod/$f" --only-show-errors
done
(cd "$RUNNER_TEMP/prod" && sha256sum -c --quiet cking-llm.tar.gz.sha256)
echo "운영 패키지: $PROD_RELEASE_PREFIX/$RELEASE/" | tee -a "$GITHUB_STEP_SUMMARY"
