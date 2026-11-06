"""운영 배포 job: 올릴 패키지 정하기, 운영 배포 버킷 복사, 배포 버전 갱신, 운영 추천 배치 서버 교체."""

import hashlib

import pytest

from tests.deploy.harness import require_tools

DEV = "s3://dev-bucket/recommendation/releases"
PROD = "s3://prod-bucket/recommendation/releases"
PARAM = "/cking/prod-recommendation/RELEASE"
ASG = "prod-cking-recommendation"
NOT_FOUND = "An error occurred (ParameterNotFound) when calling the GetParameter operation: Parameter /cking/prod-recommendation/RELEASE not found.\n"
PACKAGE = {
    "src/app.py": "app",
    "data/a.json": "{}",
    "deploy/cking.service": "unit",
    "requirements-deploy.txt": "numpy==2.3.0",
    "scripts/build-package.sh": "build",
    "README.md": "readme",
}
ACTIVE_QUERY = "InstanceRefreshes[?Status==`Pending` || Status==`InProgress`] | [0].[InstanceRefreshId,StartTime]"
SET_AT = "2026-10-22T10:46:42.873000+09:00"
AFTER_SET = "2026-10-22T01:46:52+00:00"
BEFORE_SET = "2026-10-22T01:40:00+00:00"

gnu_date = pytest.mark.skipif(bool(require_tools("gnu-date")), reason="GNU date가 필요하다")
sha256sum = pytest.mark.skipif(bool(require_tools("sha256sum")), reason="sha256sum이 필요하다")


def rid(sha, at="20261022081559"):
    return f"{at}-{sha[:12]}"


def dev_build(sb, release, verified=True):
    sb.s3_put(f"{DEV}/{release}/cking-llm.tar.gz", "pkg")
    sb.s3_put(f"{DEV}/{release}/cking-llm.tar.gz.sha256", "hash")
    if verified:
        sb.s3_put(f"{DEV}/{release}/verified", "refresh-1\n")


def pick_env(sha, release=""):
    return {"GITHUB_SHA": sha, "INPUT_RELEASE": release, "RELEASE_PREFIX": DEV, "PROD_RELEASE_PREFIX": PROD}


NO_PACKAGE = "::error::이 커밋과 패키지 내용이 같고 개발 서버 교체에 성공한 개발 패키지가 없다. develop의 개발 배포가 끝난 뒤 main에서 다시 실행한다"


def test_pick_ancestor_with_same_package(sb):
    a = sb.commit(PACKAGE)
    head = sb.commit({"README.md": "changed"})
    dev_build(sb, rid(a))
    r = sb.run("pick-prod-release.sh", pick_env(head))
    assert r.code == 0
    assert r.outputs == {"release": rid(a)}
    assert r.summary == f"올릴 패키지: {DEV}/{rid(a)}/\n"


def test_pick_latest_rebuild_of_same_commit(sb):
    a = sb.commit(PACKAGE)
    for at in ("20261001010000", "20261101010000", "20261015010000"):
        dev_build(sb, rid(a, at))
    r = sb.run("pick-prod-release.sh", pick_env(a))
    assert r.outputs == {"release": rid(a, "20261101010000")}


def test_pick_fails_when_package_changed_without_dev_build(sb):
    a = sb.commit(PACKAGE)
    head = sb.commit({"src/app.py": "changed"})
    dev_build(sb, rid(a))
    r = sb.run("pick-prod-release.sh", pick_env(head))
    assert r.code == 1
    assert NO_PACKAGE in r.stdout
    assert r.outputs == {}


@pytest.mark.parametrize("changed", ["src/app.py", "data/a.json", "deploy/cking.service", "requirements-deploy.txt", "scripts/build-package.sh"])
def test_pick_compares_every_package_input(sb, changed):
    a = sb.commit(PACKAGE)
    head = sb.commit({changed: "changed"})
    dev_build(sb, rid(a))
    r = sb.run("pick-prod-release.sh", pick_env(head))
    assert r.code == 1


def test_pick_ignores_diverged_commit_with_other_package(sb):
    a = sb.commit(PACKAGE)
    sb.git("checkout", "-q", "-b", "other")
    other = sb.commit({"src/app.py": "other"})
    sb.git("checkout", "-q", "develop")
    head = sb.commit({"README.md": "changed"})
    dev_build(sb, rid(a, "20261001010000"))
    dev_build(sb, rid(other, "20261101010000"))
    r = sb.run("pick-prod-release.sh", pick_env(head))
    assert r.outputs == {"release": rid(a, "20261001010000")}


def test_pick_ignores_unknown_commit_and_non_release_keys(sb):
    a = sb.commit(PACKAGE)
    dev_build(sb, rid(a, "20261001010000"))
    dev_build(sb, rid("e" * 40, "20261101010000"))
    sb.s3_put(f"{DEV}/latest/verified", "x")
    sb.s3_put(f"{DEV}/2026-{a[:12]}/verified", "x")
    r = sb.run("pick-prod-release.sh", pick_env(a))
    assert r.outputs == {"release": rid(a, "20261001010000")}


def squash_release(sb):
    """develop 두 커밋을 main에 squash로 합친다. main 커밋은 develop 커밋의 자손이 아니다."""
    base = sb.commit(PACKAGE)
    sb.git("branch", "main")
    d1 = sb.commit({"src/app.py": "v2"})
    d2 = sb.commit({"README.md": "v2"})
    sb.git("checkout", "-q", "main")
    sb.git("checkout", d2, "--", ".")
    main = sb.commit(message="v1.0.0 (#1)")
    assert sb.git("merge-base", d2, main) == base
    return base, d1, d2, main


def test_pick_squash_release_takes_latest_verified_build(sb):
    base, d1, d2, main = squash_release(sb)
    dev_build(sb, rid(base, "20261001010000"))
    dev_build(sb, rid(d1, "20261002010000"))
    dev_build(sb, rid(d2, "20261003010000"))
    r = sb.run("pick-prod-release.sh", pick_env(main))
    assert r.outputs == {"release": rid(d2, "20261003010000")}


def test_pick_skips_latest_build_without_verified(sb):
    base, d1, d2, main = squash_release(sb)
    dev_build(sb, rid(d1, "20261002010000"))
    dev_build(sb, rid(d2, "20261003010000"), verified=False)
    r = sb.run("pick-prod-release.sh", pick_env(main))
    assert r.outputs == {"release": rid(d1, "20261002010000")}


def test_pick_fails_without_verified_same_package(sb):
    base, d1, d2, main = squash_release(sb)
    dev_build(sb, rid(base, "20261001010000"))
    dev_build(sb, rid(d2, "20261003010000"), verified=False)
    r = sb.run("pick-prod-release.sh", pick_env(main))
    assert r.code == 1
    assert NO_PACKAGE in r.stdout


def test_pick_list_error_stops_with_aws_error(sb):
    a = sb.commit(PACKAGE)
    sb.on("s3", "ls", "--recursive", stderr="An error occurred (AccessDenied) when calling the ListObjectsV2 operation: Access Denied\n", code=1)
    r = sb.run("pick-prod-release.sh", pick_env(a))
    assert r.code == 1
    assert "AccessDenied" in r.stderr
    assert "::error::" not in r.stdout
    assert r.outputs == {}


@pytest.mark.parametrize("value", ["v1.0.0", "20261022081559-ABCDEF123456", "2026102208155-0123456789ab", "20261022081559-0123456789abc"])
def test_pick_rollback_rejects_bad_format(sb, value):
    a = sb.commit(PACKAGE)
    r = sb.run("pick-prod-release.sh", pick_env(a, value))
    assert r.code == 1
    assert f"::error::버전 ID는 <빌드 시각 14자리>-<커밋 12자리>다: {value}" in r.stdout
    assert r.calls == []


def test_pick_rollback_fails_when_not_in_prod_bucket(sb):
    a = sb.commit(PACKAGE)
    release = rid(a)
    dev_build(sb, release)
    r = sb.run("pick-prod-release.sh", pick_env(a, release))
    assert r.code == 1
    assert f"::error::운영 배포 버킷에 {release} 가 없다" in r.stdout
    assert r.outputs == {}


def test_pick_rollback_uses_prod_release(sb):
    a = sb.commit(PACKAGE)
    release = rid("1" * 40)
    sb.s3_put(f"{PROD}/{release}/cking-llm.tar.gz", "pkg")
    r = sb.run("pick-prod-release.sh", pick_env(a, release))
    assert r.code == 0
    assert r.outputs == {"release": release, "rollback": "true"}
    assert r.summary == f"운영 되돌리기: {release}\n"
    assert r.aws("--recursive") == []


RELEASE = "20261022081559-0123456789ab"


def packaged(content=b"package bytes"):
    digest = hashlib.sha256(content).hexdigest()
    return content, f"{digest}  cking-llm.tar.gz\n"


def copy_env():
    return {"RELEASE": RELEASE, "RELEASE_PREFIX": DEV, "PROD_RELEASE_PREFIX": PROD}


@sha256sum
def test_copy_first_run_copies_and_verifies(sb):
    tar, sha = packaged()
    sb.s3_put(f"{DEV}/{RELEASE}/cking-llm.tar.gz", tar)
    sb.s3_put(f"{DEV}/{RELEASE}/cking-llm.tar.gz.sha256", sha)
    r = sb.run("copy-prod-release.sh", copy_env())
    assert r.code == 0
    copies = r.aws("cp", "--copy-props")
    assert [c[2:4] for c in copies] == [
        [f"{DEV}/{RELEASE}/cking-llm.tar.gz", f"{PROD}/{RELEASE}/cking-llm.tar.gz"],
        [f"{DEV}/{RELEASE}/cking-llm.tar.gz.sha256", f"{PROD}/{RELEASE}/cking-llm.tar.gz.sha256"],
    ]
    assert all(c[4:7] == ["--copy-props", "none", "--only-show-errors"] for c in copies)
    assert sb.s3_get(f"{PROD}/{RELEASE}/cking-llm.tar.gz") == tar
    assert (sb.runner_temp / "prod" / "cking-llm.tar.gz").read_bytes() == tar
    assert r.summary == f"운영 패키지: {PROD}/{RELEASE}/\n"


@sha256sum
def test_copy_rerun_only_verifies(sb):
    tar, sha = packaged()
    sb.s3_put(f"{PROD}/{RELEASE}/cking-llm.tar.gz", tar)
    sb.s3_put(f"{PROD}/{RELEASE}/cking-llm.tar.gz.sha256", sha)
    r = sb.run("copy-prod-release.sh", copy_env())
    assert r.code == 0
    assert r.aws("--copy-props") == []
    assert len(r.aws("cp")) == 2
    assert r.summary == f"운영 패키지: {PROD}/{RELEASE}/\n"


@sha256sum
def test_copy_sha256_mismatch_fails(sb):
    _, sha = packaged()
    sb.s3_put(f"{DEV}/{RELEASE}/cking-llm.tar.gz", b"broken")
    sb.s3_put(f"{DEV}/{RELEASE}/cking-llm.tar.gz.sha256", sha)
    r = sb.run("copy-prod-release.sh", copy_env())
    assert r.code == 1
    assert "FAILED" in r.stdout + r.stderr
    assert r.summary == ""


def update_env(release, rollback=""):
    return {"PROD_RELEASE_PARAMETER": PARAM, "PROD_ASG_NAME": ASG, "RELEASE": release, "ROLLBACK": rollback}


def prod_deployed(sb, value, last_refresh=None):
    sb.on("ssm", "get-parameter", "Parameter.Value", stdout=value + "\n")
    sb.on("ssm", "get-parameter", "Parameter.LastModifiedDate", stdout=SET_AT + "\n")
    if last_refresh is not None:
        sb.on("describe-instance-refreshes", "--max-records", stdout=last_refresh + "\n")
    sb.on("ssm", "put-parameter")


@gnu_date
def test_update_first_deploy(sb):
    sb.commit(PACKAGE)
    sb.on("ssm", "get-parameter", stderr=NOT_FOUND, code=254)
    sb.on("ssm", "put-parameter")
    r = sb.run("update-prod-release.sh", update_env(RELEASE))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert r.aws("put-parameter")[0][-3:] == ["--value", RELEASE, "--overwrite"]
    assert r.summary == f"운영 배포 버전: 없음 -> {RELEASE}\n"


@gnu_date
def test_update_same_release_refreshed_after_set_skips(sb):
    prod_deployed(sb, RELEASE, f"Successful\t{AFTER_SET}")
    r = sb.run("update-prod-release.sh", update_env(RELEASE))
    assert r.code == 0
    assert r.outputs == {"updated": "false"}
    assert r.summary == f"이미 {RELEASE} 이 배포되어 있다\n"
    assert r.aws("put-parameter") == []


@gnu_date
@pytest.mark.parametrize("last, shown", [
    (f"Failed\t{AFTER_SET}", "Failed"),
    (f"Successful\t{BEFORE_SET}", "Successful"),
    ("None", "None"),
])
def test_update_same_release_without_later_success_refreshes_again(sb, last, shown):
    prod_deployed(sb, RELEASE, last)
    r = sb.run("update-prod-release.sh", update_env(RELEASE))
    assert r.code == 0
    assert f"::warning::배포 버전은 이미 {RELEASE} 이지만 그 뒤 서버 교체가 성공하지 않아({shown}) 다시 교체한다" in r.stdout
    assert r.outputs == {"updated": "true"}
    assert r.aws("put-parameter") == []


@gnu_date
def test_update_deployed_ancestor_updates(sb):
    old = sb.commit(PACKAGE)
    new = sb.commit({"src/app.py": "v2"})
    prod_deployed(sb, rid(old))
    r = sb.run("update-prod-release.sh", update_env(rid(new, "20261023000000")))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert r.summary == f"운영 배포 버전: {rid(old)} -> {rid(new, '20261023000000')}\n"


@gnu_date
def test_update_same_commit_rebuild_updates(sb):
    a = sb.commit(PACKAGE)
    prod_deployed(sb, rid(a, "20261001010000"))
    r = sb.run("update-prod-release.sh", update_env(rid(a, "20261101010000")))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert len(r.aws("put-parameter")) == 1


@gnu_date
def test_update_newer_deployed_skips(sb):
    old = sb.commit(PACKAGE)
    new = sb.commit({"src/app.py": "v2"})
    prod_deployed(sb, rid(new))
    r = sb.run("update-prod-release.sh", update_env(rid(old, "20261023000000")))
    msg = f"더 새 커밋의 버전 {rid(new)} 이 배포되어 있어 배포 버전을 바꾸지 않는다"
    assert r.code == 0
    assert f"::warning::{msg}" in r.stdout
    assert r.summary == msg + "\n"
    assert r.outputs == {"updated": "false"}
    assert r.aws("put-parameter") == []


@gnu_date
def test_update_diverged_fails(sb):
    sb.commit(PACKAGE)
    sb.git("checkout", "-q", "-b", "other")
    other = sb.commit({"src/app.py": "other"})
    sb.git("checkout", "-q", "develop")
    mine = sb.commit({"src/app.py": "mine"})
    prod_deployed(sb, rid(other))
    r = sb.run("update-prod-release.sh", update_env(rid(mine, "20261023000000")))
    assert r.code == 1
    assert f"::error::배포된 버전 {rid(other)} 과 이 패키지의 이력이 갈라졌다" in r.stdout
    assert r.aws("put-parameter") == []


@gnu_date
def test_update_unknown_deployed_commit_fails(sb):
    a = sb.commit(PACKAGE)
    prod_deployed(sb, rid("d" * 40))
    r = sb.run("update-prod-release.sh", update_env(rid(a)))
    assert r.code == 1
    assert f"::error::배포된 버전 {rid('d' * 40)} 의 커밋을 확인할 수 없다" in r.stdout
    assert r.aws("put-parameter") == []


@gnu_date
@pytest.mark.parametrize("deployed", ["newer", "unknown"])
def test_update_rollback_ignores_order(sb, deployed):
    old = sb.commit(PACKAGE)
    new = sb.commit({"src/app.py": "v2"})
    current = rid(new) if deployed == "newer" else rid("d" * 40)
    prod_deployed(sb, current)
    r = sb.run("update-prod-release.sh", update_env(rid(old, "20261001010000"), rollback="true"))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert r.summary == f"운영 배포 버전: {current} -> {rid(old, '20261001010000')}\n"


@gnu_date
def test_update_ssm_error_fails_with_aws_message(sb):
    err = "An error occurred (AccessDeniedException) when calling the GetParameter operation: denied\n"
    sb.on("ssm", "get-parameter", stderr=err, code=254)
    r = sb.run("update-prod-release.sh", update_env(RELEASE))
    assert r.code == 1
    assert err in r.stdout
    assert r.aws("put-parameter") == []


def prod_group(sb, active="None", count="1"):
    sb.on("describe-auto-scaling-groups", stdout=count + "\n")
    sb.on("ssm", "get-parameter", "Parameter.LastModifiedDate", stdout=SET_AT + "\n")
    sb.on("describe-instance-refreshes", ACTIVE_QUERY, stdout=active + "\n")
    sb.on("start-instance-refresh", stdout="new-1\n")


def refresh_env():
    return {"PROD_ASG_NAME": ASG, "PROD_RELEASE_PARAMETER": PARAM}


def statuses(*values):
    return [{"stdout": v + "\n"} for v in values]


@gnu_date
def test_prod_refresh_skips_without_group(sb):
    prod_group(sb, count="0")
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 0
    assert r.summary == "운영 추천 배치 서버 그룹이 아직 없어 교체를 건너뛴다\n"
    assert r.aws("start-instance-refresh") == []


@gnu_date
def test_prod_refresh_starts_new_when_none_active(sb):
    prod_group(sb)
    sb.seq("describe-instance-refreshes", "new-1", responses=statuses("Pending", "InProgress", "Successful"))
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 0
    assert len(r.aws("start-instance-refresh")) == 1
    assert r.sleeps == [["15"], ["15"]]
    assert r.summary == "서버 교체 완료: new-1\n"


@gnu_date
def test_prod_refresh_follows_refresh_started_after_set(sb):
    prod_group(sb, f"old-1\t{AFTER_SET}")
    sb.seq("describe-instance-refreshes", "old-1", responses=statuses("InProgress", "Successful"))
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 0
    assert r.aws("start-instance-refresh") == []
    assert r.summary == "배포 버전 값을 바꾼 뒤 시작한 교체를 지켜본다: old-1\n서버 교체 완료: old-1\n"


@gnu_date
@pytest.mark.parametrize("previous", ["Successful", "Failed", "Cancelled"])
def test_prod_refresh_waits_for_refresh_started_before_set(sb, previous):
    prod_group(sb, f"old-1\t{BEFORE_SET}")
    sb.seq("describe-instance-refreshes", "old-1", responses=statuses("InProgress", previous))
    sb.on("describe-instance-refreshes", "new-1", stdout="Successful\n")
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 0
    assert len(r.aws("describe-instance-refreshes", "old-1")) == 2
    assert len(r.aws("start-instance-refresh")) == 1
    assert r.summary == (
        "배포 버전 값을 바꾸기 전에 시작한 교체(old-1)가 끝나기를 기다린 뒤 새로 교체한다\n"
        "서버 교체 완료: new-1\n"
    )


@gnu_date
def test_prod_refresh_previous_timeout_fails(sb):
    prod_group(sb, f"old-1\t{BEFORE_SET}")
    sb.on("describe-instance-refreshes", "old-1", stdout="InProgress\n")
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 1
    assert "::error::이전 서버 교체가 20분 안에 끝나지 않았다: old-1" in r.stdout
    assert len(r.sleeps) == 80
    assert r.aws("start-instance-refresh") == []


@gnu_date
@pytest.mark.parametrize("status", ["Failed", "Cancelled", "RollbackSuccessful", "RollbackFailed"])
def test_prod_refresh_new_refresh_failure_fails(sb, status):
    prod_group(sb)
    sb.on("describe-instance-refreshes", "new-1", stdout=status + "\n")
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 1
    assert f"::error::서버 교체 {status}: new-1" in r.stdout
    assert r.summary == ""


@gnu_date
def test_prod_refresh_new_refresh_timeout_fails(sb):
    prod_group(sb)
    sb.on("describe-instance-refreshes", "new-1", stdout="InProgress\n")
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 1
    assert "::error::서버 교체가 20분 안에 끝나지 않았다: new-1" in r.stdout
    assert len(r.sleeps) == 80


@gnu_date
def test_prod_refresh_poll_error_fails(sb):
    prod_group(sb)
    sb.on("describe-instance-refreshes", "new-1", stderr="An error occurred (Throttling)\n", code=254)
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 1
    assert len(r.aws("describe-instance-refreshes", "new-1")) == 1
    assert r.sleeps == []
    assert "::error::서버 교체 상태를 조회하지 못했다: new-1" in r.stdout


@gnu_date
def test_prod_refresh_previous_poll_error_fails(sb):
    prod_group(sb, f"old-1\t{BEFORE_SET}")
    sb.on("describe-instance-refreshes", "old-1", stderr="An error occurred (Throttling)\n", code=254)
    r = sb.run("refresh-prod-servers.sh", refresh_env())
    assert r.code == 1
    assert "::error::이전 서버 교체 상태를 조회하지 못했다: old-1" in r.stdout
    assert r.aws("start-instance-refresh") == []
