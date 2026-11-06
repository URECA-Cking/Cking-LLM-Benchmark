"""개발 배포 job: 배포 버전 갱신(update-dev-release.sh), 개발 추천 배치 서버 교체(refresh-dev-servers.sh)."""

import pytest

PARAM = "/cking/dev-recommendation/RELEASE"
PREFIX = "s3://dev-bucket/recommendation/releases"
ASG = "dev-cking-recommendation"
NOT_FOUND = "An error occurred (ParameterNotFound) when calling the GetParameter operation: Parameter /cking/dev-recommendation/RELEASE not found.\n"


def rid(sha, at="20261022081559"):
    return f"{at}-{sha[:12]}"


def release_env(sha, release):
    return {"RELEASE_PARAMETER": PARAM, "RELEASE": release, "GITHUB_SHA": sha}


def deployed(sb, value):
    sb.on("ssm", "get-parameter", stdout=value + "\n")
    sb.on("ssm", "put-parameter")


def assert_no_put(r):
    assert r.aws("put-parameter") == []


def test_first_deploy_creates_release(sb):
    sha = sb.commit({"src/a.py": "1"})
    sb.on("ssm", "get-parameter", stderr=NOT_FOUND, code=254)
    sb.on("ssm", "put-parameter")
    new = rid(sha, "20261023000000")
    r = sb.run("update-dev-release.sh", release_env(sha, new))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert r.aws("put-parameter") == [["ssm", "put-parameter", "--name", PARAM, "--type", "String", "--value", new, "--overwrite"]]
    assert r.summary == f"배포 버전: 없음 -> {new}\n"


def test_deployed_ancestor_updates(sb):
    old = sb.commit({"src/a.py": "1"})
    sha = sb.commit({"src/a.py": "2"})
    deployed(sb, rid(old))
    new = rid(sha, "20261023000000")
    r = sb.run("update-dev-release.sh", release_env(sha, new))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert r.summary == f"배포 버전: {rid(old)} -> {new}\n"


def test_same_commit_rebuild_updates(sb):
    sha = sb.commit({"src/a.py": "1"})
    deployed(sb, rid(sha, "20261001010000"))
    new = rid(sha, "20261101010000")
    r = sb.run("update-dev-release.sh", release_env(sha, new))
    assert r.code == 0
    assert r.outputs == {"updated": "true"}
    assert len(r.aws("put-parameter")) == 1


def test_newer_deployed_skips(sb):
    sha = sb.commit({"src/a.py": "1"})
    newer = sb.commit({"src/a.py": "2"})
    deployed(sb, rid(newer))
    r = sb.run("update-dev-release.sh", release_env(sha, rid(sha, "20261023000000")))
    msg = f"더 새 커밋의 버전 {rid(newer)} 이 배포되어 있어 배포 버전을 바꾸지 않는다"
    assert r.code == 0
    assert f"::warning::{msg}" in r.stdout
    assert r.summary == msg + "\n"
    assert r.outputs == {"updated": "false"}
    assert_no_put(r)


def test_diverged_history_fails(sb):
    base = sb.commit({"src/a.py": "1"})
    sb.git("checkout", "-q", "-b", "other")
    other = sb.commit({"src/a.py": "other"})
    sb.git("checkout", "-q", "develop")
    sha = sb.commit({"src/a.py": "develop"})
    assert base != sha
    deployed(sb, rid(other))
    r = sb.run("update-dev-release.sh", release_env(sha, rid(sha, "20261023000000")))
    assert r.code == 1
    assert f"::error::배포된 버전 {rid(other)} 과 이 커밋의 이력이 갈라졌다" in r.stdout
    assert r.outputs == {}
    assert_no_put(r)


def test_unknown_deployed_commit_fails(sb):
    sha = sb.commit({"src/a.py": "1"})
    unknown = rid("f" * 40)
    deployed(sb, unknown)
    r = sb.run("update-dev-release.sh", release_env(sha, rid(sha, "20261023000000")))
    assert r.code == 1
    assert f"::error::배포된 버전 {unknown} 의 커밋을 확인할 수 없다" in r.stdout
    assert_no_put(r)


def test_ssm_error_fails_with_aws_message(sb):
    sha = sb.commit({"src/a.py": "1"})
    err = "An error occurred (AccessDeniedException) when calling the GetParameter operation: denied\n"
    sb.on("ssm", "get-parameter", stderr=err, code=254)
    r = sb.run("update-dev-release.sh", release_env(sha, rid(sha)))
    assert r.code == 1
    assert err in r.stdout
    assert_no_put(r)


def refresh_env(release="20261023000000-0123456789ab"):
    return {"ASG_NAME": ASG, "RELEASE_PREFIX": PREFIX, "RELEASE": release}


def group(sb, count="1"):
    sb.on("describe-auto-scaling-groups", stdout=count + "\n")
    sb.on("start-instance-refresh", stdout="refresh-1\n")


def test_refresh_skips_without_group(sb):
    group(sb, "0")
    r = sb.run("refresh-dev-servers.sh", refresh_env())
    assert r.code == 0
    assert r.summary == "개발 추천 배치 서버 그룹이 아직 없어 교체를 건너뛴다\n"
    assert r.aws("start-instance-refresh") == []
    assert sb.s3_get(f"{PREFIX}/20261023000000-0123456789ab/verified") is None


def test_refresh_success_leaves_verified(sb):
    group(sb)
    sb.seq("describe-instance-refreshes", "refresh-1", responses=[
        {"stdout": "Pending\n"}, {"stdout": "InProgress\n"}, {"stdout": "Successful\n"},
    ])
    r = sb.run("refresh-dev-servers.sh", refresh_env())
    assert r.code == 0
    assert r.aws("start-instance-refresh")[0][-6:] == [
        "--preferences", '{"MinHealthyPercentage":0}', "--query", "InstanceRefreshId", "--output", "text",
    ]
    assert len(r.aws("describe-instance-refreshes")) == 3
    assert r.sleeps == [["15"], ["15"]]
    assert sb.s3_get(f"{PREFIX}/20261023000000-0123456789ab/verified") == b"refresh-1\n"
    assert r.summary == "서버 교체 완료: refresh-1\n"


@pytest.mark.parametrize("status", ["Failed", "Cancelled", "RollbackSuccessful", "RollbackFailed"])
def test_refresh_failure_leaves_no_verified(sb, status):
    group(sb)
    sb.on("describe-instance-refreshes", "refresh-1", stdout=status + "\n")
    r = sb.run("refresh-dev-servers.sh", refresh_env())
    assert r.code == 1
    assert f"::error::서버 교체 {status}: refresh-1" in r.stdout
    assert sb.s3_get(f"{PREFIX}/20261023000000-0123456789ab/verified") is None
    assert r.summary == ""


def test_refresh_timeout_after_20_minutes(sb):
    group(sb)
    sb.on("describe-instance-refreshes", "refresh-1", stdout="InProgress\n")
    r = sb.run("refresh-dev-servers.sh", refresh_env())
    assert r.code == 1
    assert len(r.aws("describe-instance-refreshes")) == 80
    assert r.sleeps == [["15"]] * 80
    assert "::error::서버 교체가 20분 안에 끝나지 않았다: refresh-1" in r.stdout
    assert sb.s3_get(f"{PREFIX}/20261023000000-0123456789ab/verified") is None


def test_refresh_poll_error_stops_immediately(sb):
    group(sb)
    sb.on("describe-instance-refreshes", "refresh-1", stderr="An error occurred (Throttling)\n", code=254)
    r = sb.run("refresh-dev-servers.sh", refresh_env())
    assert r.code == 254
    assert r.sleeps == []
    assert "::error::" not in r.stdout
