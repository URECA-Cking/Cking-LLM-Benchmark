"""python -m src.recommendation.e2e.runner --be-dir ../Cking-BE"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parent


class EnvironmentFailure(RuntimeError):
    """Docker·빌드·실행 환경이 준비되지 않았다."""


def command(args, *, env=None, cwd=None, timeout=60):
    # 도구의 stdout/stderr에는 비밀값이나 fixture 원문이 있을 수 있어 그대로 전달하지 않는다.
    try:
        result = subprocess.run(args, env=env, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise EnvironmentFailure("command_unavailable_or_timeout") from None
    if result.returncode:
        raise EnvironmentFailure("command_failed")
    return result.stdout.decode("utf-8", errors="replace").strip()


def clean_environment():
    # 운영 DB·모델 키·Spring 설정을 상속하지 않는다. 실행 도구에 필요한 OS 설정만 허용한다.
    allowed = {"path", "systemroot", "windir", "comspec", "pathext", "temp", "tmp", "home",
               "userprofile", "appdata", "localappdata", "programdata", "java_home", "gradle_user_home",
               "docker_host", "docker_context", "docker_config"}
    return {key: value for key, value in os.environ.items() if key.lower() in allowed}


def git_sha(path):
    return command(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"])


def wait_healthy(name, env, *, deadline_seconds=120):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        state = command(["docker", "inspect", "--format", "{{.State.Health.Status}}", name], env=env)
        if state == "healthy":
            return
        if state == "unhealthy":
            break
        time.sleep(1)
    raise EnvironmentFailure("container_not_ready")


def container_port(name, port, env):
    value = command(["docker", "port", name, f"{port}/tcp"], env=env)
    host, number = value.rsplit(":", 1)
    if host != "127.0.0.1" or not number.isdecimal():
        raise EnvironmentFailure("container_binding_not_loopback")
    return int(number)


def require_local_docker(env):
    endpoint = env.get("DOCKER_HOST") or command(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], env=env)
    # 운영 서버를 쓰지 않는 테스트다. 로컬 Unix socket/Windows named pipe만 허용한다.
    if not endpoint.startswith(("unix://", "npipe://")):
        raise EnvironmentFailure("local_docker_required")


def main(argv=None):
    parser = argparse.ArgumentParser(description="전용 MySQL·Redis와 실제 BE를 실행하는 추천 E2E")
    parser.add_argument("--be-dir", type=Path, default=ROOT.parent / "Cking-BE")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "recommendation" / "e2e")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    be = args.be_dir.resolve()
    output = args.output_dir.resolve() / uuid.uuid4().hex
    output.mkdir(parents=True)
    report_path = output / "report.json"
    report = {"schemaVersion": 1, "status": "environment_failure", "checks": [],
              "externalModelCalls": 0, "cleanupVerified": False}
    env = clean_environment()
    suffix = uuid.uuid4().hex
    names = [f"cking-e2e-mysql-{suffix}", f"cking-e2e-redis-{suffix}"]
    owned = []
    phase = "prerequisites"
    exit_code = 2
    try:
        if args.timeout <= 0 or not (be / "gradlew").is_file():
            raise EnvironmentFailure("be_checkout_or_timeout_invalid")
        report["llmCommitSha"], report["beCommitSha"] = git_sha(ROOT), git_sha(be)
        require_local_docker(env)
        command(["docker", "version", "--format", "{{.Server.Version}}"], env=env)
        password, key = secrets.token_hex(32), secrets.token_hex(32)
        env.update(MYSQL_ROOT_PASSWORD=password, MYSQL_DATABASE=f"cking_e2e_{suffix}",
                   CKING_E2E_DB_PASSWORD=password, CKING_E2E_API_KEY=key,
                   CKING_E2E_KEY_HASH=hashlib.sha256(key.encode()).hexdigest(),
                   CKING_E2E_JWT_SECRET=base64.b64encode(secrets.token_bytes(32)).decode(),
                   CKING_E2E_PYTHON=sys.executable, CKING_E2E_LLM_ROOT=str(ROOT), CKING_E2E_BE_ROOT=str(be),
                   CKING_E2E_JAVA_SOURCE=str(ASSETS / "java"), CKING_E2E_OUTPUT=str(output))
        phase = "containers"
        # 고유 이름은 생성 전에 정해 둔다. 명령이 timeout되어도 finally에서 이 이름만 제거한다.
        owned.append(names[0])
        command(["docker", "run", "-d", "--name", names[0], "--label", f"cking.e2e={suffix}",
                 "-p", "127.0.0.1::3306", "-e", "MYSQL_ROOT_PASSWORD", "-e", "MYSQL_DATABASE",
                 "--health-cmd", 'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysqladmin ping -h 127.0.0.1 --silent',
                 "--health-interval", "2s", "--health-retries", "60", "mysql:8.4",
                 "--default-time-zone=+00:00", "--character-set-server=utf8mb4"], env=env, timeout=300)
        owned.append(names[1])
        command(["docker", "run", "-d", "--name", names[1], "--label", f"cking.e2e={suffix}",
                 "-p", "127.0.0.1::6379", "--health-cmd", "redis-cli ping", "--health-interval", "2s",
                 "--health-retries", "30", "redis:7.2-alpine"], env=env, timeout=300)
        for name in names:
            wait_healthy(name, env)
        mysql_port = container_port(names[0], 3306, env)
        env["CKING_E2E_REDIS_PORT"] = str(container_port(names[1], 6379, env))
        env["CKING_E2E_JDBC"] = (f"jdbc:mysql://127.0.0.1:{mysql_port}/cking_e2e_{suffix}"
                                 "?characterEncoding=UTF-8&connectionTimeZone=UTC&forceConnectionTimeZoneToSession=true")
        phase = "be_build_or_start"
        # Wrapper의 Java 진입점을 직접 호출하면 Windows .bat 인자 해석과 별도 창 생성을 피한다.
        java = str(Path(env["JAVA_HOME"]) / "bin" / "java") if "JAVA_HOME" in env else "java"
        command([java, "-classpath", str(be / "gradle/wrapper/gradle-wrapper.jar"),
                 "org.gradle.wrapper.GradleWrapperMain", "--no-daemon", "--console=plain", "-q",
                 "-I", str(ASSETS / "bridge.gradle"), "recommendationE2e"],
                env=env, cwd=be, timeout=args.timeout)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        exit_code = 0 if report["status"] == "passed" else 1
    except EnvironmentFailure:
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["status"] == "passed":
                report.update(status="verification_failure", failedCheck="be_final_fixture_audit")
            exit_code = 1
        else:
            report["failedPhase"] = phase
    finally:
        cleanup = True
        for name in reversed(owned):
            try:
                # MySQL의 익명 volume도 함께 제거한다. 공용 컨테이너/volume 이름은 받지 않는다.
                command(["docker", "rm", "-f", "-v", name], env=env)
            except EnvironmentFailure:
                cleanup = False
        report["cleanupVerified"] = cleanup
        if not cleanup:
            report.update(status="environment_failure", failedPhase="cleanup")
            exit_code = 2
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[recommendation-e2e] status={report['status']} checks={len(report['checks'])} cleanup={report['cleanupVerified']}")
    print(f"report={report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
