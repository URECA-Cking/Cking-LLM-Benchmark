"""배포 스크립트(scripts/deploy) 실행 환경: PATH 앞의 가짜 aws·sleep·docker, 임시 git 저장소, GitHub Actions 출력 파일."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "deploy"
FAKE_NAMES = ("aws", "sleep", "docker")

FAKE = r'''#!__PYTHON__
import fcntl, json, os, pathlib, sys

state = pathlib.Path(os.environ["FAKE_STATE"])
prog = os.path.basename(sys.argv[0])
args = sys.argv[1:]
stdin = None
if (prog == "aws" and args[:2] == ["s3", "cp"] and "-" in args) or (prog == "docker" and "--password-stdin" in args):
    stdin = sys.stdin.read()

with open(state / "lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    rules = json.loads((state / "rules.json").read_text())
    counts_file = state / "counts.json"
    counts = json.loads(counts_file.read_text()) if counts_file.exists() else {}
    hit = None
    for n, rule in enumerate(rules):
        if rule["prog"] == prog and all(t in args for t in rule["match"]):
            used = counts.get(str(n), 0)
            hit = rule["responses"][min(used, len(rule["responses"]) - 1)]
            counts[str(n)] = used + 1
            counts_file.write_text(json.dumps(counts))
            break
    with open(state / "calls.jsonl", "a") as log:
        log.write(json.dumps({"prog": prog, "args": args, "stdin": stdin, "rule": hit is not None}, ensure_ascii=False) + "\n")


def reply(stdout="", stderr="", code=0):
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    sys.exit(code)


if hit is not None:
    reply(**hit)
if prog in ("sleep", "docker"):
    reply()
if prog != "aws" or args[:1] != ["s3"]:
    with open(state / "unmatched.jsonl", "a") as log:
        log.write(json.dumps(args, ensure_ascii=False) + "\n")
    reply(stderr="가짜 aws: 규칙 없음 " + " ".join(args) + "\n", code=254)

root = state / "s3"


def local(uri):
    return root / uri[len("s3://"):]


def line(size, name):
    return "2026-10-22 08:15:59 %10d %s\n" % (size, name)


sub, rest = args[1], args[2:]
pos = []
i = 0
while i < len(rest):
    if rest[i] == "--copy-props":
        i += 2
        continue
    if not rest[i].startswith("--"):
        pos.append(rest[i])
    i += 1

if sub == "cp":
    src, dst = pos
    if src == "-":
        data = stdin.encode()
    elif src.startswith("s3://"):
        if not local(src).is_file():
            reply(stderr="fatal error: An error occurred (404) when calling the HeadObject operation: Key \"%s\" does not exist\n" % src, code=1)
        data = local(src).read_bytes()
    else:
        data = pathlib.Path(src).read_bytes()
    target = local(dst) if dst.startswith("s3://") else pathlib.Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    reply()

if sub == "ls":
    bucket, _, prefix = pos[0][len("s3://"):].partition("/")
    base = root / bucket
    keys = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
    keys = [k for k in keys if k.startswith(prefix)]
    out = []
    if "--recursive" in rest:
        out = [line((base / k).stat().st_size, k) for k in keys]
    else:
        head = prefix[: prefix.rfind("/") + 1]
        for k in keys:
            tail = k[len(head):]
            if "/" in tail:
                entry = "                           PRE %s/\n" % tail.split("/")[0]
                if entry not in out:
                    out.append(entry)
            else:
                out.append(line((base / k).stat().st_size, tail))
    reply("".join(out), code=0 if out else 1)

reply(stderr="가짜 aws: 지원하지 않는 s3 명령 " + sub + "\n", code=254)
'''

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def has_gnu_date() -> bool:
    return subprocess.run(["date", "-d", "@0", "+%s"], capture_output=True).returncode == 0


@dataclass
class Result:
    code: int
    stdout: str
    stderr: str
    outputs: dict[str, str]
    summary: str
    calls: list[dict] = field(default_factory=list)

    def aws(self, *tokens: str) -> list[list[str]]:
        return [c["args"] for c in self.calls if c["prog"] == "aws" and all(t in c["args"] for t in tokens)]

    def docker(self, *tokens: str) -> list[list[str]]:
        return [c["args"] for c in self.calls if c["prog"] == "docker" and all(t in c["args"] for t in tokens)]

    @property
    def sleeps(self) -> list[list[str]]:
        return [c["args"] for c in self.calls if c["prog"] == "sleep"]


class Sandbox:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.bin = self.root / "bin"
        self.state = self.root / "state"
        self.repo = self.root / "repo"
        self.runner_temp = self.root / "runner"
        for d in (self.bin, self.state, self.state / "s3", self.repo, self.runner_temp):
            d.mkdir(parents=True, exist_ok=True)
        fake = self.bin / "fake.py"
        fake.write_text(FAKE.replace("__PYTHON__", sys.executable), encoding="utf-8")
        fake.chmod(0o755)
        for name in FAKE_NAMES:
            (self.bin / name).symlink_to(fake)
        self.rules: list[dict] = []
        self._time = 1_790_000_000
        self.git("init", "-q", "-b", "develop")

    def git(self, *args: str, env: dict | None = None) -> str:
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=self.repo, env={**os.environ, **GIT_ENV, **(env or {})},
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(self, files: dict[str, str | None] | None = None, message: str = "c") -> str:
        for name, content in (files or {}).items():
            path = self.repo / name
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.git("add", "-A")
        self._time += 60
        when = f"{self._time} +0000"
        self.git("commit", "-q", "--allow-empty", "-m", message, env={"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})
        return self.git("rev-parse", "HEAD")

    def on(self, *match: str, stdout: str = "", stderr: str = "", code: int = 0, prog: str = "aws") -> None:
        self.seq(*match, prog=prog, responses=[dict(stdout=stdout, stderr=stderr, code=code)])

    def seq(self, *match: str, responses: list[dict], prog: str = "aws") -> None:
        """규칙 하나에 차례로 돌려줄 응답들. 다 쓰면 마지막 응답을 계속 돌려준다. 먼저 넣은 규칙이 우선이다."""
        self.rules.append({"prog": prog, "match": list(match), "responses": responses})

    def s3_put(self, uri: str, data: bytes | str) -> None:
        path = self.state / "s3" / uri[len("s3://"):]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)

    def s3_get(self, uri: str) -> bytes | None:
        path = self.state / "s3" / uri[len("s3://"):]
        return path.read_bytes() if path.is_file() else None

    def run(self, script: str, env: dict[str, str], allow_unmatched: bool = False) -> Result:
        (self.state / "rules.json").write_text(json.dumps(self.rules, ensure_ascii=False), encoding="utf-8")
        for name in ("counts.json", "calls.jsonl", "unmatched.jsonl"):
            (self.state / name).unlink(missing_ok=True)
        output = self.root / "github_output"
        summary = self.root / "step_summary"
        output.write_text("")
        summary.write_text("")
        full_env = {
            **os.environ, **GIT_ENV,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_STATE": str(self.state),
            "RUNNER_TEMP": str(self.runner_temp),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
            **env,
        }
        proc = subprocess.run(
            [str(SCRIPTS / script)], cwd=self.repo, env=full_env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )
        unmatched = self.state / "unmatched.jsonl"
        if unmatched.exists() and not allow_unmatched:
            raise AssertionError(f"규칙 없는 aws 호출:\n{unmatched.read_text()}\nstderr:\n{proc.stderr}")
        outputs = {}
        for item in output.read_text().splitlines():
            key, _, value = item.partition("=")
            outputs[key] = value
        calls_file = self.state / "calls.jsonl"
        calls = [json.loads(x) for x in calls_file.read_text().splitlines()] if calls_file.exists() else []
        return Result(proc.returncode, proc.stdout, proc.stderr, outputs, summary.read_text(), calls)


def require_tools(*tools: str) -> str | None:
    """없는 도구가 있으면 건너뛸 이유를 돌려준다. CI(ubuntu)에는 모두 있다."""
    for tool in tools:
        if tool == "gnu-date":
            if not has_gnu_date():
                return "GNU date가 필요하다"
        elif shutil.which(tool) is None:
            return f"{tool}가 필요하다"
    return None
