import subprocess
import sys

import pytest

from src.recommendation.locking import batch_locks, file_lock, BatchAlreadyRunning
from src.recommendation import locking


# Windows venv 실행기는 자식 인터프리터를 띄운다. OS 잠금 종료 검증은 실제 프로세스를
# 직접 실행하고 표준 라이브러리뿐인 잠금 모듈만 읽어 모델 패키지 초기화를 피한다.
PYTHON = getattr(sys, "_base_executable", sys.executable)


def test_another_process_is_rejected_and_lock_is_reusable(tmp_path):
    path = tmp_path / "batch.lock"
    script = """from pathlib import Path
import runpy
import sys
module = runpy.run_path(sys.argv[2])
file_lock, BatchAlreadyRunning = module['file_lock'], module['BatchAlreadyRunning']
try:
    with file_lock(Path(sys.argv[1])): pass
except BatchAlreadyRunning:
    sys.exit(7)
"""
    with file_lock(path):
        result = subprocess.run([PYTHON, "-c", script, str(path), locking.__file__], timeout=15)
        assert result.returncode == 7
    with file_lock(path):
        pass
    assert path.exists()  # 파일을 지우면 inode 분리로 동시 잠금이 가능해진다.


def test_process_termination_releases_os_lock(tmp_path):
    path = tmp_path / "batch.lock"
    script = """from pathlib import Path
import runpy
import sys
file_lock = runpy.run_path(sys.argv[2])['file_lock']
with file_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen([PYTHON, "-c", script, str(path), locking.__file__],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(BatchAlreadyRunning):
            with file_lock(path):
                pass
    finally:
        process.terminate()
        process.wait(timeout=15)
        process.stdin.close()
        process.stdout.close()
    with file_lock(path):
        pass


def test_failed_multi_lock_acquisition_releases_already_acquired_lock(tmp_path):
    output, cache = tmp_path / "a-output", tmp_path / "z-cache.json"
    with file_lock(cache.with_name(f".{cache.name}.lock")):
        with pytest.raises(BatchAlreadyRunning):
            with batch_locks(output, cache):
                pass
        with file_lock(output / ".batch.lock"):
            pass
