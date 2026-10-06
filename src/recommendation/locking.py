"""프로세스 종료 시 OS가 회수하는 비차단 파일 잠금이다."""

from __future__ import annotations

import os
from contextlib import contextmanager, ExitStack
from pathlib import Path


class BatchAlreadyRunning(RuntimeError):
    """같은 실행 상태나 모델 캐시를 다른 배치가 사용하고 있다."""


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise BatchAlreadyRunning("같은 배치 상태 또는 모델 캐시를 사용하는 실행이 진행 중입니다.") from None
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    # 잠금 파일은 삭제하지 않는다. 삭제하면 대기 프로세스와 새 파일의 inode가 갈라진다.


@contextmanager
def batch_locks(output_dir: Path, cache: Path):
    resolved_cache = cache.resolve()
    paths = {output_dir.resolve() / ".batch.lock",
             resolved_cache.with_name(f".{resolved_cache.name}.lock")}
    with ExitStack() as stack:
        for path in sorted(paths, key=str):
            stack.enter_context(file_lock(path))
        yield
