"""내용 캐시와 분리한 적용 번호의 영속 발급 및 체크포인트 바인딩."""
from __future__ import annotations
import json
from pathlib import Path
from src.recommendation.storage import _write_json_atomic
from src.recommendation.locking import file_lock

MAX_SEQUENCE = 9223372036854775807

def validate_sequence(value):
    if type(value) is not int or not 1 <= value <= MAX_SEQUENCE:
        raise ValueError("applicationSequence는 양의 signed 64-bit 정수여야 합니다.")
    return value

def _read_ledger(path: Path, target: str, known_sequence=None):
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(state, dict):
        raise ValueError("적용 번호 ledger가 손상되었습니다.")
    previous = state.get(target, 0)
    if type(previous) is not int or not 0 <= previous <= MAX_SEQUENCE:
        raise ValueError("적용 번호 ledger가 손상되었습니다.")
    if known_sequence is not None and previous < validate_sequence(known_sequence):
        raise ValueError("적용 ledger가 유실되었거나 체크포인트보다 뒤처졌습니다. BE 최대 번호를 확인하고 복구하세요.")
    return state, previous


def _allocate_locked(path: Path, target: str, state, previous: int) -> int:
    if previous == MAX_SEQUENCE:
        raise ValueError("적용 번호 ledger의 범위를 소진했습니다.")
    sequence = previous + 1
    state[target] = sequence
    _write_json_atomic(path, state)
    return sequence


def allocate_sequence(path: Path, target: str, *, known_sequence=None) -> int:
    # 모든 조정자가 같은 ledger를 사용해야 한다. 발급 후 중단된 번호는 재사용하지 않는다.
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        state, previous = _read_ledger(path, target, known_sequence)
        return _allocate_locked(path, target, state, previous)


def resolve_sequence(path: Path | None, target: str, checkpoint, explicit=None) -> int:
    """공유 ledger를 확인하고 명시 번호, 같은 대상의 재개 번호, 신규 번호 순으로 결정한다."""
    if path is None:
        raise ValueError("apply에는 공유 sequence_ledger 경로가 필요합니다.")
    saved = checkpoint.get("applicationSequence")
    same_target = checkpoint.get("applyTarget") == target
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        # 다른 대상으로 전환해도 과거 적용 이력이 있으면 사라진 ledger를 새로 만들지 않는다.
        if saved is not None and not path.exists():
            raise ValueError("적용 ledger가 유실되었습니다. BE 최대 번호를 확인하고 복구하세요.")
        state, previous = _read_ledger(path, target, saved if same_target else None)
        sequence = explicit if explicit is not None else (saved if same_target else None)
        if sequence is None:
            return _allocate_locked(path, target, state, previous)
        sequence = validate_sequence(sequence)
        if previous < sequence:
            raise ValueError("적용 번호가 공유 ledger에서 발급되지 않았습니다.")
        return sequence


def reset_apply_records(records):
    for record in records.values():
        if not isinstance(record, dict):
            continue
        record["applyStatus"] = "pending"
        error = record.get("error")
        if isinstance(error, dict) and error.get("stage") == "apply":
            record.pop("error", None)


def bind_application(checkpoint, target, sequence, records_key):
    if (checkpoint.get("applyTarget"), checkpoint.get("applicationSequence")) == (target, sequence):
        return
    reset_apply_records(checkpoint[records_key])
    checkpoint.update(applyTarget=target, applicationSequence=sequence)
