"""내용 캐시와 분리한 적용 번호의 영속 발급 및 체크포인트 바인딩."""
from __future__ import annotations
import json
from pathlib import Path
from src.recommendation.batch import _write_json_atomic
from src.recommendation.locking import file_lock

MAX_SEQUENCE = 9223372036854775807

def validate_sequence(value):
    if type(value) is not int or not 1 <= value <= MAX_SEQUENCE:
        raise ValueError("applicationSequence는 양의 signed 64-bit 정수여야 합니다.")
    return value

def allocate_sequence(path: Path, target: str) -> int:
    # 모든 조정자가 같은 ledger를 사용해야 한다. 발급 후 중단된 번호는 재사용하지 않는다.
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        previous = state.get(target, 0)
        if type(previous) is not int or not 0 <= previous < MAX_SEQUENCE:
            raise ValueError("적용 번호 ledger가 손상되었거나 범위를 소진했습니다.")
        sequence = previous + 1
        state[target] = sequence
        _write_json_atomic(path, state)
        return sequence

def bind_application(checkpoint, target, sequence, records_key):
    if (checkpoint.get("applyTarget"), checkpoint.get("applicationSequence")) == (target, sequence):
        return
    for record in checkpoint[records_key].values():
        record["applyStatus"] = "pending"
        if record.get("error", {}).get("stage") == "apply":
            record.pop("error", None)
    checkpoint.update(applyTarget=target, applicationSequence=sequence)
