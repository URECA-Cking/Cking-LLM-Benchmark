"""dev 규모 민감도 실험용 합성 크리에이터 300명을 만든다 (이슈 #8).

카테고리당 30명(10명 x 3배치)을 LLM으로 만들어 `data/creators_large.csv`에 쓴다. gold·declared는
모델이 아니라 코드가 정한다 — 모델은 이름·소개·이벤트·세부주제만 쓰고, "이 크리에이터는 어떤
카테고리(들)를 다룬다"는 지시만 받는다. 배치 결과는 `results/cache/large_gen/`에 저장해 재실행 때
API를 다시 부르지 않는다. 사용법: `python3 -m src.generate_large [--model ...] [--dry-run]`
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.config import CACHE_DIR, CREATORS_LARGE_CSV
from src.data import Category, load_categories, load_creators

GENERATOR_MODELS = {
    "gpt-5.4-mini-2026-03-17": {"input_price": 0.75, "output_price": 4.50},
    "gpt-5.4-nano-2026-03-17": {"input_price": 0.20, "output_price": 1.25},
}
DEFAULT_GENERATOR_MODEL = "gpt-5.4-mini-2026-03-17"
BATCH_SIZE = 10
BATCHES_PER_CATEGORY = 3
MAX_ATTEMPTS = 3
ESTIMATED_TOKENS_PER_CALL = (1500, 2500)  # (입력, 출력) 대략치, --dry-run 비용 추정용
SOFT_PREFIX = "[길이] "  # 글자 수만 어긴 경우는 마지막 시도에서 경고만 남기고 받아들인다(스타일 다양성은 부가 목적)
CSV_FIELDS = ["id", "name", "bio", "events", "subtopic", "gold", "written_by", "note", "declared"]

# 배치 안 슬롯 10개의 글 스타일. 짧은 말투부터 이벤트만 있는 경우까지 섞어 문체가 한쪽으로 쏠리지 않게 한다.
STYLE_BY_SLOT = (
    "short_casual", "medium", "long_polished", "short_casual", "medium",
    "long_polished", "short_casual", "events_only", "medium", "long_polished",
)
CROSS_SLOT = 4  # 두 분야를 대등하게 다루는 크리에이터 (gold 2개)
MISDECLARED_SLOT = 9  # 소개는 주 분야인데 본인이 다른 분야로 입력한 크리에이터 (declared만 다름)
# 카테고리별로 자연스럽게 겹치거나 헷갈릴 만한 분야 3개. 배치 번호로 돌려 쓴다.
PARTNERS = {
    "FITNESS": ("FOOD", "TRAVEL", "BEAUTY"),
    "FOOD": ("FITNESS", "TRAVEL", "KNOWLEDGE"),
    "GAME": ("TECH", "MUSIC", "KNOWLEDGE"),
    "BEAUTY": ("FASHION", "FITNESS", "KNOWLEDGE"),
    "FASHION": ("BEAUTY", "TRAVEL", "MUSIC"),
    "TRAVEL": ("FOOD", "FITNESS", "PET"),
    "MUSIC": ("KNOWLEDGE", "FASHION", "GAME"),
    "PET": ("TRAVEL", "FOOD", "KNOWLEDGE"),
    "TECH": ("GAME", "KNOWLEDGE", "FASHION"),
    "KNOWLEDGE": ("TECH", "FITNESS", "FOOD"),
}
STYLE_GUIDE = {
    "short_casual": "bio 15~45자, 구어체·이모지·해시태그 자유. 정보는 적지만 분야는 짐작 가능. events는 빈 문자열",
    "medium": "bio 1~2문장(60~120자). events는 빈 문자열",
    "long_polished": "bio 반드시 3~4문장, 공백 포함 150~260자(80자 미만이면 불합격). 어떤 콘텐츠를 누구에게 어떤 방식으로 하는지 구체적으로. events는 빈 문자열",
    "events_only": "bio는 반드시 빈 문자열. events에 '이벤트 제목 / 이벤트 제목' 형태로 분야가 드러나는 제목 1~2개",
}
SYSTEM_PROMPT = (
    "너는 한국어 크리에이터 플랫폼의 가상 크리에이터 프로필을 만드는 데이터 작성자다. "
    "실제 인물·채널을 흉내 내지 말고, 사람마다 말투와 관심사가 다르게 자연스럽게 쓴다. "
    "요청한 슬롯 수만큼, 슬롯 순서 그대로 JSON으로만 답한다."
)
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["creators"],
    "properties": {
        "creators": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "bio", "events", "subtopic"],
                "properties": {
                    "name": {"type": "string"},
                    "bio": {"type": "string"},
                    "events": {"type": "string"},
                    "subtopic": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class Slot:
    """배치 안 한 명에 대한 지시다. gold·declared는 여기서 정해지고 모델은 관여하지 않는다."""

    style: str
    gold: tuple[str, ...]
    declared: tuple[str, ...]  # gold와 다를 때만 채운다(기존 CSV와 같은 규칙)
    kind: str  # "plain" | "cross" | "misdeclared"


def build_slots(category_code: str, batch_index: int) -> list[Slot]:
    """카테고리·배치 번호로 슬롯 10개를 결정적으로 만든다."""
    partners = PARTNERS[category_code]
    slots = []
    for i, style in enumerate(STYLE_BY_SLOT):
        if i == CROSS_SLOT:
            slots.append(Slot(style, (category_code, partners[batch_index % 3]), (), "cross"))
        elif i == MISDECLARED_SLOT:
            slots.append(Slot(style, (category_code,), (partners[(batch_index + 1) % 3],), "misdeclared"))
        else:
            slots.append(Slot(style, (category_code,), (), "plain"))
    return slots


def build_prompt(categories: dict[str, Category], slots: list[Slot], avoid: list[str]) -> str:
    """배치 하나의 사용자 프롬프트를 만든다. avoid는 같은 카테고리에서 이미 쓴 이름·세부주제다."""
    lines = []
    for n, slot in enumerate(slots, start=1):
        if slot.kind == "cross":
            field = f"분야: {' + '.join(categories[code].name for code in slot.gold)} 두 분야를 대등하게 다룸"
        else:
            field = f"분야: {categories[slot.gold[0]].name}"
        lines.append(f"{n}. {field} / 스타일: {STYLE_GUIDE[slot.style]}")
    avoid_text = f"\n이미 만든 크리에이터라 겹치지 않게 피할 것: {', '.join(avoid)}" if avoid else ""
    first = categories[slots[0].gold[0]]
    return (
        f"카테고리 '{first.name}'({first.description})의 가상 크리에이터 {len(slots)}명을 만들어라. "
        "이름은 닉네임 스타일로 서로 달라야 하고, 세부주제(subtopic)는 서로 다른 좁은 주제로 쓴다.\n"
        "슬롯별 요구:\n" + "\n".join(lines) + avoid_text
    )


def _norm_subtopic(text: str) -> str:
    return "".join(text.split()).lower()


def validate_batch(
    items: list[dict], slots: list[Slot], used_names: set[str], used_subtopics: set[str] | None = None
) -> list[str]:
    """모델 응답이 슬롯 요구를 지켰는지 확인하고 위반 사항을 문자열 목록으로 돌려준다.

    used_subtopics는 같은 카테고리에서 앞선 배치가 쓴 세부주제(정규화본)다. 배치 안 중복도 함께 막는다.
    """
    if len(items) != len(slots):
        return [f"항목 수 {len(items)} != 슬롯 수 {len(slots)}"]
    errors = []
    seen = {name.lower() for name in used_names}
    seen_subtopics = set(used_subtopics or ())
    for n, (item, slot) in enumerate(zip(items, slots), start=1):
        name = item["name"].strip()
        if not name or name.lower() in seen:
            errors.append(f"{n}번 이름이 비었거나 중복: {name!r}")
        seen.add(name.lower())
        subtopic = _norm_subtopic(item["subtopic"])
        if not subtopic:
            errors.append(f"{n}번 subtopic 비어 있음")
        elif subtopic in seen_subtopics:
            errors.append(f"{n}번 subtopic이 같은 카테고리에서 중복: {item['subtopic'].strip()!r}")
        seen_subtopics.add(subtopic)
        bio, events = item["bio"].strip(), item["events"].strip()
        if slot.style == "events_only":
            if bio or not events:
                errors.append(f"{n}번 events_only인데 bio가 있거나 events가 비어 있음")
        elif not bio:
            errors.append(f"{n}번 bio가 비어 있음")
        elif slot.style == "short_casual" and len(bio) > 80:
            errors.append(f"{SOFT_PREFIX}{n}번 short_casual인데 bio가 {len(bio)}자")
        elif slot.style == "long_polished" and len(bio) < 80:
            errors.append(f"{SOFT_PREFIX}{n}번 long_polished인데 bio가 {len(bio)}자")
    return errors


def to_csv_row(creator_id: str, item: dict, slot: Slot) -> dict[str, str]:
    """검증을 통과한 모델 출력과 슬롯 지시를 CSV 한 행으로 합친다."""
    return {
        "id": creator_id,
        "name": item["name"].strip(),
        "bio": item["bio"].strip(),
        "events": item["events"].strip(),
        "subtopic": item["subtopic"].strip(),
        "gold": "|".join(slot.gold),
        "written_by": "synthetic-large",
        "note": f"large:{slot.style}:{slot.kind}",
        "declared": "|".join(slot.declared),
    }


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    """CSV를 LF 줄바꿈으로 쓴다. 기본값(CRLF)이면 다른 OS에서 hash 검증이 깨진다(이슈 #4 교훈)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _generate_batch(
    client: OpenAI, model: str, prompt: str, slots: list[Slot], used_names: set[str], used_subtopics: set[str] | None = None
) -> tuple[list[dict], int, int]:
    """배치 하나를 생성하고 검증한다. 위반이면 MAX_ATTEMPTS번까지 다시 시도하며 토큰은 시도마다 합산한다."""
    in_tokens = out_tokens = 0
    last_errors: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        feedback = f"\n\n직전 응답이 다음 요구를 어겼다. 이번엔 반드시 지켜라: {'; '.join(last_errors)}" if last_errors else ""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + feedback}],
            response_format={"type": "json_schema", "json_schema": {"name": "creators", "strict": True, "schema": _SCHEMA}},
        )
        in_tokens += response.usage.prompt_tokens
        out_tokens += response.usage.completion_tokens
        items = json.loads(response.choices[0].message.content)["creators"]
        last_errors = validate_batch(items, slots, used_names, used_subtopics)
        if not last_errors:
            return items, in_tokens, out_tokens
    if all(error.startswith(SOFT_PREFIX) for error in last_errors):
        print(f"[generate-large] 경고: 길이 요구만 못 지켰지만 받아들임 -> {last_errors}")
        return items, in_tokens, out_tokens
    raise ValueError(f"{MAX_ATTEMPTS}번 시도해도 검증을 통과하지 못했습니다: {last_errors}")


def estimate_cost(model: str, calls: int) -> float:
    """--dry-run용 대략적인 비용(USD)이다. 실제 비용은 생성 후 출력되는 토큰 합계로 계산한다."""
    price = GENERATOR_MODELS[model]
    per_in, per_out = ESTIMATED_TOKENS_PER_CALL
    return calls * (per_in * price["input_price"] + per_out * price["output_price"]) / 1_000_000


def main() -> None:
    """카테고리별 배치를 만들어(캐시가 있으면 재사용) creators_large.csv로 합친다."""
    parser = argparse.ArgumentParser(description="dev 규모 민감도용 합성 크리에이터 생성")
    parser.add_argument("--model", default=DEFAULT_GENERATOR_MODEL, choices=list(GENERATOR_MODELS))
    parser.add_argument("--dry-run", action="store_true", help="API를 부르지 않고 계획·예상 비용만 출력")
    args = parser.parse_args()

    categories = load_categories()
    by_code = {c.code: c for c in categories}
    calls = len(categories) * BATCHES_PER_CATEGORY
    cache_dir = CACHE_DIR / "large_gen"
    for path in cache_dir.glob("*.json"):
        cached_model = json.loads(path.read_text(encoding="utf-8")).get("model")
        if cached_model != args.model:
            raise ValueError(f"{path.name}은 {cached_model}로 만든 캐시입니다. --model {args.model}로 쓰려면 results/cache/large_gen/을 지우고 다시 만드세요.")
    pending = [
        (c.code, b) for c in categories for b in range(BATCHES_PER_CATEGORY)
        if not (cache_dir / f"{c.code}_{b}.json").exists()
    ]
    print(f"[generate-large] {len(categories)}개 카테고리 x {BATCHES_PER_CATEGORY}배치 x {BATCH_SIZE}명 = {calls * BATCH_SIZE}명, "
          f"남은 API 호출 {len(pending)}건, 예상 비용 약 ${estimate_cost(args.model, len(pending)):.2f} ({args.model})")
    if args.dry_run:
        return

    client = OpenAI()
    used_names = {c.name for c in load_creators()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    rows: list[dict[str, str]] = []
    next_id = 1
    for category in categories:
        avoid: list[str] = []
        used_subtopics: set[str] = set()
        for batch in range(BATCHES_PER_CATEGORY):
            slots = build_slots(category.code, batch)
            cache_path = cache_dir / f"{category.code}_{batch}.json"
            if cache_path.exists():
                items = json.loads(cache_path.read_text(encoding="utf-8"))["items"]
                # 최초 생성과 같은 기준: 길이(소프트) 위반은 허용하고 그 외 위반만 거부한다
                errors = [e for e in validate_batch(items, slots, used_names, used_subtopics) if not e.startswith(SOFT_PREFIX)]
                if errors:
                    raise ValueError(f"{cache_path.name} 캐시가 현재 검증 규칙을 통과하지 못합니다({errors}). 지우고 다시 만드세요.")
            else:
                prompt = build_prompt(by_code, slots, avoid)
                items, in_tokens, out_tokens = _generate_batch(client, args.model, prompt, slots, used_names, used_subtopics)
                total_in += in_tokens
                total_out += out_tokens
                cache_path.write_text(json.dumps({"model": args.model, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[generate-large] {category.code} 배치 {batch + 1}/{BATCHES_PER_CATEGORY} 완료")
            for item, slot in zip(items, slots):
                rows.append(to_csv_row(f"L{next_id:03d}", item, slot))
                used_names.add(item["name"].strip())
                used_subtopics.add(_norm_subtopic(item["subtopic"]))
                avoid.append(f"{item['name'].strip()}({item['subtopic'].strip()})")
                next_id += 1
    write_csv(rows, CREATORS_LARGE_CSV)
    price = GENERATOR_MODELS[args.model]
    cost = (total_in * price["input_price"] + total_out * price["output_price"]) / 1_000_000
    print(f"[generate-large] {len(rows)}명을 {CREATORS_LARGE_CSV.name}에 저장했습니다. 이번 실행 토큰 입력 {total_in}/출력 {total_out}, 비용 약 ${cost:.4f}")
    print("hash를 src/config.py의 CREATORS_LARGE_CSV_SHA256에 고정하세요: shasum -a 256 data/creators_large.csv")


if __name__ == "__main__":
    main()
