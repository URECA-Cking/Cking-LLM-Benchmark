"""사용자 취향 요약문 쿼리 90개를 만든다 (이슈 #12).

서비스가 사용자의 취향을 요약한 텍스트를 쿼리로 크리에이터를 추천할 때를 가정한 검증용 쿼리다. 프로필
30개(카테고리당 단일 분야 2개 + 두 분야를 함께 좋아하는 1개)마다 세 가지 표현(키워드형·문장형·서술형)을
LLM으로 만들어 `data/taste_queries.csv`에 쓴다. 정답 분야(gold)는 모델이 아니라 코드가 정한다. 프로필 결과는
`results/cache/taste_gen/`에 저장해 재실행 때 API를 다시 부르지 않는다.
사용법: `python3 -m src.generate_taste [--model ...] [--dry-run]`
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass

from openai import OpenAI

from src.config import CACHE_DIR, TASTE_QUERIES_CSV
from src.data import load_categories
from src.generate_large import DEFAULT_GENERATOR_MODEL, GENERATOR_MODELS, PARTNERS

MAX_ATTEMPTS = 3
STYLES = ("keyword", "sentence", "long")
CSV_FIELDS = ["id", "profile_id", "style", "text", "gold"]
ESTIMATED_TOKENS_PER_CALL = (700, 600)  # (입력, 출력) 대략치, --dry-run 비용 추정용
STYLE_GUIDE = {
    "keyword": "쉼표로 이은 짧은 키워드 나열. 4~6개, 60자 이하",
    "sentence": "취향을 한두 문장으로 요약. 40~150자",
    "long": "취향과 시청 습관을 3~4문장으로 서술. 120자 이상",
}
SYSTEM_PROMPT = (
    "너는 크리에이터 플랫폼이 사용자 취향을 요약한 문장을 흉내 내는 데이터 작성자다. "
    "카테고리 이름 자체(예: '운동·건강')를 그대로 쓰지 말고 구체적인 관심사·콘텐츠 종류로 표현한다. "
    "같은 사용자의 취향을 세 가지 길이로 쓰며, JSON으로만 답한다."
)
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(STYLES),
    "properties": {style: {"type": "string"} for style in STYLES},
}


@dataclass(frozen=True)
class Profile:
    """취향 프로필 하나다. gold(정답 분야)는 코드가 정한다."""

    profile_id: str
    gold: tuple[str, ...]
    variant: str  # "single_a" | "single_b" | "cross"


def build_profiles(category_codes: list[str]) -> list[Profile]:
    """카테고리마다 단일 분야 프로필 2개와, 자연스럽게 겹치는 분야와 묶은 프로필 1개를 만든다(총 3x카테고리 수)."""
    profiles = []
    for code in category_codes:
        for variant, gold in (("single_a", (code,)), ("single_b", (code,)), ("cross", (code, PARTNERS[code][0]))):
            profiles.append(Profile(f"P{len(profiles) + 1:02d}", gold, variant))
    return profiles


def build_prompt(names: dict[str, str], profile: Profile, previous: str) -> str:
    """프로필 하나의 프롬프트를 만든다. previous는 같은 분야 앞선 프로필의 문장형 쿼리라 겹침을 피하게 한다."""
    topic = " + ".join(names[code] for code in profile.gold)
    guide = "\n".join(f"- {style}: {STYLE_GUIDE[style]}" for style in STYLES)
    avoid = f"\n같은 분야의 다른 사용자가 이미 이렇게 썼으니 관심사가 겹치지 않게 다르게 쓸 것: {previous}" if previous else ""
    return f"이 사용자는 다음 분야에 관심이 있다: {topic}.\n길이별 요구:\n{guide}{avoid}"


SOFT_PREFIX = "[길이] "


def validate(result: dict) -> list[str]:
    """세 표현이 비어 있지 않고 길이 요구를 지켰는지 본다. 길이 위반만 소프트(마지막 시도에서 허용)다."""
    errors = [f"{style}가 비어 있음" for style in STYLES if not result.get(style, "").strip()]
    if errors:
        return errors
    if len(result["keyword"].strip()) > 60:
        errors.append(f"{SOFT_PREFIX}keyword가 {len(result['keyword'].strip())}자")
    if len(result["long"].strip()) < 120:
        errors.append(f"{SOFT_PREFIX}long이 {len(result['long'].strip())}자")
    return errors


def _generate_profile(client: OpenAI, model: str, prompt: str) -> tuple[dict, int, int]:
    """프로필 하나를 생성·검증한다. 위반이면 MAX_ATTEMPTS번까지 다시 시도하고 토큰을 합산한다."""
    in_tokens = out_tokens = 0
    errors: list[str] = []
    result: dict = {}
    for _ in range(MAX_ATTEMPTS):
        feedback = f"\n\n직전 응답이 다음 요구를 어겼다. 이번엔 반드시 지켜라: {'; '.join(errors)}" if errors else ""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + feedback}],
            response_format={"type": "json_schema", "json_schema": {"name": "taste", "strict": True, "schema": _SCHEMA}},
        )
        in_tokens += response.usage.prompt_tokens
        out_tokens += response.usage.completion_tokens
        result = json.loads(response.choices[0].message.content)
        errors = validate(result)
        if not errors:
            return result, in_tokens, out_tokens
    if errors and all(e.startswith(SOFT_PREFIX) for e in errors):
        print(f"[generate-taste] 경고: 길이 요구만 못 지켰지만 받아들임 -> {errors}")
        return result, in_tokens, out_tokens
    raise ValueError(f"{MAX_ATTEMPTS}번 시도해도 검증을 통과하지 못했습니다: {errors}")


def to_rows(profile: Profile, result: dict) -> list[dict[str, str]]:
    """프로필 하나를 CSV 행 3개(표현별)로 바꾼다."""
    return [
        {"id": f"{profile.profile_id}_{style}", "profile_id": profile.profile_id, "style": style, "text": result[style].strip(), "gold": "|".join(profile.gold)}
        for style in STYLES
    ]


def write_csv(rows: list[dict[str, str]], path) -> None:
    """LF 줄바꿈으로 쓴다(hash 검증이 OS에 따라 깨지지 않게)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def estimate_cost(model: str, calls: int) -> float:
    """--dry-run용 대략적인 비용(USD)이다."""
    price = GENERATOR_MODELS[model]
    per_in, per_out = ESTIMATED_TOKENS_PER_CALL
    return calls * (per_in * price["input_price"] + per_out * price["output_price"]) / 1_000_000


def main() -> None:
    """프로필별로 쿼리를 만들어(캐시가 있으면 재사용) taste_queries.csv로 합친다."""
    parser = argparse.ArgumentParser(description="사용자 취향 요약문 쿼리 생성")
    parser.add_argument("--model", default=DEFAULT_GENERATOR_MODEL, choices=list(GENERATOR_MODELS))
    parser.add_argument("--dry-run", action="store_true", help="API를 부르지 않고 계획·예상 비용만 출력")
    args = parser.parse_args()

    categories = load_categories()
    names = {c.code: c.name for c in categories}
    profiles = build_profiles([c.code for c in categories])
    cache_dir = CACHE_DIR / "taste_gen"
    for path in cache_dir.glob("*.json"):
        cached_model = json.loads(path.read_text(encoding="utf-8")).get("model")
        if cached_model != args.model:
            raise ValueError(f"{path.name}은 {cached_model}로 만든 캐시입니다. --model {args.model}로 쓰려면 results/cache/taste_gen/을 지우고 다시 만드세요.")
    pending = [p for p in profiles if not (cache_dir / f"{p.profile_id}.json").exists()]
    print(f"[generate-taste] 프로필 {len(profiles)}개 x 표현 {len(STYLES)}종 = {len(profiles) * len(STYLES)}개, "
          f"남은 API 호출 {len(pending)}건, 예상 비용 약 ${estimate_cost(args.model, len(pending)):.2f} ({args.model})")
    if args.dry_run:
        return

    client = OpenAI()
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    rows: list[dict[str, str]] = []
    previous_by_gold: dict[str, str] = {}
    for profile in profiles:
        cache_path = cache_dir / f"{profile.profile_id}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))["result"]
            errors = [e for e in validate(result) if not e.startswith(SOFT_PREFIX)]
            if errors:
                raise ValueError(f"{cache_path.name} 캐시가 현재 검증 규칙을 통과하지 못합니다({errors}). 지우고 다시 만드세요.")
        else:
            previous = previous_by_gold.get(profile.gold[0], "") if len(profile.gold) == 1 else ""
            result, in_tokens, out_tokens = _generate_profile(client, args.model, build_prompt(names, profile, previous))
            total_in += in_tokens
            total_out += out_tokens
            cache_path.write_text(json.dumps({"model": args.model, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[generate-taste] {profile.profile_id} 완료")
        if len(profile.gold) == 1:
            previous_by_gold[profile.gold[0]] = result["sentence"].strip()
        rows.extend(to_rows(profile, result))
    write_csv(rows, TASTE_QUERIES_CSV)
    price = GENERATOR_MODELS[args.model]
    cost = (total_in * price["input_price"] + total_out * price["output_price"]) / 1_000_000
    print(f"[generate-taste] {len(rows)}개를 {TASTE_QUERIES_CSV.name}에 저장했습니다. 이번 실행 토큰 입력 {total_in}/출력 {total_out}, 비용 약 ${cost:.4f}")
    print("hash를 src/config.py의 TASTE_QUERIES_CSV_SHA256에 고정하세요: shasum -a 256 data/taste_queries.csv")


if __name__ == "__main__":
    main()
