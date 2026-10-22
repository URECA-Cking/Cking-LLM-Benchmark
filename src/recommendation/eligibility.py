"""분류 자격과 현재 소개 지문을 점수/추천 근거와 별도로 전달한다."""
from src.recommendation.cache import text_hash

POLICY = 'RELATED_V1'


def proof(profiles, tags, taxonomy_version, taxonomy_hash):
    return {'policyVersion': POLICY, 'taxonomyVersion': taxonomy_version, 'taxonomyHash': taxonomy_hash,
            'profiles': [{'creatorId': row.creator_id, 'introductionHash': text_hash(row.text),
                          'tags': sorted(tags.get(row.creator_id, ()))} for row in sorted(profiles, key=lambda r:r.creator_id)]}


def validate(value, profiles, selected, *, seed=None, interest=None, allowed=()):
    if not isinstance(value, dict) or set(value) != {"policyVersion", "taxonomyVersion", "taxonomyHash", "profiles"} or value.get('policyVersion') != POLICY or not isinstance(value.get('profiles'), list):
        raise ValueError('RELATED_V1 proof required')
    expected = set(selected) | ({seed} if seed is not None else set())
    actual = {}
    previous = 0
    for row in value['profiles']:
        if not isinstance(row, dict) or set(row) != {'creatorId', 'introductionHash', 'tags'}:
            raise ValueError('Invalid profile proof')
        cid = row.get('creatorId')
        tags = row.get('tags')
        if (type(cid) is not int or cid <= previous or cid not in expected or cid in actual or cid not in profiles
                or row.get('introductionHash') != text_hash(profiles[cid].text)
                or not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
                or len(tags) > 3 or tags != sorted(set(tags))
                or not set(tags) <= set(allowed)):
            raise ValueError('Invalid classification/input proof')
        actual[cid] = set(tags)
        previous = cid
    if set(actual) != expected:
        raise ValueError('Incomplete proof')
    for cid in selected:
        if not profiles[cid].text or (interest is not None and interest not in actual[cid]) or (
                seed is not None and (not profiles[seed].text or not actual[seed] & actual[cid])):
            raise ValueError('Unrelated candidate')
