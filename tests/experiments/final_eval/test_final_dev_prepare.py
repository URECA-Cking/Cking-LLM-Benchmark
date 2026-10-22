"""Development sampling keeps reviewed creators out of the new diagnostic."""

import pytest
from src.final_dev_prepare import select_cases


def test_balanced_reproducible_development_sample_excludes_reviewed():
    cases = [
        {"id": f"{g}-{i}", "route": "creators", "group": g, "seeds": [f"{g}-{i}"]}
        for g in ("regular", "short")
        for i in range(12)
    ]
    excluded = {"regular-0", "short-0"}
    selected = select_cases(cases, excluded)
    assert len(selected) == 20
    assert sum(c["group"] == "regular" for c in selected) == 10
    assert all(not set(c["seeds"]) & excluded for c in selected)
    assert select_cases(list(reversed(cases)), excluded) == selected
    with pytest.raises(ValueError):
        select_cases(cases, {"short-" + str(i) for i in range(3)})
