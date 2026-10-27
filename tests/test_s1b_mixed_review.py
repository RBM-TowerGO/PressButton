import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from prepare_s1b_mixed_review import mixed_splits
from export_s1b_real_lerobot import split_episodes


def test_mixed_split_preserves_held_out_episodes_and_acquisition_blocks():
    first = split_episodes(96)
    # GUI order need not be acquisition order.
    second = [dict(original_id=i) for i in reversed(range(116))]
    mixed = mixed_splits(96,second,first)
    assert [i for i in mixed['train'] if i<96] == first['train']
    assert [i for i in mixed['validation'] if i<96] == first['validation']
    assert not set(mixed['train']) & set(mixed['validation'])
    assert set(mixed['train']) | set(mixed['validation']) == set(range(212))
    for block in mixed['second_acquisition_blocks']:
        assert set(block)<=set(mixed['train']) or set(block)<=set(mixed['validation'])
        ids = [second[i-96]['original_id'] for i in block]
        assert ids == sorted(ids)
    assert mixed == mixed_splits(96,second,first)


def test_invalid_legacy_split_cannot_leak_into_new_train_set():
    with pytest.raises(ValueError):
        mixed_splits(3,[dict(original_id=i) for i in range(16)],dict(train=[0,1],validation=[1,2]))
