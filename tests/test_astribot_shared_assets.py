from concurrent.futures import ThreadPoolExecutor
import os
import pytest
from pressb.astribot_collection.assets import share_asset


def test_concurrent_workers_share_complete_immutable_asset(tmp_path):
    paths = [tmp_path / f'worker{i}.usda' for i in range(4)]
    content = b'#usda 1.0\n' + bytes(range(256)) * 4096
    for path in paths:
        path.write_bytes(content)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda p: share_asset(p, tmp_path / 'cache'), paths))
    assert len({r['canonical'] for r in records}) == 1
    assert len({os.stat(p).st_ino for p in paths}) == 1
    assert all(p.read_bytes() == content for p in paths)


def test_corrupt_cached_asset_is_rejected_without_replacing_input(tmp_path):
    original = tmp_path / 'first.usda'
    original.write_bytes(b'complete original')
    share_asset(original, tmp_path / 'cache')
    original.write_bytes(b'tampered')  # all hard links now reveal corruption
    second = tmp_path / 'second.usda'
    second.write_bytes(b'complete original')
    with pytest.raises(ValueError, match='Corrupted shared asset'):
        share_asset(second, tmp_path / 'cache')
    assert second.read_bytes() == b'complete original'


def test_resume_rejects_modified_external_asset(tmp_path):
    from pressb.astribot_collection.contract import atomic_json, commit_episode, validate_commit
    asset = tmp_path / 'scene.usda'
    asset.write_bytes(b'frozen scene')
    reference = share_asset(asset, tmp_path / 'cache')
    staging = tmp_path / 'episode.inprogress'
    staging.mkdir()
    atomic_json(staging / 'report.json', dict(shared_assets={'scene.usda': reference}))
    final = tmp_path / 'episode'
    commit_episode(staging, final, dict(status='passed', scope='shared_asset_integrity_fixture'))
    validate_commit(final)
    asset.write_bytes(b'modified scene')
    with pytest.raises(ValueError, match='Corrupt referenced asset'):
        validate_commit(final)
