from pressb.astribot_collection.dispatch import resource_violation, latest_resource


def test_limits_stop_new_work_without_discarding_an_inflight_wave():
    sample = dict(available_ram_bytes=32*1024**3, disk_free_bytes=20*1024**3,
                  gpus=[dict(index=0, memory_used_mib=80, memory_total_mib=100)])
    assert resource_violation(sample, [0, 1]) is None
    assert resource_violation({**sample, 'available_ram_bytes':32*1024**3-1}, [0]) == 'available_ram_below_32_GiB'
    assert resource_violation({**sample, 'disk_free_bytes':20*1024**3-1}, [0]) == 'disk_free_below_20_GiB'
    sample['gpus'][0]['memory_used_mib'] = 81
    assert resource_violation(sample, [0]) == 'gpu_0_vram_above_80_percent'
    assert resource_violation(sample, [1]) is None


def test_monitor_tail_ignores_an_unfinished_write(tmp_path):
    path = tmp_path/'resources.jsonl'
    path.write_text('{"available_ram_bytes": 123}\n{"available_ram_bytes":')
    assert latest_resource(path) == {'available_ram_bytes':123}
