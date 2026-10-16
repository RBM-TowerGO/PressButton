import pytest
from pressb.astribot_collection.verification import choose_concurrency,concurrent_throughput


def row(k,speed,**kwargs):
    return dict(envs_per_gpu=k,accepted_per_minute=speed,max_vram_fraction=.2,
                min_available_ram_gib=100,**kwargs)


def test_concurrency_prefers_smaller_within_five_percent_and_respects_memory():
    rows=[row(1,1),row(2,2),row(4,2.08),row(8,3)]
    rows[-1]['max_vram_fraction']=.81
    assert choose_concurrency(rows)['envs_per_gpu']==2
    rows[1]['min_available_ram_gib']=31
    assert choose_concurrency(rows)['envs_per_gpu']==4
    with pytest.raises(ValueError,match='Incomplete'):choose_concurrency(rows[:3])


def test_dual_throughput_includes_actual_interval_union_and_rejects_sequential():
    rows=[dict(gpu=0,accepted=4,measurement_start_unix_s=100,measurement_end_unix_s=200),
          dict(gpu=1,accepted=4,measurement_start_unix_s=110,measurement_end_unix_s=220)]
    assert concurrent_throughput(rows)==dict(accepted=8,wall_s=120,accepted_per_minute=4)
    rows[1].update(measurement_start_unix_s=300,measurement_end_unix_s=400)
    with pytest.raises(ValueError,match='concurrently'):concurrent_throughput(rows)
