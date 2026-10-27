import numpy as np
import pytest
from starVLA.training.train_s1b_real import validation_batch_metrics


def test_source_batches_do_not_hide_each_others_errors():
    rows = [dict(source_batch=batch, error=np.full((7,10),error).tolist(),
                 baseline=np.full(10,0.1).tolist(),gripper_correct=[1.]*7)
            for batch,error in [('elevator1',1.),('elevator1',3.),('elevator2',10.)]]
    metrics = validation_batch_metrics(rows)
    assert metrics['elevator1']['anchors'] == 2
    assert metrics['elevator2']['anchors'] == 1
    assert metrics['elevator1']['arm_mae_rad'] == 2.
    assert metrics['elevator2']['arm_mae_rad'] == 10.
    assert metrics['elevator2']['state_copy_first_arm_mae_rad'] == pytest.approx(.1)
