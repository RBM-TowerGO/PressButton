"""Real CUDA regression: preserve camera order and exact native RGB bytes."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('PRESSB_TEST_GPU') != '1', reason='Opt-in CUDA hardware check')


@pytest.mark.parametrize('count', [1, 2, 7, 16])
def test_gpu_packing_preserves_native_camera_pixels(count):
    import torch
    import warp as wp
    from pressb.astribot_collection.gpu_pixels import pack_rgb
    from pressb.dataset_scene import split_tiled_rgb
    wp.init()
    width, height = 64, 48
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    source = torch.randint(0, 256, (rows * height, columns * width, 4), dtype=torch.uint8, device='cuda:0')
    packed = torch.empty((count, height, width, 3), dtype=torch.uint8, device='cuda:0')
    for _ in range(2):
        pack_rgb(source, packed, columns)
        torch.cuda.synchronize()
        expected = np.stack(split_tiled_rgb(source.cpu().numpy(), count, (width, height)))
        np.testing.assert_array_equal(packed.cpu().numpy(), expected)
        source.bitwise_xor_(255)
