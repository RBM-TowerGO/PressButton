"""Native Qwen images must preserve pixels without altering JEPA supervision."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from transformers import AutoProcessor

from starVLA.dataloader.s1b_real import S1BRealDataset, collate_fn
from starVLA.training.train_s1b_real import training_contract, validate_continuation_contract

ROOT = Path(os.environ.get('S1B_PROJECT', Path(__file__).resolve().parents[3]))


def decode_anchor(dataset, sample):
    episode = dataset._episodes[sample['source_episode_id']]
    images = []
    for key in dataset.camera_keys:
        path, offset = episode['videos'][key]
        with av.open(path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == offset + sample['frame_index']:
                    images.append(frame.to_ndarray(format='rgb24'))
                    break
    assert len(images) == 2
    return images


@pytest.fixture(scope='module')
def real_samples():
    options = dict(root=ROOT/'data/astribot_s1b_real_joint10_v1',
                   contract_path=ROOT/'training/s1b_real/dataset_contract.json', split='val', frame_cache_size=4)
    legacy = S1BRealDataset(**options)
    native = S1BRealDataset(**options, image_mode='native')
    rows = []
    try:
        for i in (0, len(native)-1):
            old, new = legacy[i], native[i]
            raw = decode_anchor(native, new)
            assert [im.size for im in new['image']] == [(1280, 720), (640, 360)]
            for original, im, small in zip(raw, new['image'], old['image']):
                np.testing.assert_array_equal(original, np.asarray(im))
                np.testing.assert_array_equal(np.asarray(small), np.asarray(
                    Image.fromarray(original).resize((224, 224), Image.Resampling.BILINEAR)))
            for key in ('video', 'state', 'action', 'action_is_pad', 'video_is_pad', 'video_timestamps'):
                np.testing.assert_array_equal(old[key], new[key])
            assert new['video'].shape == (2, 8, 224, 224, 3)
            assert len(native._frames) <= 4
            # Cached retrieval must give the same pixels and masks.
            again = native[i]
            for a, b in zip(new['image'], again['image']):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            rows.append((new, raw))
        yield rows
    finally:
        native.close()
        legacy.close()


def test_native_pixels_and_unchanged_supervision(real_samples):
    batch = collate_fn([row[0] for row in real_samples])
    assert len(batch) == 2
    assert all([im.size for im in item['image']] == [(1280, 720), (640, 360)] for item in batch)


def test_training_deployment_processor_equality(real_samples):
    pytest.importorskip('transformers.models.qwen3_5')
    from starVLA.model.modules.vlm.QWen3_5 import Qwen3_5Interface
    sys.path.insert(0, str(ROOT/'src'))
    from pressb.s1b_deployment.policy import Policy
    model_path = ROOT/'models/astribot_backbones/Qwen3.5-4B'
    if not model_path.is_dir():
        model_path = OmegaConf.load(ROOT/'configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml').framework.qwenvl.base_vlm
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    # Exercise the real multimodal input builder without allocating 4B model weights.
    interface = Qwen3_5Interface.__new__(Qwen3_5Interface)
    torch.nn.Module.__init__(interface)
    interface.processor = processor
    interface.model = SimpleNamespace(device=torch.device('cpu'))
    interface.config = OmegaConf.create({'datasets': {'vla_data': {'CoT_prompt': '{instruction}'}}})
    policy = Policy.__new__(Policy)
    policy.image_mode = 'native'
    policy.resolution = 224
    policy.stats = json.loads((ROOT/'training/s1b_real/dataset_contract.json').read_text())['normalization']
    deployed = []
    def capture(images, state, seed):
        deployed.append(images)
        return np.zeros((7, 10), np.float32), 0.
    policy.predict_prepared = capture
    for _, raw in real_samples:
        policy.predict(*raw, np.zeros(10, np.float32))
    instructions = [row[0]['lang'] for row in real_samples]
    training = interface.build_qwenvl_inputs([row[0]['image'] for row in real_samples], instructions)
    inference = interface.build_qwenvl_inputs(deployed, instructions)
    assert set(training) == set(inference)
    for key in training:
        assert torch.equal(training[key], inference[key]), key
    assert training['image_grid_thw'].tolist() == [[1, 44, 80], [1, 22, 40]] * 2


def test_image_mode_contract_and_legacy_fingerprint():
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml'), resolve=True)
    native, native_hash = training_contract(cfg)
    cfg['datasets']['vla_data'].pop('image_mode')
    legacy, legacy_hash = training_contract(cfg)
    assert native_hash != legacy_hash
    assert 'image_mode' not in legacy['data']
    expected = copy.deepcopy(native)
    expected['data'].pop('image_mode')
    assert legacy_hash == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='Continuation changed'):
        validate_continuation_contract(legacy, native)
    validate_continuation_contract(native, native)


def test_published_legacy_fingerprint_unchanged():
    run = ROOT/'models/s1b_real_joint10_8000_v1'
    if not (run/'training_contract.json').is_file():
        pytest.skip('Published legacy deployment snapshot is not installed')
    expected = json.loads((run/'training_contract.json').read_text())
    config = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'), resolve=True)
    actual, fingerprint = training_contract(config)
    assert actual == expected['contract']
    assert fingerprint == expected['fingerprint']


@pytest.mark.parametrize('mode,image_size', [('native', [224, 224]), ('typo', None)])
def test_bad_image_config_rejected_before_model_loading(mode, image_size):
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA
    cfg = OmegaConf.create({'datasets': {'vla_data': {'image_mode': mode, 'image_size': image_size}}})
    with pytest.raises(ValueError, match='image_mode'):
        VLA_JEPA(cfg)
