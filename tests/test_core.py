import numpy as np
import torch

from disp_quality.features import build_features
from disp_quality.model import build_model
from disp_quality.targets import build_targets


def minimal_cfg():
    return {
        "data": {"use_intensity": False},
        "features": {"max_disparity_px": 128.0, "local_residual_clip_px": 6.0},
        "targets": {"bad_pixel_threshold_px": 3.0, "reliability_tau_px": 3.0, "max_error_px": 32.0},
        "model": {"base_channels": 4, "group_norm_groups": 4, "dropout": 0.0},
    }


def test_targets_and_model_shapes():
    cfg = minimal_cfg()
    disparity = np.full((64, 96), 40.0, dtype=np.float32)
    gt = disparity.copy()
    gt[:, 50:] += 5.0
    roi = np.ones_like(disparity, dtype=bool)
    features, names = build_features(disparity, roi, None, cfg)
    target, mask, _ = build_targets(disparity, gt, roi, cfg)
    assert features.shape == (6, 64, 96)
    assert np.array_equal(features[-1], roi.astype(np.float32))
    assert target.shape == (3, 64, 96)
    assert mask.shape == (1, 64, 96)
    assert target[0, :, :50].mean() > target[0, :, 50:].mean()
    assert target[2, :, 50:].mean() == 1.0

    model = build_model(cfg, len(names))
    output = model(torch.from_numpy(features[None]))
    assert output.shape == (1, 3, 64, 96)


def test_grouped_partition_keeps_left_right_together():
    from pathlib import Path
    import random

    from disp_quality.cli.make_split import capture_group_key, grouped_partition

    candidates = [
        Path('/tmp/capture_a_l'), Path('/tmp/capture_a_r'),
        Path('/tmp/capture_b_l'), Path('/tmp/capture_b_r'),
        Path('/tmp/capture_c_l'), Path('/tmp/capture_c_r'),
        Path('/tmp/capture_d_l'), Path('/tmp/capture_d_r'),
    ]
    train, val, metadata = grouped_partition(candidates, 2, random.Random(42), r'_(l|r)$')
    train_groups = {capture_group_key(path, r'_(l|r)$') for path in train}
    val_groups = {capture_group_key(path, r'_(l|r)$') for path in val}
    assert train_groups.isdisjoint(val_groups)
    assert metadata['grouping_enabled'] is True
    assert len(val) == 2


def test_thesis_parameter_count_and_deployment_mask():
    from disp_quality.predict import apply_deployment_rules

    cfg = {
        "model": {"base_channels": 16, "group_norm_groups": 8, "dropout": 0.05},
    }
    model = build_model(cfg, 6)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_030_915

    disparity = np.array([[10.0, 0.0], [5.0, np.nan]], dtype=np.float32)
    prediction = {
        "reliability": np.full((2, 2), 0.8, dtype=np.float32),
        "error_normalized": np.full((2, 2), 0.2, dtype=np.float32),
        "error_px": np.full((2, 2), 1.5, dtype=np.float32),
        "bad_score": np.full((2, 2), 0.3, dtype=np.float32),
    }
    deployed = apply_deployment_rules(prediction, disparity)
    assert np.array_equal(deployed["reliability"], np.array([[0.8, 0.0], [0.8, 0.0]], dtype=np.float32))
    assert np.all(deployed["bad_score"] == 0.3)
