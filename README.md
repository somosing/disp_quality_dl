# Learning-Based Disparity Reliability — Thesis-Aligned Repository

This repository implements the final learning-based method described in the thesis **Depth Data Quality and Reliability Assessment for Warehouse Environments**.

## Final reported method

### Inputs

The model requires six aligned full-resolution channels:

1. globally scaled sensor disparity,
2. robustly normalized sensor disparity,
3. sensor-validity mask,
4. normalized disparity-gradient magnitude,
5. normalized local median residual,
6. binary ROI mask supplied by the upstream segmentation stage.

Intensity and sensor confidence are not model inputs. ToF-based reference disparity is used only for training targets and reference-based evaluation.

### Outputs

A shared `1×1` projection produces three logits:

1. reliability,
2. normalized absolute error,
3. uncalibrated bad-pixel score.

The deployed reliability is deterministically validity-masked:

```text
reliability_deployed = sensor_validity * sigmoid(reliability_logit)
```

### Reported architecture and training configuration

- resolution: `1024 × 1224`
- input/output channels: `6 / 3`
- encoder widths: `16, 32, 64, 128`
- bottleneck: `256`
- decoder widths: `128, 64, 32, 16`
- trainable parameters: `2,030,915`
- GroupNorm groups: `8`
- dropout: `0.05`
- effective batch size: `4`
- AdamW learning rate: `2e-4`
- weight decay: `1e-4`
- fixed learning rate
- maximum epochs: `80`
- early stopping after stable validation loss
- reported final epoch: `65`
- selected checkpoint: epoch `58`, validation loss `0.0895`

The authoritative configuration is:

```text
configs/thesis_final_fullres.yaml
```

## Dataset layout

```text
scene_id/
├── disparity.png
├── gt_disparity.npy       # stored at 1/16-pixel fixed-point scale
├── roi_mask.png
└── intensity_left.png     # not used by the final model
```

The reported supervised split contains 2,223 training scenes and 247 validation scenes after 22 invalid folders were excluded from 2,492 prepared folders.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Workflow

1. Edit the dataset path in `configs/thesis_final_fullres.yaml`.
2. Audit the dataset.
3. Generate or supply the fixed 2,223/247 split.
4. Train with the thesis-aligned configuration.
5. Run inference or evaluation.

```bash
python -m disp_quality.cli.audit_dataset \
  --config configs/thesis_final_fullres.yaml \
  --output-dir audit_output

python -m disp_quality.cli.make_split \
  --config configs/thesis_final_fullres.yaml

python -m disp_quality.cli.train \
  --config configs/thesis_final_fullres.yaml
```

Inference requires the ROI because it is the sixth network input:

```bash
python -m disp_quality.cli.infer \
  --checkpoint outputs/thesis_final_fullres/best.pth \
  --disparity /path/to/disparity.png \
  --roi-mask /path/to/roi_mask.png \
  --output-dir inference_output
```

Evaluation samples at most 20,000 reference-valid ROI pixels per scene and reports combined and jointly-valid bad-pixel ranking, complete-ROI score correlations, and scene-level quality–coverage outputs.

## Important scope note

The repository formalizes and executes the algorithm described in the thesis. It does not include the industrial dataset, company-internal segmentation model, final trained checkpoint, or company-provided ToF acquisition pipeline.
