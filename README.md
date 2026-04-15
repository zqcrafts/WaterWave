# WaterWave: Underwater Image/Video Enhancement

WaterWave is a BasicSR-based project for underwater enhancement.  
It focuses on common underwater degradations such as:

- color cast (blue/green shift)
- low contrast and haze-like appearance
- texture/detail attenuation
- temporal flicker and instability in video sequences

The current primary training config uses the `WaveField` model in `options/train/0.yml`.

---

## Project Structure (Relevant to This Work)

- `options/train/0.yml`: main training config (`model_type: WaveField`)
- `basicsr/train.py`: training entry
- `basicsr/test.py`: test entry
- `scripts/dist_train.sh`: distributed training script
- `scripts/dist_test.sh`: distributed testing script
- `flow_estimate/`: optical-flow preprocessing and related tools

---

## Environment Setup

Recommended: Python 3.8+ and PyTorch 1.10+ (matching your CUDA version).

```bash
cd WaterWave
pip install -r requirements.txt
pip install -v -e .
```

---

## Data Preparation

The default paths in `options/train/0.yml` include:

- `datasets/uw3_mini` (GT/training data)
- `dataroot_flow` (optical flow)
- `dataroot_flow_conf` (optical-flow confidence)

Update these paths to your local environment before running training.

---

## Training

### Single-GPU Training

```bash
python basicsr/train.py -opt options/train/0.yml
```

### Multi-GPU Training

```bash
bash scripts/dist_train.sh 2 options/train/0.yml
```

---

## Testing / Validation

### Single-GPU Testing

```bash
python basicsr/test.py -opt options/train/0.yml
```

### Multi-GPU Testing

```bash
bash scripts/dist_test.sh 2 options/train/0.yml
```

---

## Outputs and Logs

Training/testing outputs are typically written to:

- `experiments/<exp_name>/`
- `results/<exp_name>/`

`<exp_name>` comes from the `name` field in the config (currently `wavewater`).

---

## Notes

1. `basicsr/train.py` currently contains a hardcoded path append:
   - `sys.path.append('/home/ubuntu/data/code/waterwave_code')`
   - If your path is different, switch this to a project-relative path to avoid cross-machine issues.

2. `options/train/0.yml` is a runnable template. Before formal experiments, tune it based on your data scale:
   - `total_iter`
   - `batch_size_per_gpu`
   - `num_frame`
   - learning rate and scheduler

---

## Acknowledgement

This project is built on top of [BasicSR](https://github.com/XPixelGroup/BasicSR).  
Please refer to `LICENSE` and `LICENSE.txt` for upstream licensing and third-party notices.
