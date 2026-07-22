# jetson-feature-research

Standalone research repo (independent of the production C++ tracker) analyzing whether
scale-invariant ResNet18 features can improve template tracking and automatic stereo handoff.

Based on: *Naturally Computed Scale Invariance in the Residual Stream of ResNet18* (CVPRW 2025).

## Workflow
- Code is authored on the PC, pushed to GitHub.
- Jetson pulls and runs (CPU torch is fine for this analysis).
- Errors get pasted back for fixing.

## Jetson environment (verified 2026-07-22)
- Python 3.8.10
- torch 2.4.1 (CPU), torchvision 0.19.1
- numpy 1.24.4, opencv 4.5.4, pillow 7.0.0, matplotlib 3.1.2
- scikit-learn 1.3.2, scipy 1.10.1

**Required before running anything (libgomp TLS fix):**
```bash
export LD_PRELOAD=/home/nvidia/.local/lib/python3.8/site-packages/torch/lib/../../torch.libs/libgomp-804f19d4.so.1.0.0
```
(Already appended to `~/.bashrc`.)

## Data
Each experiment runs against one recorded session folder containing:
`raw_left_frames/`, `raw_right_frames/`, `events.jsonl`, `tracker.json`,
`annotations.json`, `meta.json`.

Pass the session path with `--session`.

## Experiments
| # | File | What it answers |
|---|------|-----------------|
| 2 | `experiments/exp02_scale_stability.py` | Which ResNet block is most scale-invariant (cosine sim across scales) |
| 7 | `experiments/exp07_handoff.py` | Does feature similarity predict a successful handoff |

(More experiments added incrementally: 3 pixel-vs-feature, 4 layer-wise, 5 channel-wise,
6 aspect-ratio, 8 failure, 9 feature-evolution, 10 feature-template-DB.)

## Run
```bash
python3 experiments/exp02_scale_stability.py --session /path/to/2026-07-18T16-40-35
python3 experiments/exp07_handoff.py        --session /path/to/2026-07-18T16-40-35
```
Outputs (plots + CSV) go to `results/<experiment>/`.
