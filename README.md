# Street View Gaussian Splatting

Existing Python 3D Gaussian splatting baseline for the 15-474 final project.
The source now focuses on posed-image loading, 3D fitting, densification,
evaluation, and saved-scene previews.

## Training Data:

KITTI-360:
https://www.cvlibs.net/datasets/kitti-360/index.php

## Source map

| File | Responsibility |
|---|---|
| `src/train_3d.py` | Train one fixed-count scene; save checkpoint, metrics, and render comparisons |
| `src/compare_3d.py` | Compare fixed-count and densified scenes on training and held-out views |
| `src/make_gif.py` | Reload a checkpoint, evaluate it, and render a camera-arc GIF |
| `src/fitting_3d.py` | Initialize, project, and optimize a shared 3D Gaussian cloud |
| `src/gaussians_3d.py` | Quaternion rotations, 3D covariance, and pinhole projection |
| `src/densification_3d.py` | Clone, split, prune, and preserve Adam state when the cloud changes |
| `src/renderer.py` | Evaluate projected Gaussian weights and blend pixels front to back |
| `src/evaluation_3d.py` | Per-view MSE, PSNR, and selected renders |
| `src/scene_data.py` | Load calibrated training and validation cameras |
| `src/image_utils.py` | Device selection and RGB image input/output |

The renderer works in image space after projecting the 3D scene. Its 2D
Gaussian weights remain necessary for 3D splatting. The standalone 2D image
fitting exercises and assignment count sweeps have been removed.

## Setup

Run commands from this folder. On Windows:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r req.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The existing `req.txt` pins CUDA 12.8 PyTorch builds. The project environment
uses Python 3.9.12 and PyTorch 2.8.0+cu128.

## Existing workflows

Train the supplied spheres scene:

```powershell
.\venv\Scripts\python.exe src/train_3d.py spheres --num-gaussians 1024 --iters 1500 --device cuda
```

Compare fixed-count training with densification, including held-out metrics:

```powershell
.\venv\Scripts\python.exe src/compare_3d.py spheres --initial-count 512 --budget 1024 --iters 1500 --device cuda
```

Reload a scene and make the existing camera-arc preview:

```powershell
.\venv\Scripts\python.exe src/make_gif.py spheres outputs/train/scene.pt --device cuda
```

Default output folders are `outputs/train`, `outputs/compare`, and
`outputs/preview`. Each command also accepts `--output-dir` and `--help`.
Training saves `scene.pt`; comparison saves `plain_scene.pt` and
`densified_scene.pt`. Checkpoints contain raw parameter tensors, training
history, and run settings. Metrics are computed on float renders before image
conversion. Validation images do not enter the optimization loss.

## Dataset format and current limits

`scene_data.py` currently expects a `cameras.json` file with shared `K`,
`width`, `height`, training `frames`, and held-out `val_frames`. Each frame
provides an image `file`, world-to-camera rotation `R_wc`, and translation `t`.
The convention is `X_camera = R_wc @ X_world + t`, with positive Z forward.
The supplied `spheres/` dataset follows this format.

This cleanup preserves the assignment's existing algorithms: random centers
in `[-1.5, 1.5]^3`, constant per-Gaussian RGB, and a pixel-batched PyTorch
renderer. Panorama reprojection, COLMAP ingestion/point-cloud initialization,
Rust-viewer export/integration, SSIM, capture-density experiments, and speed
or memory profiling remain future project work. The GIF command renders an
orbit about the world origin; it is not the proposed free-navigation viewer.
No new reconstruction or rendering functionality was added in this cleanup.
