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
| `src/train_3d.py` | Train one scene from random or colored point initialization; optionally densify and evaluate held-out views |
| `src/prepare_kitti_section.py` | Resize posed KITTI-360 camera-00 frames into a shared camera manifest |
| `src/prepare_panoramas.py` | Sample perspective crops from posed 2:1 panoramas and keep each capture in one split |
| `src/colmap_seed.py` | Triangulate and check a training-only colored cloud from calibrated cameras |
| `src/point_initialization.py` | Validate a point seed and set Gaussian centers, RGB, initial scales, and opacity |
| `src/camera_manifest.py` | Validate pinhole camera geometry and enforce capture-based splits and stride |
| `src/compare_3d.py` | Compare fixed-count and densified scenes on training and held-out views |
| `src/make_gif.py` | Reload a checkpoint, evaluate it, and render a camera-arc GIF |
| `src/fitting_3d.py` | Initialize, project, and optimize a shared 3D Gaussian cloud |
| `src/gaussians_3d.py` | Quaternion rotations, 3D covariance, and pinhole projection |
| `src/densification_3d.py` | Clone, split, prune, and preserve Adam state when the cloud changes |
| `src/renderer.py` | Evaluate projected Gaussian weights and blend pixels front to back |
| `src/evaluation_3d.py` | Per-view MSE, PSNR, optional SSIM, and selected renders |
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

## KITTI-360: the first calibrated street run

The local section is `datasets/kitti_0000_section_0980_1006`: 11 training
positions, three held-out positions (984, 992, 1000), 352 × 94 training
images, and about 8.77 m of travel. All images come from rectified front
camera 00. This data can test interpolation along the forward route; it does
not show how well we can look sideways or turn around. The supplied poses
use full-sequence KITTI optimization (GPS, LiDAR, and imagery). Only training
RGB enters our feature matching, seed construction, and training.

### Install COLMAP (Windows and macOS)

COLMAP is a structure-from-motion tool. It detects matching features across
images and triangulates colored 3D landmarks from their camera rays. It can
also estimate camera poses; on this KITTI section, use the supplied camera
poses and ask it to triangulate only. This avoids asking a very small
forward-motion set to recover its own camera trajectory. The cloud supplies
initial Gaussian centers, colors, and approximate local spacing. It is not a
finished reconstruction. See the [fixed-pose workflow](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses).

- **Windows:** Download the x64 Windows archive from the [official COLMAP
  releases](https://github.com/colmap/colmap/releases). The non-CUDA build is
  sufficient for this small job. Extract the entire archive together. Open
  its folder, run `./COLMAP.bat -h` in PowerShell, then pass the full path
  to that launcher with `--colmap` below. COLMAP's CPU mode is independent
  of the Python trainer using CUDA.
- **macOS:** With [Homebrew](https://brew.sh/) installed, run
  `brew install colmap` and `colmap -h`. The Python seed command can use
  `colmap` from PATH. The Windows CUDA pins in `req.txt` are not for
  macOS; to run these Python tools there, use a separate environment with
  supported builds of PyTorch, NumPy, and Pillow.

After installing COLMAP, run this from the project root in PowerShell
and replace the COLMAP path with yours:

```powershell
$scene = 'datasets/kitti_0000_section_0980_1006'
$native = 'datasets/download_2d_perspective/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect'
$colmap = 'C:/Tools/COLMAP/COLMAP.bat'
./venv/Scripts/python.exe src/colmap_seed.py $scene --image-dir $native --colmap $colmap --output-dir outputs/kitti_seed
./venv/Scripts/python.exe src/train_3d.py $scene --point-seed outputs/kitti_seed/point_seed.npz --num-gaussians 512 --iters 600 --lr 0.005 --eval-heldout --device cuda --output-dir outputs/kitti_seeded
```

`--image-dir` uses the native 1408 × 376 frames to match features;
only the selected training frame names are copied. It accepts uncropped
images with the same filenames, view, and geometry as the resized images,
not extra cameras. Without it, COLMAP uses the prepared training images.
On macOS, use your environment's `python` and omit the explicit
`--colmap` when COLMAP is on PATH; choose `--device cpu` for the first
Python smoke test (or try `mps` on Apple Silicon). Use fresh output folders.

Open `seed_report.json`: check the accepted point count, residuals in the
COLMAP source pixels, number of views per track, and views without verified
observations. To inspect the actual geometry, open COLMAP's GUI, use
**File → Import model** on the seed workspace's `sparse` folder, and
look for consistent structure across the overlapping façade/road. Tiny
residuals by themselves do not prove good depth under near-forward motion.
No COLMAP seed from this real section has been validated yet.

First compare against the same train command without `--point-seed` at
the same *actual* initial Gaussian count, then change one setting at a time.
`--loss l1`, `--position-lr` and `--densify --budget N` are optional.
`--eval-heldout` reads held-out RGB only after optimization. Camera stride
`--train-stride 1`, `2`, or `4` selects 11, 6, or 3 training positions in
this section while preserving the same three held-out positions. Regenerate
the seed with the same stride for each run; the trainer rejects seeds built
from a different selection/calibration.

The six-position CUDA smoke runs used seed 0, 220 steps, and 256 random
starting Gaussians. Fixed-count MSE scored **13.53 dB / 0.3586 SSIM** on the
three held-out views; L1, a lower position learning rate, and assignment
densification together scored **13.57 dB / 0.3872 SSIM** (384 Gaussians
at the end). Training took about 27 and 29 seconds respectively. Peak CUDA
tensor allocation was about 72 and 99 MiB, excluding memory held outside
PyTorch's tensor allocator. These runs exercised the code paths. They
changed several variables at once and provide no evidence for a specific
method or for FSGS.

## Panorama mode and dataset format

`scene_data.py` currently expects a `cameras.json` file with shared `K`,
`width`, `height`, training `frames`, and held-out `val_frames`. Each frame
provides an image `file`, world-to-camera rotation `R_wc`, and translation `t`.
Set `capture_id` to the original position for every crop; older single-view
manifests use the image path as the identity.
The convention is `X_camera = R_wc @ X_world + t`, with positive Z forward.
The supplied `spheres/` dataset follows this format.

Use `python src/prepare_panoramas.py captures.json --output-dir <new-folder>`
for future **posed, full 2:1 equirectangular panoramas**. Each entry in the
input JSON's `captures` array needs `capture_id`, `file` (relative to the
JSON or absolute), `split` (`train` or `val`), `R_wc`, and `t`. The
zero-longitude ray looks forward through the image center and the image
midline is the equator. By default it makes four yaw crops per capture;
`--yaws`, `--hfov`, `--width` and `--height` let you choose coverage.
If the panorama source does not supply poses, this tool does not estimate
them. Its output keeps all crops from a position together and does not
invent any translation. It was checked on synthetic panoramas; don't run
it on KITTI's already-perspective images.

The Python renderer still uses constant per-Gaussian RGB and a dense
pixel/Gaussian calculation. The scene checkpoint reloads in Python; a Rust
export format and navigation are later work. The GIF command renders an
orbit about the world origin. Use the [FSGS implementation plan](FSGS_IMPLEMENTATION_PLAN.md)
for the learning exercises and which baseline changes to try before FSGS.
