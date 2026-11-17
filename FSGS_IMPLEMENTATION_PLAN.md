# FSGS implementation plan for the street-view project

Updated November 17, 2026. The prerequisites below distinguish features tested on KITTI,
features tested on synthetic data, and FSGS exercises still yours to implement.

## 1. Goal and reading guide

Build a small, convincing street reconstruction from sparse panorama **positions**, then export the learned Gaussians to the proposed Rust/WebGPU viewer. Judge improvements on withheld capture positions and continuous camera motion, not only training images.

This plan combines three kinds of evidence:

- **Paper:** [FSGS, arXiv v2, including supplementary material](https://arxiv.org/html/2312.00451v2), by Zhu, Fan, Jiang, and Wang. Read Sections 3, 4.3, 5, and 6 alongside this plan.
- **Released implementation:** [VITA-Group/FSGS](https://github.com/VITA-Group/FSGS), inspected at commit [`a536a64`](https://github.com/VITA-Group/FSGS/tree/a536a64c5b366b1088be64eeadf9e791ca26897c). This predates paper v2; differences below do not establish which recipe generated every published result.
- **Our project recommendations:** the panorama protocol, staged rollout, resource limits, acceptance checks, and viewer contract below. These are engineering choices for this project, not claims that the authors used them.

Project scope follows the attached proposal: panoramic images -> COLMAP/calibrated cameras -> Python training -> saved splats -> Rust viewer, with three capture densities and fixed held-out positions. FSGS is a quality extension; retain a usable baseline throughout development.

### The central idea

FSGS combines neighbor-based growth, real-view depth guidance, and virtual-view regularization. Its ablations support testing each separately; unobserved occluded content remains a limitation. [Paper, Sections 3, 4.3, and 5](https://arxiv.org/html/2312.00451v2)

### Your implementation lab

**Start with a baseline you can explain.** Follow the COLMAP commands in the
[README](README.md), and open `seed_report.json` and the sparse model. Pick a
lamppost or a building edge seen in several training photos. Explain how
matching image features and intersecting their calibrated rays gives a seed
point, and why near-parallel rays make depth uncertain even with a small
pixel residual. Compare random vs. colored-point initialization at the same
actual count, seed, and optimization settings. Save one held-out panel and
record whether the change reduced a geometry failure or mostly recovered
color faster.

**1. Learn what proximity growth adds.** In `densification_3d.py`, write a
pure function that proposes midpoints before touching parameters or Adam.
Try four centers along x at 0, 2, 4, and 10. With three neighbors and a
strict mean-distance cutoff of 5, only 0 and 10 should propose new centers.
With existing centers and duplicate proposals removed, expect x = 1, 5, 6,
7. Deduplication is our adaptation. Compare this rule with the starter
gradient clone/split rule, which may have no gradient signal in a gap with
no splat. Then wire midpoint insertion into the existing optimizer resize
path, zero moments for new rows, cap the count, and plot the proposed edges
on the real cloud. Check whether they fill a wall or connect the wall to a
tree across empty air. See [Section 5](#5-fsgs-component-a-proximity-guided-gaussian-growth).

**2. Add depth to the RGB renderer, then give it a real-view prior.** In
`renderer.py`, expose compositing weights once and reuse them for RGB,
camera-Z depth, and coverage. For two splats at Z = 2 and 6 with alpha 0.5
each, predict the weights before running code: 0.5 and 0.25, accumulated
depth 2.5, coverage 0.75. Verify depth moves when you change the means or
opacities. Next implement `1 - correlation` and test increasing, reversed,
affine-shifted, and constant target arrays before using an estimator. Cache
priors for training images and check their near/far ordering. Run the toy
depth loss on a complete small image first: averaging correlations of
pixel chunks changes both the statistic and its gradient. See
[Section 6](#6-fsgs-component-b-rendered-depth-and-real-view-geometry-guidance).

**3. Sample virtual cameras and follow both gradient paths.** Draw the
first virtual camera halfway between two training centers; only then add
small bounded perturbations. Use the rendered RGB to predict relative
depth, then compare that prior with depth composited from the same
Gaussians. On a tiny fixture replace the frozen estimator with a
differentiable stand-in and verify gradients reach both the RGB and
depth branches. Repeat with the RGB-derived prior detached and explain
what changes. This prior is an estimate from our render, not a new
photograph. See [Section 7](#7-fsgs-component-c-virtual-cameras-and-online-depth-guidance).

Change one component at a time and write down the failure you expect it to
fix before running each ablation. Reuse views 984, 992, and 1000 for the
forward KITTI experiment. If you tune on those views, call them validation
views in your report and reserve another segment for a final test if possible.

## 2. Where our implementation stands

These observations come from the local source inspected for this plan.

| Component | Current behavior | Required next work |
|---|---|---|
| `scene_data.py`, `camera_manifest.py` | Shared calibrated intrinsics, capture IDs, validation of poses/splits, training-only capture stride | Verify real panorama capture/pose input |
| `prepare_panoramas.py` | Posed equirectangular-to-pinhole reprojection; synthetic seam and camera-center checks pass | Exercise the tool on a permitted panorama set |
| `colmap_seed.py` | Known-pose triangulation workflow, seed provenance, and observation residual checks; verified on synthetic text model | Run COLMAP on the actual native-resolution KITTI training frames |
| `fitting_3d.py` | Random or colored-point initialization, optional position LR, MSE or L1 | Tune a seeded baseline; add schedules, combined losses |
| `gaussians_3d.py` | Quaternion covariance and pinhole projection | Retain and verify conventions against imported cameras |
| `renderer.py` | Dense pixel/Gaussian interactions; RGB only | Differentiable RGB/depth/coverage; efficient rasterization |
| `densification_3d.py` | Clone/split from world-position gradients; opacity pruning; Adam-state resizing | Screen-space statistics, proximity growth, explicit growth schedule |
| `evaluation_3d.py` | MSE, PSNR, optional SSIM, and selected images | Position-level aggregation when crop totals differ |
| `train_3d.py` | Optional clone/split, held-out metrics after fitting, training time and CUDA peak allocation | Add explicitly named FSGS variants and provenance |
| `compare_3d.py` | Plain versus densified comparison | Eventually compare the controlled variants in Section 11 |
| `make_gif.py` | Saved-scene orbit preview | Retain as a diagnostic; add a separate street-path preview later |

The current real scene is the forward-facing KITTI-360 camera-00 section,
frames 980–1006. Six-position CUDA runs checked fixed and densified
training plus checkpoint reload. No real point seed or FSGS variant has
been run yet. This section can test forward-route interpolation, but it
cannot validate panoramic coverage or free movement to the side.

## 3. First prerequisite: trustworthy street data

**Project recommendation.** Do this before adding regularization. Optimization cannot tell us whether a bad image came from a poor pose, moving traffic, or inadequate scene geometry.

### 3.1 Split original capture positions before preprocessing

1. Assign each panorama a stable `capture_id`, location/order, image identity, and split.
2. Reserve validation positions for tuning and final test positions for the report, where the dataset size permits.
3. Make three nested training-position subsets, such as dense, medium, and sparse spacing. Record actual distances when available.
4. Keep every perspective crop from one panorama in that panorama's split.
5. Exclude held-out imagery from feature matching, point triangulation, dense stereo, training depth caches, and pseudo-view generation. Construct each density's initial cloud from that density's allowed training images.
6. If externally supplied poses were estimated using a larger image collection, disclose that provenance. Do not describe the experiment as fully training-only SfM.

Extra crops add angular coverage, not new camera centers or translational parallax. Count panorama positions as the main sparsity measure and report the crop count separately.

### 3.2 Reproject panoramas consistently

- Generate perspective crops with known field of view, pixel dimensions, and orientation relative to the panorama.
- Use identical crop directions/FOV across capture positions initially. Avoid excessive overlap that repeatedly weights the same rays.
- Preserve one optical center per panorama. Compose crop rotations with the panorama pose; do not introduce translations between crops.
- Store the projection convention, camera axes, and any resizing transformation. Update `fx`, `fy`, `cx`, and `cy` when resizing or cropping.
- Inspect stitching seams, exposure changes, blurred faces, moving cars, sky, and vehicle/rig regions. Start with a mostly static segment; document any masks and use them consistently.

**Acceptance gate:** overlay reprojected 3D points on crops from several positions. Check both near and distant structures. Verify that panoramas at distinct positions produce real baseline and plausible point tracks.

### 3.3 Build and verify initialization

The released repository describes optional dense COLMAP stereo after calibration and uses a fused cloud for its training setup. A reproduction must record whether it uses sparse triangulated points or fused stereo points. [Official data-preparation notes](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/README.md)

For our first baseline, use the reliable training-only cloud we can actually obtain. Do not block the entire project on dense stereo if the sparse cloud is usable. Save cloud provenance, reconstruction settings, point count, and a point-cloud preview.

Use the project's convention `X_camera = R_wc @ X_world + t`, with positive camera Z forward. Recover optical centers using `C = -R_wc.T @ t`; the translation vector `t` is not the world-space camera position. Choose one scene normalization and apply it to cameras and points together; retain its inverse for export.

## 4. Second prerequisite: a sound 3DGS baseline

### 4.1 Initialize from observed geometry

The released `create_from_pcd` uses input positions/colors, neighbor-derived log-scales, identity quaternions, and opacity 0.1. Its appearance parameters separate the constant SH term from higher-order terms. [Initialization code](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/scene/gaussian_model.py)

**Our implementation tasks:** replace the random cube initializer for street runs; validate finite positions, positive scales, and color ranges; retain random initialization only for controlled diagnostics. If temporarily retaining RGB logits, encode point colors with a clamped inverse sigmoid. Do not insert RGB values directly into logits or confuse those logits with SH coefficients.

Begin with constant color while debugging geometry. Treat SH support as a separate baseline upgrade so that its effects are measurable and its viewer representation is explicit.

### 4.2 Improve the optimization contract

Use separate optimizer groups and save their schedules. The released configuration provides the following reference values; they are not universal street-scene settings. [Optimization arguments](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/arguments/__init__.py)

| Parameter | Released reference value |
|---|---:|
| Position learning rate | 0.00016 initially; 0.0000016 finally, with spatial scaling |
| Appearance learning rate | 0.0025 |
| Opacity learning rate | 0.05 |
| Scale learning rate | 0.005 |
| Rotation learning rate | 0.001 |

For photometric training use `0.8 * L1 + 0.2 * (1 - SSIM)` as the reference objective; continue reporting MSE/PSNR for evaluation. The official loss helper uses an 11-pixel SSIM window by default. [Loss helpers](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/utils/loss_utils.py)

**Our implementation cautions:** averaging independent pixel-chunk SSIM scores changes the objective, because SSIM needs neighborhoods. Use a full image or correctly overlapped spatial tiles. Keep masks and reduction rules explicit. Save per-term losses so a dominating prior can be detected.

### 4.3 Make rendering scalable before large growth experiments

Our renderer allocates tensors proportional to pixel count times Gaussian count. At 640 x 480 and 60,000 Gaussians, one float32 interaction matrix alone would occupy **73.7 GB**. A 4,096-pixel chunk still needs about **983 MB** for one such matrix, before other intermediates and gradients.

**Project decision:** keep the current renderer for tiny numerical checks; make an efficient differentiable rasterizer a prerequisite for realistic street-scale growth. The official rendering wrapper exposes RGB, depth, projected positions, and visibility/radius information. [Renderer interface](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/gaussian_renderer/__init__.py)

Require an adapter whose outputs have documented layouts and units. Check gradients against the small reference renderer. Record CUDA/PyTorch/compiler requirements before choosing a backend; do not assume an old custom extension will build in our current environment. Rust inference and Python training can use different rasterizers, but they must agree on projection and compositing.

**Acceptance gate:** fit a small calibrated scene, reload its checkpoint, and reproduce its renders. Establish held-out metrics and GPU memory/time before enabling FSGS components.

## 5. FSGS component A: proximity-guided Gaussian growth

Paper-style growth uses three non-self nearest neighbors, mean Euclidean distance, and midpoint insertion above a proximity threshold. [Paper, Section 3.2](https://arxiv.org/html/2312.00451v2#S3.SS2)

For centers `mu_i`, write the intended paper-style rule as:

```text
neighbors(i) = 3 nearest other centers
P_i = mean(||mu_i - mu_j|| for j in neighbors(i))
if P_i > proximity_threshold:
    propose mu_new = (mu_i + mu_j) / 2 for each neighbor j
```

### 5.1 Paper/code distinction that affects reproduction

The released `proximity` function additionally requires a large Gaussian scale, uses `5 * scene_extent` in its distance test, and runs only before iteration 2,000. The split function has a separate distance condition using `dist_thres * scene_extent`. New midpoint splats copy destination scale/opacity, use identity rotation, and start with zero SH features. [Growth implementation](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/scene/gaussian_model.py)

The underlying KNN kernel returns **mean squared** neighbor distances, not the paper's mean Euclidean distance. [KNN kernel](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/submodules/simple-knn/simple_knn.cu)

**Our choice must be named:** start with a transparent `paper_style` growth policy and an explicit distance unit. If reproducing released behavior, implement a separate `released_code` policy. Do not claim these thresholds are interchangeable. Scaling a scene by `s` scales Euclidean distances by `s` and squared distances by `s^2`.

### 5.2 Implementation tasks and checks

- Recompute neighbors after positions/counts change. Use a spatial index or accelerated KNN rather than an all-pairs distance matrix at realistic counts.
- Handle fewer than four Gaussians, duplicate centers, non-finite values, and self-neighbor exclusion.
- Preserve surviving Adam moments; initialize new rows' moments to zero. Resize all auxiliary statistics together.
- Keep clone/split densification as a separate operation. Our current world-position gradient score is not the released view-space densification statistic.
- Use normalized identity quaternions. Literal zero quaternions are unsafe in implementations that divide by their norm.
- If retaining our RGB representation, document how zero-SH initialization is translated; do not pretend a zero RGB logit is identical to an arbitrary zero-SH convention.
- Log proposed/accepted insertions, clones, splits, prunes, scales, and point count at every refinement pass.

**Our street adaptation:** introduce a configurable growth budget and inspect proposals that bridge disconnected surfaces, such as a tree and a wall. Deduplication, visibility restrictions, and surface-consistency gates are optional adaptations to ablate, not unreported parts of FSGS.

**Acceptance gate:** a hand-constructed point set yields the expected neighbors and exact midpoint locations. After a growth/pruning pass, one optimizer step succeeds and every parameter/statistic array has the correct count.

## 6. FSGS component B: rendered depth and real-view geometry guidance

### 6.1 Extend the renderer contract

Retain camera-space depth before sorting. Use the same front-to-back compositing weights for RGB and depth:

```text
w_i = alpha_i * product(1 - alpha_j for j before i)
RGB(p) = sum_i w_i(p) * color_i
D_sum(p) = sum_i w_i(p) * z_camera_i
A(p) = sum_i w_i(p)
```

This is the paper's accumulated-depth convention. Dividing `D_sum` by `A` produces a different estimator; if trying it, expose and record that variant. [Paper, Equation 7](https://arxiv.org/html/2312.00451v2#S3.SS3)

**Our checks:** use a one-splat scene and a two-layer scene with known ordering. Verify RGB/depth share weights, depth has finite gradients to geometry, and empty/low-coverage pixels are handled consistently. Do not substitute Euclidean distance from the camera for camera-space Z without recording the change.

### 6.2 Cache monocular priors on training crops

The released helper loads MiDaS `DPT_Hybrid`, freezes its parameters, normalizes RGB, resizes to 384 x 512 for prediction, and resizes the prediction back. [Depth helper](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/utils/depth_utils.py)

**Our tasks:** cache predictions only for the actual training crops and save estimator/checkpoint identity plus preprocessing. Visually verify near/far ordering. Treat the output as relative inverse-depth-like structure, not meters. If choosing another estimator, keep that experiment separate.

### 6.3 Implement the loss with the correct direction

The released trainer minimizes `1 - PearsonCorrelation`. Its real-view term selects the smaller loss from correlations against negated MiDaS output and against `1 / (MiDaS + 200)`. Its pseudo-view term uses negated MiDaS output. [Training loss](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/train.py)

**Our robust loss contract:** center valid pixel values, compute covariance divided by standard deviations with an epsilon, and minimize `1 - corr`. Minimizing positive correlation would drive the wrong behavior. Resolve the printed objective using the implemented loss direction. [Paper Equation 8](https://arxiv.org/html/2312.00451v2#S3.SS4), [released loss](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/train.py)

Define valid pixels before reduction: finite prior, finite rendered depth, sufficient coverage, and the selected static-region mask. Log excluded fractions. Skip/report views with too few valid pixels or near-zero variance. These masking/stability rules are our additions and must be consistent across compared variants.

Global correlation is not the average of per-chunk correlations. For chunked training, implement mathematically consistent whole-view statistics or use a smaller full-image loss path. Verify its value and gradient against an unchunked reference.

**Acceptance gate:** affine-related depth arrays with positive scale have near-zero correlation loss, reversed ordering has high loss, constant arrays stay finite, and geometry receives nonzero gradients. Then test real-view depth guidance alone.

## 7. FSGS component C: virtual cameras and online depth guidance

The paper interpolates nearby cameras with translation noise. The released utilities instead contain dataset-specific random LLFF and 360-degree paths. [Paper, Section 3.3](https://arxiv.org/html/2312.00451v2#S3.SS3), [released pose utilities](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/utils/pose_utils.py)

**Our street adaptation:** sample modest translations between neighboring *capture positions*, pairing compatible viewing directions. Do not choose two crops at the same panorama center and call the resulting rotation interpolation a new baseline. Interpolate camera centers and rotations in a consistent camera-to-world representation, then convert back. Do not average rotation-matrix entries directly.

Constrain early virtual cameras to the observed corridor; scale perturbations to local capture spacing. Reject cameras inside obvious geometry or looking far outside observed coverage. These are project safeguards, not a reproduction of a published street-view sampler.

At each scheduled virtual-camera step:

1. Render RGB and differentiable depth from the current scene.
2. Feed the **rendered RGB** to the frozen monocular estimator.
3. Compare its relative-depth output with rendered depth.
4. Add this geometry term to the real-view objective.

There is no real RGB target at that virtual camera. Do not train a photometric loss against our own rendered RGB or treat generated observations as independent ground truth.

### The gradient distinction matters

In the released helper, `mode='train'` retains gradients through the estimator's input while keeping its weights frozen. Therefore the pseudo loss can reach the scene through both RGB->estimator and rendered depth. Calling `no_grad()` or detaching the estimated pseudo depth changes this behavior. [Depth-helper gradient branches](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/utils/depth_utils.py)

**Project recommendation:** make `full_input_gradient` and `detached_teacher` explicit variants if memory forces that comparison. Test that estimator weights do not change and that the chosen scene gradient paths are present. Cache real-image priors; pseudo priors must be refreshed as their rendered inputs evolve.

**Acceptance gate:** virtual cameras use only permitted training poses; rendered RGB/depth align; priors have correct ordering; gradients and memory are inspected; held-out geometry improves over the real-depth-only run.

## 8. Configuration and training schedule

These are **released argument defaults**, unless the row says otherwise. They are a reference to audit, not tuned values for our scene. [Pinned configuration](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/arguments/__init__.py)

| Setting | Reference |
|---|---|
| Total iterations | 10,000 |
| Densification interval / start / stop | 100 / after 500 / before 10,000 |
| Gradient threshold / opacity-prune threshold | 0.0005 / 0.005 |
| Pseudo-view window | after 2,000, before 9,500 |
| Pseudo sampling interval | 10; README LLFF example overrides it to 1 |
| Real-depth weight | 0.05 |
| Pseudo-depth weight | 0.5; README Mip-NeRF360 example uses 0.03 |
| Configured SH maximum | 3 |
| Configured opacity-reset interval | 3,000 |

The released loop ramps pseudo-depth weight over its first 500 iterations and increases active SH degree every 500 iterations. Its opacity reset timing is offset from pseudo-view start. [Loop implementation](https://github.com/VITA-Group/FSGS/blob/a536a64c5b366b1088be64eeadf9e791ca26897c/train.py)

Supplementary text instead gives SH degree 4, opacity resets at 2,000/5,000/7,000, and an inconsistent pruning inequality. [Paper supplementary Section 6](https://arxiv.org/html/2312.00451v2)

**Our recommendation:** use explicitly configured, tested behavior and report which recipe was followed.

**Our rollout:** first verify losses/growth in very small runs; then attempt the reference-length schedule on a reduced-resolution scene. A 300-step diagnostic never reaches a 2,000-step pseudo-view start. For faster diagnostics shorten schedules deliberately and label the run accordingly. Log resets and refinement events rather than inferring them from defaults.

### Intended iteration order

This is a proposed project loop, not a verbatim copy of the released optimizer ordering:

```text
load split manifest, calibrated cameras, training-only initial points
initialize Gaussians and optimizer; cache real-image depth priors
for each iteration:
    sample real training camera
    render RGB, depth, and coverage
    compute real RGB loss and optional real-depth loss
    if pseudo stage and sampling interval permit:
        sample permitted virtual camera; render RGB and depth
        predict relative depth from rendered RGB with chosen gradient policy
        add ramped pseudo-depth loss
    backpropagate; collect projected-position growth statistics
    update scene parameters
    if refinement is scheduled:
        clone/split; apply selected proximity policy; prune
        preserve optimizer state and resize auxiliary arrays
    apply configured opacity resets and learning-rate/SH schedules
    log diagnostics; save checkpoints and validation snapshots
evaluate the selected checkpoint once on final test positions
export and compare Python/Rust renders
```

If matching released optimizer/refinement ordering exactly, implement that as a documented reproduction setting rather than mixing orders silently.

## 9. Implementation order and file responsibilities

Rows distinguish baseline work already prepared from planned changes.
New filenames are suggestions; avoid creating parallel versions of the same trainer.

| Order | Existing module or suggested addition | Work package | Completion evidence |
|---|---|---|---|
| 1 | `prepare_panoramas.py`, `scene_data.py`, `colmap_seed.py` | Capture IDs, crops, pose validation, and triangulation are prepared | Inspect actual KITTI points and test real panoramas later |
| 2 | `fitting_3d.py` | Colored initialization and optional position LR are prepared | Compare seeded and random training on real data |
| 3 | `renderer.py` and backend adapter | Efficient RGB/depth/coverage; projected-point statistics | Reference comparisons and depth-gradient checks |
| 4 | Suggested `losses.py`; `train_3d.py` | L1 and SSIM evaluation are separate now; add an SSIM training term as a controlled baseline variant | Stable calibrated baseline |
| 5 | `densification_3d.py` | KNN growth policy, count budget, statistic/Adam updates | Toy-cloud checks and growth logs |
| 6 | Suggested `depth_prior.py`; `losses.py` | Frozen estimator, training caches, correlation | Depth ordering, finite gradients, real-depth ablation |
| 7 | Suggested `pseudo_views.py`; `fitting_3d.py` | Street virtual cameras, online priors, warm-up/ramp | Pose audit and pseudo-view ablation |
| 8 | `evaluation_3d.py`; `compare_3d.py` | Reproducible variant/density experiments | Metrics tables, plots, and saved camera paths |
| 9 | Suggested `export_scene.py`; Rust viewer | Versioned export and matching rendering | Python/Rust parity and measured navigation FPS |

Start by completing the real-data checks in rows 1 and 2. Keep the small
spheres diagnostic while adding one feature at a time.

## 10. RTX 2060 resource plan

**Project recommendations for the available 6 GB GPU:**

- Begin with a short segment and low-resolution perspective crops. Measure before increasing either image count or Gaussian count.
- Keep unused camera images/priors on CPU where practical. A cache of all high-resolution views consumes memory even before rasterization.
- Cache real-image depth once. Schedule online pseudo inference sparingly while diagnosing performance; changing its frequency is an explicit experimental setting.
- Expect frozen-estimator input gradients to retain activations. Freezing weights alone does not eliminate that memory cost.
- Use accelerated KNN and tiling; avoid full pairwise neighbor matrices and the current dense rendering path at large counts.
- Track peak allocated/reserved CUDA memory, Gaussian count, iteration time, and depth-estimator time. Set a growth cap from measured headroom rather than copying a large benchmark's count.
- Save resumable checkpoints including optimizer, step, schedules, RNG state, normalization, and split identity. Current scene exports are not automatically full training-resume checkpoints.

The published rendering speed is not a promise for our Python renderer, GPU, street scene, or Rust viewer. Measure the actual final setup.

## 11. Experiments that establish whether it works

**Project experiment design:** use one calibrated scene and fixed data splits first. Keep initialization, resolution, seeds, training budget, and baseline upgrades constant across comparisons.

| Run | Baseline | Proximity growth | Real depth | Pseudo depth |
|---|---|---|---|---|
| A | Improved standard-3DGS-style baseline | Off | Off | Off |
| B | Same baseline | On | Off | Off |
| C | Same baseline | On | On | Off |
| D | Same baseline | On | On | On |

Then repeat the selected comparison across the three capture densities. For fair density experiments, rebuild initialization from each allowed training subset. If an external common point cloud is used instead, label that as a separate protocol.

Record per-view metrics, aggregate per capture position, and then average across positions so extra crops do not quietly receive extra weight. Report PSNR and SSIM, with LPIPS if time permits; also training time, peak GPU memory, final Gaussian count, and viewer FPS at 640 x 480. Use identical evaluation images, masks, and rendering settings for all runs.

Use validation positions for thresholds and checkpoint selection. Keep final test imagery out of tuning. If the small dataset only supports one held-out set, state that limitation rather than presenting tuned results as untouched test performance.

For visual assessment save the same training views, held-out views, RGB/depth/coverage maps, and a continuous path between real positions. Inspect poles, building edges, road boundaries, distant detail, floaters, and changes as the camera translates. Virtual images without real photographs cannot receive ordinary ground-truth PSNR.

### Failure-driven debugging order

| Symptom | Inspect first | Next controlled change |
|---|---|---|
| Doubled edges or inconsistent building alignment | Pose convention, crop calibration, dynamic content | Fix data before changing loss weights |
| Good training images, poor held-out views | Split leakage, geometry coverage, oversized splats | Test C/D against A with identical data |
| Growth fills space between unrelated objects | KNN edges and proposal locations | Test distance/visibility gates as adaptations |
| Geometry becomes flat or inverted | Prior ordering, depth definition, correlation sign | Validate loss on synthetic depth arrays |
| Early virtual views look nonsensical | Warm-up, pose bounds, coverage | Delay/restrict pseudo sampling; inspect intermediate renders |
| Memory spikes after adding priors | Estimator input graph and rasterizer intermediates | Reduce resolution/frequency; explicitly test detached-teacher variant |
| Bright seams across panorama crops | Exposure, panorama stitching, crop overlap | Improve capture preprocessing and sampling balance |
| Fast metrics but poor navigation | Visibility changes along fixed path | Inspect coverage and missing observations; restrict demo corridor |

## 12. Export, viewer, and definition of done

**Project contract:** export versioned means, scales/covariances, normalized quaternions, opacity, and RGB or SH appearance. Document activated values versus logits/log-scales, quaternion order, SH basis/order, coordinate axes, and scene normalization. Include a known camera and expected Python render.

Verify Rust output against that reference before optimizing viewer speed. FSGS's training priors should not require a depth network at inference: the viewer consumes the learned Gaussian representation.

The quality milestone is complete when:

- [ ] A training-only calibrated cloud and split manifest are reproducible.
- [ ] The improved baseline trains, reloads, and renders with documented conventions.
- [ ] Depth, correlation, growth, and optimizer-resizing checks pass independently.
- [ ] A-D comparisons show the contribution of each enabled component.
- [ ] Three capture-density settings use the same held-out positions without imagery leakage.
- [ ] Quality and resource measurements include failures, not only favorable views.
- [ ] The exported scene agrees with the Python reference and supports the planned street-path demo.

**First implementation target:** one short, well-calibrated segment with point-cloud initialization and a scalable baseline renderer. Add real-view depth guidance and proximity growth as isolated experiments, then virtual-view guidance. This order is our dependency-driven recommendation; the experiment table remains the comparison protocol.

## Source index

Links throughout this file identify whether a detail comes from the paper or released code. Implementation links are pinned to the inspected commit.

- [FSGS project page](https://zehaozhu.github.io/FSGS/)
- [Paper PDF, arXiv v2](https://arxiv.org/pdf/2312.00451v2)
- [Paper HTML and supplementary material](https://arxiv.org/html/2312.00451v2)
- [Released code snapshot](https://github.com/VITA-Group/FSGS/tree/a536a64c5b366b1088be64eeadf9e791ca26897c)
- Local project reference: `C:/Users/jacej/Downloads/Neural_Graphics_Project_Proposal.pdf`.
- Local baseline inspected: this project's `src/` files and `README.md` on November 9, 2026.
