"""Prepare a small KITTI-360 camera-00 section for the existing 3D trainer."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def calibration_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if ':' in line:
            name, value = line.split(':', 1)
            try:
                values[name] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kitti_root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='2013_05_28_drive_0000_sync')
    parser.add_argument('--start-frame', type=int, default=980)
    parser.add_argument('--end-frame', type=int, default=1006)
    parser.add_argument('--spacing', type=float, default=0.5, help='Minimum consecutive selected displacement in metres')
    parser.add_argument('--width', type=int, default=352)
    parser.add_argument('--scene-scale', type=float, default=10, help='Metres per normalized world unit')
    parser.add_argument('--focus-distance', type=float, default=15, help='Forward offset of scene origin from mean training camera, metres')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output already exists; choose a new output directory')
    if min(args.spacing, args.width, args.scene_scale, args.focus_distance) <= 0 or args.end_frame <= args.start_frame:
        parser.error('Use positive sizes/spacing and an increasing frame range')

    root = args.kitti_root.resolve()
    image_dir = root / 'data_2d_raw' / args.sequence / 'image_00/data_rect'
    pose_dir = root / 'data_poses' / args.sequence
    if not pose_dir.exists():
        pose_dir = root / args.sequence  # The official ZIP omits the data_poses parent.
    raw = np.loadtxt(pose_dir / 'poses.txt')
    intrinsics = calibration_values(root / 'calibration/perspective.txt')
    camera_to_imu = np.eye(4)
    camera_to_imu[:3] = calibration_values(root / 'calibration/calib_cam_to_pose.txt')['image_00'].reshape(3, 4)
    rectification = np.eye(4)
    rectification[:3, :3] = intrinsics['R_rect_00'].reshape(3, 3)
    K_original = intrinsics['P_rect_00'].reshape(3, 4)[:, :3]
    source_size = tuple(intrinsics['S_rect_00'].astype(int))

    selected = []
    for row in raw:
        frame = int(row[0])
        if not args.start_frame <= frame <= args.end_frame:
            continue
        source = image_dir / f'{frame:010d}.png'
        if not source.is_file():
            continue
        ego_to_world = np.eye(4)
        ego_to_world[:3] = row[1:].reshape(3, 4)
        camera_to_world = ego_to_world @ camera_to_imu @ np.linalg.inv(rectification)
        if selected and np.linalg.norm(camera_to_world[:3, 3] - selected[-1][2][:3, 3]) < args.spacing:
            continue
        selected.append((frame, source, camera_to_world))
    if len(selected) < 6:
        parser.error('Section needs at least six posed, spatially separated images')

    # Never select the final held-out image using training quality.
    validation_ids = {item[0] for i, item in enumerate(selected) if i % 4 == 2}
    training = [item for item in selected if item[0] not in validation_ids]
    centers = np.stack([item[2][:3, 3] for item in training])
    basis = training[0][2][:3, :3]
    forward = np.stack([item[2][:3, 2] for item in training]).mean(0)
    forward /= np.linalg.norm(forward)
    origin = centers.mean(0) + args.focus_distance * forward
    world_from_normalized = np.eye(4)
    world_from_normalized[:3, :3] = args.scene_scale * basis
    world_from_normalized[:3, 3] = origin

    width = args.width
    height = round(source_size[1] * width / source_size[0])
    sx, sy = width / source_size[0], height / source_size[1]
    K = K_original.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K_original[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K_original[1, 2] + 0.5) * sy - 0.5

    frames, heldout, samples, errors = [], [], [], []
    rng = np.random.default_rng(0)
    probe = rng.uniform(-1.0, 1.0, (32, 3))
    args.output_dir.mkdir(parents=True)
    for frame, source, camera_to_world in selected:
        world_to_camera = np.linalg.inv(camera_to_world)
        transformed = world_to_camera @ world_from_normalized
        normalized = transformed[:3] / args.scene_scale
        rotation, translation = normalized[:, :3], normalized[:, 3]
        # The distributed rectification matrices have six-decimal precision.
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=5e-6)
        # Similarity normalization must preserve projections and positive-Z visibility.
        world_points = probe @ world_from_normalized[:3, :3].T + origin
        original_camera = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        normalized_camera = probe @ rotation.T + translation
        np.testing.assert_allclose(original_camera / args.scene_scale, normalized_camera, atol=1e-10)
        pixels_a = original_camera @ K.T
        pixels_b = normalized_camera @ K.T
        error = np.max(np.abs(pixels_a[:, :2] / pixels_a[:, 2:] - pixels_b[:, :2] / pixels_b[:, 2:]))
        errors.append(float(error))
        if not (normalized_camera[:, 2] > 0).any():
            raise ValueError(f'Initialization volume is behind camera {frame}')
        split = 'val' if frame in validation_ids else 'train'
        relative = f'{split}/{frame:010d}.png'
        target = args.output_dir / relative
        target.parent.mkdir(exist_ok=True)
        with Image.open(source) as image:
            if image.size != source_size:
                raise ValueError(f'Image size does not match calibration: {source}')
            resized = image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
            resized.save(target)
            samples.append((frame, split, resized.copy()))
        record = dict(file=relative, capture_id=str(frame), R_wc=rotation.tolist(), t=translation.tolist())
        (heldout if split == 'val' else frames).append(record)

    assert {x['file'].split('/')[-1] for x in frames}.isdisjoint(x['file'].split('/')[-1] for x in heldout)
    data = dict(width=width, height=height, K=K.tolist(), frames=frames, val_frames=heldout,
                convention='X_cam = R_wc @ X_normalized + t; positive Z forward')
    (args.output_dir / 'cameras.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    distance = sum(np.linalg.norm(b[2][:3, 3] - a[2][:3, 3]) for a, b in zip(selected, selected[1:]))
    manifest = dict(sequence=args.sequence, camera='00', requested_range=[args.start_frame, args.end_frame],
                    training_frame_ids=[item[0] for item in training], validation_frame_ids=sorted(validation_ids),
                    image_size=[width, height], minimum_selected_displacement_m=args.spacing,
                    selected_path_length_m=float(distance), source_root=str(root),
                    pose_source=str(pose_dir / 'poses.txt'), calibration_source=str(root / 'calibration'),
                    pose_construction='T_world_imu @ T_imu_cam_raw @ inverse(R_rect_00)',
                    normalization=dict(world_from_normalized=world_from_normalized.tolist(),
                                       metres_per_unit=args.scene_scale, focus_distance_m=args.focus_distance,
                                       origin_uses='training camera poses only'),
                    max_projection_invariance_error_pixels=max(errors),
                    initialization='Existing random-box baseline; no SfM, LiDAR, FSGS, or depth priors',
                    supplied_pose_provenance='Official poses use full-sequence optimization with GPS/LiDAR/image information; not training-only SfM')
    (args.output_dir / 'section_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    columns = 4
    sheet = Image.new('RGB', (columns * width, ((len(samples) + columns - 1) // columns) * (height + 22)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (frame, split, image) in enumerate(samples):
        x, y = i % columns * width, i // columns * (height + 22)
        sheet.paste(image, (x, y + 22))
        draw.text((x + 4, y + 3), f'{frame}  {split}', fill='black')
    sheet.save(args.output_dir / 'section_preview.jpg')
    (args.output_dir / 'README.md').write_text(
        '# KITTI-360 small-section baseline\n\n'
        f'Sequence {args.sequence}, rectified camera 00, frames {selected[0][0]}-{selected[-1][0]}. '
        f'{len(frames)} training and {len(heldout)} held-out views at {width} x {height}; '
        f'selected path length {distance:.2f} m.\n\n'
        'Images were resized without modifying originals. Cameras use supplied official poses and calibration, '
        'including perspective rectification. A saved similarity transform maps normalized scene coordinates '
        'back to KITTI world coordinates. The transform was chosen using training poses.\n\n'
        'This tests the existing randomly initialized trainer. It does not implement FSGS or point-cloud '
        'initialization. Supplied poses incorporate full-sequence GPS/LiDAR/image information, so this is '
        'a supplied-pose experiment, not a training-only camera reconstruction. See section_manifest.json '
        'for exact frame IDs, provenance, and the normalization transform.\n\n'
        'Sources: [KITTI documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php) '
        'and [official camera conversion](https://github.com/autonomousvision/kitti360Scripts/blob/'
        '7ecc14eab6fa30e5d2ac71ad37fed2bb4b0b8073/kitti360scripts/helpers/project.py).\n',
        encoding='utf-8')
    print(f'Prepared {len(frames)} train + {len(heldout)} held-out views, {width}x{height}, {distance:.2f} m')
    print(f'Max projection invariance error: {max(errors):.3g} pixels')


if __name__ == '__main__':
    main()
