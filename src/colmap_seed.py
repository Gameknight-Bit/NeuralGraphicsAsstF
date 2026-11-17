"""Create training-only COLMAP points using fixed, calibrated camera poses."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np
from PIL import Image

from camera_manifest import capture_id, read_manifest, select_training, training_signature, validate_pose


def quaternion(R):
    # Convert proper rotation to scalar-first quaternion; diagonal cases include 180°.
    w = np.sqrt(max(0, 1 + np.trace(R))) / 2
    if w > 1e-3:
        return np.array([w, R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / [1, 4*w, 4*w, 4*w]
    i = np.argmax(np.diag(R))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.empty(4)
    q[i + 1] = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) / 2
    q[j + 1] = (R[j, i] + R[i, j]) / (4 * q[i + 1])
    q[k + 1] = (R[k, i] + R[i, k]) / (4 * q[i + 1])
    q[0] = (R[k, j] - R[j, k]) / (4 * q[i + 1])
    return q / np.linalg.norm(q)


def values(numbers):
    return " ".join(f"{x:.16g}" for x in numbers)


def run_colmap(executable, command, *args, capture=False):
    located = Path(shutil.which(executable) or executable)
    env = None
    if os.name == "nt":
        if located.suffix.lower() == ".bat":
            located = located.parent / "bin" / "colmap.exe"
        if located.is_file():
            located = located.resolve()
            bundle = located.parent.parent
            if (bundle / "COLMAP.bat").is_file():
                # Preserve the official launcher plugin paths without invoking a command shell.
                env = os.environ.copy()
                for key, directory in (("PATH", located.parent), ("QT_PLUGIN_PATH", bundle / "plugins")):
                    env[key] = os.pathsep.join(filter(None, (str(directory), env.get(key))))
    invocation = [str(located), command, *map(str, args)]
    if not capture:
        print(f"COLMAP {command}", flush=True)
    completed = subprocess.run(invocation, check=True, text=True, env=env,
                               stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.STDOUT if capture else None)
    return completed.stdout


def gpu_option(executable, command, modern, legacy, use_gpu):
    help_text = run_colmap(executable, command, "-h", capture=True)
    for option in (modern, legacy):
        if option in help_text:
            return option, int(use_gpu)
    raise ValueError(f"Cannot identify the GPU option for COLMAP {command}; check your COLMAP build")


def model_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]


def verify_model(model, staged, expected_params, size, max_error):
    cameras = [line.split() for line in model_lines(model / "cameras.txt") if line]
    if (len(cameras) != 1 or cameras[0][1] != "PINHOLE" or tuple(map(int, cameras[0][2:4])) != size
            or not np.allclose(list(map(float, cameras[0][4:])), expected_params, rtol=1e-6, atol=1e-6)):
        raise ValueError("COLMAP changed the calibrated intrinsics")
    poses, keypoints = {}, {}
    lines = model_lines(model / "images.txt")
    if len(lines) != 2 * len(staged):
        raise ValueError("COLMAP model does not have exactly the selected training images")
    for header, observations in zip(lines[::2], lines[1::2]):
        fields = header.split(maxsplit=9)
        image_id, name = int(fields[0]), fields[9]
        R, t = validate_pose(staged[name])
        reference = quaternion(R)
        reference /= np.linalg.norm(reference)
        q = np.array(fields[1:5], dtype=float)
        if fields[8] != cameras[0][0] or not (
                min(np.linalg.norm(reference - q), np.linalg.norm(reference + q)) < 1e-5
                and np.allclose(np.array(fields[5:8], dtype=float), t, atol=1e-6)):
            raise ValueError(f"COLMAP changed the fixed pose for {name}")
        if image_id in poses or any(entry[0] == name for entry in poses.values()):
            raise ValueError(f"Repeated image in model: {name}")
        poses[image_id] = (name, R, t)
        keypoints[image_id] = np.array(observations.split(), dtype=float).reshape(-1, 3)
    fx, fy, cx, cy = expected_params
    points, colors, errors, track_sizes = [], [], [], []
    per_image = dict.fromkeys(staged, 0)
    raw_count = 0
    for line in model_lines(model / "points3D.txt"):
        if not line:
            continue
        raw_count += 1
        fields = line.split()
        point_id = int(fields[0])
        xyz = np.array(fields[1:4], dtype=float)
        rgb = np.array(fields[4:7], dtype=int)
        track = np.array(fields[8:], dtype=int).reshape(-1, 2)
        if (not np.isfinite(xyz).all() or ((rgb < 0) | (rgb > 255)).any()
                or len({capture_id(staged[poses[i][0]]) for i, _ in track}) < 2):
            continue
        residuals = []
        for image_id, point_index in track:
            _, R, t = poses[image_id]
            x, y, z = R @ xyz + t
            u, v, measured_id = keypoints[image_id][point_index]
            if z <= 0 or point_id != measured_id:
                break
            residuals.append(np.hypot(fx * x / z + cx - u, fy * y / z + cy - v))
        if len(residuals) != len(track) or not np.isfinite(residuals).all() or max(residuals) > max_error:
            continue
        points.append(xyz)
        colors.append(rgb)
        errors += residuals
        track_sizes.append(len(track))
        for i, _ in track:
            per_image[poses[i][0]] += 1
    if len(points) < 2:
        raise ValueError("Fewer than two verified points; check overlap, poses, image size and COLMAP match logs")
    stats = dict(raw_points=raw_count, verified_points=len(points),
                 median_reprojection_px=float(np.median(errors)), p95_reprojection_px=float(np.percentile(errors, 95)),
                 median_track_length=float(np.median(track_sizes)),
                 verified_observations_by_view={staged[name]["file"]: count for name, count in per_image.items()})
    return np.asarray(points, np.float32), np.asarray(colors, np.uint8), stats


def build_seed(dataset, output, executable, stride, max_error, use_gpu, image_dir):
    data = read_manifest(dataset)
    frames = select_training(data, stride)
    if len({capture_id(frame) for frame in frames}) < 2:
        raise ValueError("Point triangulation needs at least two distinct translated training captures")
    centers = np.stack([-R.T @ t for R, t in map(validate_pose, frames)])
    if np.max(np.linalg.norm(centers - centers[0], axis=1)) < 1e-6:
        raise ValueError("Training poses have no translational baseline")
    if image_dir is not None and len({Path(frame["file"]).name for frame in frames}) != len(frames):
        raise ValueError("--image-dir requires unique basenames across training views")
    if not np.isfinite(max_error) or max_error <= 0:
        raise ValueError("Maximum reprojection error must be finite and positive")
    extraction = gpu_option(executable, "feature_extractor", "--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu", use_gpu)
    matching = gpu_option(executable, "exhaustive_matcher", "--FeatureMatching.use_gpu", "--SiftMatching.use_gpu", use_gpu)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images, known, sparse, model = (output / name for name in ("images", "known", "sparse", "text"))
    for path in (images, known, sparse, model):
        path.mkdir()
    staged = {}
    size = None
    for n, frame in enumerate(frames):
        source = dataset / frame["file"] if image_dir is None else image_dir / Path(frame["file"]).name
        with Image.open(source) as image:
            if size is None:
                size = image.size
            if image.size != size:
                raise ValueError(f"COLMAP inputs must share one image size: {source}")
        name = f"{n:06d}{source.suffix.lower()}"
        shutil.copyfile(source, images / name)
        staged[name] = frame
    W, H = size
    sx, sy = W / data["width"], H / data["height"]
    K = np.array(data["K"])
    # COLMAP pixel centers start at (.5, .5); Python's start at (0, 0).
    params = (sx * K[0, 0], sy * K[1, 1], sx * (K[0, 2] + .5), sy * (K[1, 2] + .5))
    database = output / "database.db"
    db_flags = ("--database_path", database)
    image_flags = ("--image_path", images)
    run_colmap(executable, "feature_extractor", *db_flags, *image_flags,
               "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", 1,
               "--ImageReader.camera_params", ",".join(str(x) for x in params), *extraction)
    with sqlite3.connect(database) as conn:
        rows = conn.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
    if {row[1] for row in rows} != staged.keys() or len({row[2] for row in rows}) != 1:
        raise ValueError("COLMAP database image list or calibration differs from the staged training split")
    camera_id = rows[0][2]
    (known / "cameras.txt").write_text(f"{camera_id} PINHOLE {W} {H} {values(params)}\n", encoding="utf-8")
    lines = []
    for image_id, name, _ in rows:
        R, t = validate_pose(staged[name])
        q = quaternion(R)
        q /= np.linalg.norm(q)
        lines.append(f"{image_id} {values(q)} {values(t)} {camera_id} {name}\n\n")
    (known / "images.txt").write_text("".join(lines), encoding="utf-8")
    (known / "points3D.txt").touch()
    run_colmap(executable, "exhaustive_matcher", *db_flags, *matching)
    run_colmap(executable, "point_triangulator", *db_flags, *image_flags,
               "--input_path", known, "--output_path", sparse)
    run_colmap(executable, "model_converter", "--input_path", sparse, "--output_path", model, "--output_type", "TXT")
    xyz, rgb, stats = verify_model(model, staged, params, size, max_error)
    stats.update(train_stride=stride, source_image_size=size, max_observation_error_px=max_error,
                 training_files=[frame["file"] for frame in frames], external_poses=True)
    (output / "seed_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    np.savez(output / "point_seed.npz", xyz=xyz, rgb=rgb,
             training_signature=training_signature(data, frames))
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="New workspace for COLMAP and the verified seed")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable or its absolute path")
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--image-dir", type=Path, help="Optional same camera frames at uncropped native size (same basenames)")
    parser.add_argument("--max-error-px", type=float, default=3, help="Maximum residual in COLMAP input pixels")
    parser.add_argument("--gpu", action="store_true", help="Use COLMAP CUDA instead of portable CPU SIFT/matching")
    args = parser.parse_args()
    build_seed(args.dataset, args.output_dir, args.colmap, args.train_stride, args.max_error_px, args.gpu, args.image_dir)


if __name__ == "__main__":
    main()
