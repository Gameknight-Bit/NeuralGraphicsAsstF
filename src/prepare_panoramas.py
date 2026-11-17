"""Convert posed equirectangular panoramas to calibrated pinhole views."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from camera_manifest import validate_pose


def yaw_rotation(degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    # Maps crop coordinates to panorama coordinates, both right/down/forward.
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def perspective_crop(rgb, K, width, height, panorama_from_crop):
    y, x = np.indices((height, width), dtype=float)
    rays = np.stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x)), -1)
    right, down, forward = np.moveaxis(rays @ panorama_from_crop.T, -1, 0)
    h, w = rgb.shape[:2]
    u = (np.arctan2(right, forward) / (2 * np.pi) + 0.5) * w - 0.5
    v = np.clip((np.arctan2(down, np.hypot(right, forward)) / np.pi + 0.5) * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    ax, ay = (u - x0)[..., None], (v - y0)[..., None]
    x1, y1 = (x0 + 1) % w, np.minimum(y0 + 1, h - 1)
    x0 %= w
    top = (1 - ax) * rgb[y0, x0] + ax * rgb[y0, x1]
    bottom = (1 - ax) * rgb[y1, x0] + ax * rgb[y1, x1]
    return np.rint((1 - ay) * top + ay * bottom).astype(np.uint8)


def prepare(spec_path, output, width, height, hfov, yaws):
    if min(width, height) < 1 or not 0 < hfov < 180 or not yaws or not np.isfinite(yaws).all():
        raise ValueError("Use positive image dimensions, HFOV between 0 and 180, and finite yaws")
    if len({yaw % 360 for yaw in yaws}) != len(yaws):
        raise ValueError("Duplicate yaw directions")
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    captures = spec["captures"]
    ids = [str(capture["capture_id"]) for capture in captures]
    files = [(spec_path.parent / capture["file"]).resolve() for capture in captures]
    splits = [capture["split"] for capture in captures]
    if len(ids) != len(set(ids)) or len(set(files)) != len(files) or set(splits) != {"train", "val"}:
        raise ValueError("Require distinct capture IDs and files and both train and val positions")
    poses = [validate_pose(capture) for capture in captures]
    focal = width / (2 * np.tan(np.radians(hfov) / 2))
    K = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    output.mkdir(parents=True, exist_ok=False)
    frames = {"train": [], "val": []}
    for index, (path, capture, split, (R, t)) in enumerate(zip(files, ids, splits, poses)):
        with Image.open(path) as image:
            if image.width != 2 * image.height:
                raise ValueError(f"Expected a full 2:1 equirectangular panorama: {path}")
            rgb = np.asarray(image.convert("RGB"))
        for i, yaw in enumerate(yaws):
            rotation = yaw_rotation(yaw)
            name = f"{split}/{index:05d}_{i:02d}.png"
            (output / split).mkdir(exist_ok=True)
            Image.fromarray(perspective_crop(rgb, K, width, height, rotation)).save(output / name)
            frames[split].append(dict(file=name, capture_id=capture, yaw_deg=yaw,
                                      R_wc=(rotation.T @ R).tolist(), t=(rotation.T @ t).tolist()))
    data = dict(K=K.tolist(), width=width, height=height,
                frames=frames["train"], val_frames=frames["val"])
    (output / "cameras.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Prepared {len(frames['train'])} train / {len(frames['val'])} held-out crops")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON of input panoramas, poses, and capture splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--hfov", type=float, default=100)
    parser.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
    args = parser.parse_args()
    prepare(args.spec, args.output_dir, args.width, args.height, args.hfov, args.yaws)


if __name__ == "__main__":
    main()
