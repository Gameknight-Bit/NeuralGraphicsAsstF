"""Camera geometry and split rules shared by preprocessing and training."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


def read_manifest(folder):
    folder = Path(folder)
    data = json.loads((folder / "cameras.json").read_text(encoding="utf-8-sig"))
    width, height = int(data["width"]), int(data["height"])
    K = np.asarray(data["K"], dtype=float)
    if (min(width, height) <= 0 or K.shape != (3, 3) or not np.isfinite(K).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[[0, 1], [1, 0]], 0)
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("Expected positive dimensions and a shared zero-skew pinhole K")
    seen_files, train_ids, val_ids = set(), set(), set()
    for frames, capture_ids in ((data["frames"], train_ids), (data.get("val_frames", []), val_ids)):
        for frame in frames:
            name = frame["file"]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError(f"Expected a relative image path: {name}")
            resolved = (folder / name).resolve().as_posix().casefold()
            if resolved in seen_files:
                raise ValueError(f"Repeated image path: {name}")
            seen_files.add(resolved)
            capture_ids.add(capture_id(frame))
            validate_pose(frame)
    if train_ids & val_ids:
        raise ValueError(f"Capture positions cross the train/held-out split: {train_ids & val_ids}")
    return data


def validate_pose(frame):
    rotation = np.asarray(frame["R_wc"], dtype=float)
    translation = np.asarray(frame["t"], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError(f"Invalid world-to-camera pose: {frame.get('file', frame.get('capture_id'))}")
    return rotation, translation


def capture_id(frame):
    # Legacy datasets may number train and val images independently.
    return str(frame.get("capture_id", frame["file"]))


def select_training(data, stride=1):
    if stride < 1:
        raise ValueError("Training stride must be positive")
    frames = data["frames"]
    ordered_ids = list(dict.fromkeys(capture_id(frame) for frame in frames))
    selected_ids = set(ordered_ids[::stride])
    return [frame for frame in frames if capture_id(frame) in selected_ids]


def training_signature(data, frames):
    payload = {key: data[key] for key in ("width", "height", "K")}
    payload["frames"] = frames
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
