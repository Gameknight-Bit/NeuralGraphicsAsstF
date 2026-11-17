"""Initialize the training cloud from verified colored 3D points."""

import numpy as np
import torch


def load_seed(path, signature, limit, seed):
    with np.load(path, allow_pickle=False) as cloud:
        if cloud["training_signature"].item() != signature:
            raise ValueError("Point seed came from a different training selection or camera calibration")
        xyz = cloud["xyz"].astype(np.float32)
        rgb = cloud["rgb"]
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or rgb.dtype != np.uint8
            or len(xyz) < 2 or not np.isfinite(xyz).all() or limit < 2):
        raise ValueError("Expected at least two finite XYZ points with uint8 RGB and a limit of at least two")
    _, unique = np.unique(xyz, axis=0, return_index=True)
    if len(unique) < 2:
        raise ValueError("Expected at least two distinct XYZ points")
    rows = np.sort(np.random.default_rng(seed).choice(unique, min(limit, len(unique)), replace=False))
    return xyz[rows], rgb[rows].astype(np.float32) / 255


@torch.no_grad()
def point_parameters(points, device):
    xyz, rgb = (torch.as_tensor(value, device=device, dtype=torch.float32) for value in points)
    # Estimate local spacing using the closest three neighbors (or fewer for tiny fixtures).
    k = min(3, len(xyz) - 1)
    row_ids = torch.arange(len(xyz), device=device)
    scales = []
    # Bound the temporary distance matrix even when the cloud is larger than a smoke test.
    for rows in row_ids.split(max(1, min(512, 1_000_000 // len(xyz)))):
        distances = torch.cdist(xyz[rows], xyz)
        distances[torch.arange(len(rows), device=device), rows] = torch.inf
        scales.append(distances.topk(k, largest=False).values.square().mean(1).sqrt())
    scale = torch.cat(scales).clamp_min(1e-4)
    scale = scale.clamp_max(torch.quantile(scale, .95))
    quat = torch.zeros(len(xyz), 4, device=device)
    quat[:, 0] = 1
    return dict(mu3=xyz.clone(), log_s=scale.log()[:, None].repeat(1, 3), quat=quat,
                color=torch.logit(rgb, eps=1e-3), op_raw=xyz.new_full((len(xyz),), -2.1972246))
