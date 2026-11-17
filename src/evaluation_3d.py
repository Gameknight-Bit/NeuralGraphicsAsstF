"""Evaluate a fitted scene without optimization on a chosen camera split."""

import math

import torch
from torch.nn.functional import conv2d

from fitting_3d import render_view


def image_ssim(a, b):
    """RGB SSIM over valid 11×11 Gaussian windows, input range [0, 1]."""
    size = min(11, *a.shape[:2])
    size -= int(size % 2 == 0)
    x = torch.arange(size, device=a.device, dtype=a.dtype) - size // 2
    window = torch.softmax(-x.square() / (2 * 1.5 ** 2), dim=0)
    kernel = (window[:, None] * window).expand(3, 1, size, size).contiguous()
    a, b = (image.permute(2, 0, 1) for image in (a, b))
    ma, mb, aa, bb, ab = conv2d(torch.stack((a, b, a * a, b * b, a * b)),
                               kernel, groups=3)
    va, vb = (aa - ma.square()).clamp_min(0), (bb - mb.square()).clamp_min(0)
    cov = ab - ma * mb
    return (((2 * ma * mb + .01**2) * (2 * cov + .03**2))
            / ((ma.square() + mb.square() + .01**2) * (va + vb + .03**2))).mean()


@torch.no_grad()
def evaluate_scene(parameters, cameras, pixel_chunk=2048, save_indices=(), include_ssim=False):
    """Return per-view metrics, mean PSNR, and selected RGB renders."""
    if not cameras:
        raise ValueError("Evaluation needs at least one camera")
    rows, images = [], {}
    for i, camera in enumerate(cameras):
        image = render_view(parameters, camera, pixel_chunk)
        mse = ((image - camera["image"]) ** 2).mean().item()
        psnr = -10 * math.log10(mse) if mse > 0 else math.inf
        rows.append(dict(view=camera["file"], mse=mse, psnr=psnr))
        if include_ssim:
            rows[-1]["ssim"] = image_ssim(image, camera["image"]).item()
        if i in save_indices:
            images[i] = image.cpu()
    return rows, sum(row["psnr"] for row in rows) / len(rows), images
