"""Differentiable front-to-back rasterization over a black background."""

import torch

def gaussian_weight(xy, mu, Sigma):
    """
    xy: (P, 2) pixel coordinates.
    mu: (N, 2) Gaussian centers.
    Sigma: (N, 2, 2) positive definite covariance matrices.
    Returns: (P, N), one weight for every pixel/Gaussian pair.
    """
    delta = xy[:, None, :] - mu[None, :, :]  # (P, N, 2)

    # Solve Sigma * q = delta instead of explicitly computing Sigma^-1.
    # Group the P pixel displacements as columns for each Gaussian.
    rhs = delta.permute(1, 2, 0)  # (N, 2, P)
    q = torch.linalg.solve(Sigma, rhs).permute(2, 0, 1)  # (P, N, 2)
    squared_distance = (delta * q).sum(dim=-1)  # (P, N)
    return torch.exp(-0.5 * squared_distance)


def pixel_grid(H, W, device=None, dtype=torch.float32):
    """Return (H*W, 2) coordinates in (x, y) order, flattened row by row."""
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def render_pixels(xy, mu, Sigma, color, opacity, order):
    """Render (P, 2) pixel coordinates with differentiable over compositing."""
    order = torch.as_tensor(order, device=mu.device, dtype=torch.long)
    alpha = (gaussian_weight(xy, mu, Sigma) * opacity[None, :])[:, order]
    
    remaining = torch.cat((alpha.new_ones((len(xy), 1)), 1.0 - alpha), dim=1)
    T = torch.cumprod(remaining, dim=1)[:, :-1]
    return (T * alpha) @ color[order]


def render(mu, Sigma, color, opacity, order, H, W, pixel_chunk=2048):
    """Render N Gaussians as an (H, W, 3) floating-point RGB tensor.

    mu: (N, 2) gaussian centers in pixel coordinates.
    Sigma: (N, 2, 2) positive definite projected covariances.
    color: (N, 3) straight (non-premultiplied) RGB in [0, 1].
    opacity: (N,) values in [0, 1].
    order: permutation of 0..N-1, sorted front to back.
    H, W: positive image height and width.
    """
    if H <= 0 or W <= 0 or pixel_chunk <= 0:
        raise ValueError("H, W, and pixel_chunk must be positive")
    xy = pixel_grid(H, W, device=mu.device, dtype=mu.dtype)
    chunks = [render_pixels(points, mu, Sigma, color, opacity, order)
              for points in xy.split(pixel_chunk)]

    return torch.cat(chunks, dim=0).reshape(H, W, 3)
