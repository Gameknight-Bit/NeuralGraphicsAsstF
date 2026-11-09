"""Train a 3D Gaussian scene and save its checkpoint and training metrics."""

import argparse
import csv
from pathlib import Path

import torch

from evaluation_3d import evaluate_scene
from fitting_3d import fit_scene
from image_utils import get_device, save_image
from scene_data import load_cameras


def main():
    parser = argparse.ArgumentParser(description="Fit 3D Gaussians to posed training views")
    parser.add_argument("dataset", type=Path, help="Extracted folder containing cameras.json and train/")
    parser.add_argument("--num-gaussians", type=int, default=4000)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pixel-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    args = parser.parse_args()
    if min(args.num_gaussians, args.iters, args.pixel_chunk) <= 0 or args.lr <= 0:
        parser.error("Gaussian count, iterations, pixel batch size, and lr must be positive")
    torch.manual_seed(args.seed)
    device = args.device or get_device()
    cameras = load_cameras(args.dataset, device=device, split="train")
    print(f"Device: {device}; training views: {len(cameras)}; Gaussians: {args.num_gaussians}")
    parameters, history = fit_scene(cameras, args.num_gaussians, args.iters, args.lr,
                                    args.pixel_chunk, camera_seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "parameters": {name: value.cpu() for name, value in parameters.items()},
        "history": history, "num_gaussians": args.num_gaussians, "iters": args.iters,
        "lr": args.lr, "seed": args.seed, "dataset": str(args.dataset),
    }, args.output_dir / "scene.pt")

    selected = sorted({0, len(cameras) // 2, len(cameras) - 1})
    rows, mean_psnr, renders = evaluate_scene(parameters, cameras, args.pixel_chunk, selected)
    comparisons = []
    for i in selected:
        name = f"train_compare_{i:03d}.png"
        save_image(torch.cat((cameras[i]["image"].cpu(), renders[i]), dim=1), args.output_dir / name)
        comparisons.append(f"{cameras[i]['file']} - {rows[i]['psnr']:.2f} dB\n\n![Ground truth left, render right]({name})")
    with (args.output_dir / "train_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["view", "mse", "psnr"])
        writer.writeheader()
        writer.writerows(rows)
    report = (
        f"# 3D training results\n\nMean training-view PSNR: **{mean_psnr:.2f} dB** over {len(rows)} views.\n\n"
        f"Gaussians: {args.num_gaussians}; Adam steps: {args.iters}; lr: {args.lr}; seed: {args.seed}.\n\n"
        "PSNR is the arithmetic mean over per-view float renders. "
        "Comparison panels show ground truth on the left and the render on the right.\n\n"
    )
    (args.output_dir / "results.md").write_text(report + "\n\n".join(comparisons) + "\n", encoding="utf-8")
    print(f"Mean training PSNR: {mean_psnr:.2f} dB; saved checkpoint, metrics, comparisons, and report in {args.output_dir}")


if __name__ == "__main__":
    main()
