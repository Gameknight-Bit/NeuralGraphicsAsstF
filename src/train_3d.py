"""Train a 3D Gaussian scene and save its checkpoint and training metrics."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch

from camera_manifest import read_manifest, select_training, training_signature
from evaluation_3d import evaluate_scene
from fitting_3d import fit_scene
from image_utils import get_device, save_image
from point_initialization import load_seed
from scene_data import load_cameras


def report_split(folder, split, parameters, cameras, chunk):
    selected = sorted({0, len(cameras) // 2, len(cameras) - 1})
    rows, mean_psnr, renders = evaluate_scene(parameters, cameras, chunk, selected, include_ssim=True)
    comparisons = []
    for i in selected:
        name = f"{split}_compare_{i:03d}.png"
        save_image(torch.cat((cameras[i]["image"].cpu(), renders[i]), dim=1), folder / name)
        comparisons.append(f"{cameras[i]['file']} - {rows[i]['psnr']:.2f} dB\n\n![Ground truth left, render right]({name})")
    with (folder / f"{split}_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_ssim = sum(row["ssim"] for row in rows) / len(rows)
    score = f"{split}: **{mean_psnr:.2f} dB PSNR, {mean_ssim:.4f} SSIM** over {len(rows)} views."
    print(score.replace("**", ""))
    return "\n\n".join([score, *comparisons])


def main():
    parser = argparse.ArgumentParser(description="Fit 3D Gaussians to posed training views")
    parser.add_argument("dataset", type=Path, help="Extracted folder containing cameras.json and train/")
    parser.add_argument("--num-gaussians", type=int, default=4000)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--position-lr", type=float, help="Optional lower rate for moving centers")
    parser.add_argument("--loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--point-seed", type=Path, help="Verified point_seed.npz; num-gaussians caps the seed size")
    parser.add_argument("--train-stride", type=int, default=1, help="Use every nth training capture; keep all its crops")
    parser.add_argument("--densify", action="store_true", help="Enable the existing assignment clone/split/prune")
    parser.add_argument("--budget", type=int, default=None, help="Maximum Gaussian count when densifying")
    parser.add_argument("--densify-every", type=int, default=200)
    parser.add_argument("--grad-threshold", type=float, default=2e-4)
    parser.add_argument("--eval-heldout", action="store_true", help="Load and render held-out images after training")
    parser.add_argument("--pixel-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    args = parser.parse_args()
    if min(args.num_gaussians, args.iters, args.pixel_chunk, args.train_stride) <= 0 or args.lr <= 0:
        parser.error("Gaussian count, iterations, pixel batch size, and lr must be positive")
    torch.manual_seed(args.seed)
    device = args.device or get_device()
    data = read_manifest(args.dataset)
    selected = select_training(data, args.train_stride)
    if args.eval_heldout and not data.get("val_frames"):
        parser.error("--eval-heldout requested but cameras.json has no held-out views")
    if args.point_seed is not None:
        signature = training_signature(data, selected)
        points = load_seed(args.point_seed, signature, args.num_gaussians, args.seed)
    else:
        points = None
    cameras = load_cameras(args.dataset, device=device, split="train", train_stride=args.train_stride)
    initial_count = len(points[0]) if points is not None else args.num_gaussians
    budget = args.budget if args.budget is not None else args.num_gaussians
    print(f"Device: {device}; training views: {len(cameras)}; initial Gaussians: {initial_count}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    parameters, history = fit_scene(
        cameras, args.num_gaussians, args.iters, args.lr, args.pixel_chunk, camera_seed=args.seed,
        points=points, objective=args.loss, position_lr=args.position_lr, densification=args.densify,
        budget=budget, densify_every=args.densify_every, grad_threshold=args.grad_threshold,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else None
    final_count = len(parameters["mu3"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "parameters": {name: value.cpu() for name, value in parameters.items()},
        "history": history, "num_gaussians": final_count, "initial_count": initial_count, "iters": args.iters,
        "lr": args.lr, "position_lr": args.position_lr, "seed": args.seed, "dataset": str(args.dataset),
        "training_files": [frame["file"] for frame in selected],
        "train_stride": args.train_stride, "loss": args.loss,
        "point_seed": str(args.point_seed) if args.point_seed is not None else None,
        "densification": args.densify, "budget": budget, "densify_every": args.densify_every,
        "grad_threshold": args.grad_threshold,
        "training_seconds": training_seconds, "peak_cuda_allocated_bytes": peak_bytes,
    }, args.output_dir / "scene.pt")

    splits = [report_split(args.output_dir, "train", parameters, cameras, args.pixel_chunk)]
    if args.eval_heldout:
        heldout = load_cameras(args.dataset, device=device, split="val")
        splits.append(report_split(args.output_dir, "heldout", parameters, heldout, args.pixel_chunk))
    peak = f"{peak_bytes / 2**20:.1f} MiB" if peak_bytes is not None else "n/a"
    report = (
        f"# 3D training results\n\nGaussians: {initial_count} → {final_count}; Adam steps: {args.iters}; "
        f"{args.loss.upper()} objective; lr: {args.lr}; position lr: {args.position_lr or args.lr}; seed: {args.seed}.\n\n"
        f"Training time: {training_seconds:.1f} s; peak CUDA allocated: {peak}.\n\n"
        "PSNR and SSIM are arithmetic means over per-view float renders. "
        "Comparison panels show ground truth on the left and the render on the right.\n\n"
    )
    (args.output_dir / "results.md").write_text(report + "\n\n".join(splits) + "\n", encoding="utf-8")
    print(f"Saved checkpoint, metrics, comparisons, and report in {args.output_dir}")


if __name__ == "__main__":
    main()
