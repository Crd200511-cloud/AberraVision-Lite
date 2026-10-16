"""Step 1: change defocus/noise and inspect physically simulated images.

Learning experiments never overwrite the future frozen research configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.optics import make_psf, simulate_observation
from src.patterns import learning_pattern


def draw_comparison(result: dict, *, title: str | None = None):
    """Common intensity mapping; noise remains unclipped in numeric outputs."""
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.8), layout="constrained")
    peak_reference = float(make_psf(defocus_waves=0.0).max())
    panels = [("Original scene", result["image"]),
              ("PSF (common log scale)", result["psf"]),
              ("Blur only", result["blurred"]),
              ("Blur + noise", result["noisy"])]
    for axis, (label, data) in zip(axes, panels):
        if label.startswith("PSF"):
            c = data.shape[0] // 2
            axis.imshow(np.maximum(data[c-32:c+33, c-32:c+33], 1e-15),
                        cmap="gray", norm=LogNorm(vmin=peak_reference*1e-5, vmax=peak_reference),
                        interpolation="nearest")
        else:
            axis.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label, fontsize=11)
        axis.set_axis_off()
    fig.suptitle(title or "Step 1: optical defocus and additive noise", fontsize=13)
    return fig


def run_experiment(defocus: float = 0.15, noise: float = 0.01, seed: int = 42,
                   output: str | Path = ROOT / "results" / "learning") -> dict:
    result = simulate_observation(learning_pattern(), defocus_waves=defocus,
                                  noise_sigma=noise, seed=seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "current_data.npz", **{key: result[key] for key in
                        ("image", "psf", "blurred", "noisy")})
    metadata = dict(result["metadata"])
    metadata.update({"purpose": "learning_only_not_formal_research", "source": "procedural pattern",
                     "display_range": [0, 1], "raw_noisy_min": float(result["noisy"].min()),
                     "raw_noisy_max": float(result["noisy"].max())})
    (out / "current_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
    fig = draw_comparison(result, title=f"Defocus = {defocus:.2f} waves RMS | Noise sigma = {noise:.3f} | Seed = {seed}")
    fig.savefig(out / "current_comparison.png", dpi=150)
    plt.close(fig)
    return result


def draw_learning_grid(output: str | Path = ROOT / "results" / "learning") -> Path:
    """Two controlled experiments: keep one variable fixed each time."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    image = learning_pattern()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 7.1), layout="constrained")
    for j, a in enumerate((0.0, 0.15, 0.30)):
        result = simulate_observation(image, defocus_waves=a, noise_sigma=0.01, seed=42)
        axes[0, j].imshow(result["noisy"], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"Defocus {a:.2f} waves | sigma 0.01", fontsize=10)
    for j, sigma in enumerate((0.0, 0.01, 0.03)):
        result = simulate_observation(image, defocus_waves=0.15, noise_sigma=sigma, seed=42)
        axes[1, j].imshow(result["noisy"], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"Defocus 0.15 waves | sigma {sigma:.2f}", fontsize=10)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle("Learning experiments: change one parameter at a time", fontsize=13)
    dest = out / "parameter_comparison.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defocus", type=float, default=0.15, help="Defocus coefficient in RMS waves")
    parser.add_argument("--noise", type=float, default=0.01, help="Gaussian standard deviation, intensity units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "learning")
    parser.add_argument("--gallery", action="store_true", help="Also create the six-condition teaching grid")
    args = parser.parse_args()
    try:
        result = run_experiment(args.defocus, args.noise, args.seed, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print("Learning experiment only; no AI reconstruction or benchmark score.")
    print(f"Figure: {(args.output / 'current_comparison.png').resolve()}")
    print(f"Parameters: defocus={args.defocus} waves, sigma={args.noise}, seed={args.seed}")
    print(f"PSF sum: {result['psf'].sum():.12f}")
    if args.gallery:
        print(f"Comparison grid: {draw_learning_grid(args.output).resolve()}")


if __name__ == "__main__":
    main()
