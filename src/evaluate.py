"""
evaluate.py
-----------
Loads a trained model and the held-out test set, then produces everything
your flowchart calls for at the "SR Image output" branch point:

  - PSNR / SSIM / RMSE          (accuracy metrics vs ground truth)
  - Spectral consistency check  (per-channel histogram correlation)
  - Geospatial consistency check (STUB - see note below)
  - Uncertainty mapping         (Monte Carlo Dropout variance map)

Saves sample before/after/uncertainty images to results/, and writes a
metrics.csv summary. If PostgreSQL is configured (see db.py), also logs the
run there for the dashboard (app.py) to display.

IMPORTANT NOTE ON GEOSPATIAL CONSISTENCY:
EuroSAT_RGB (this project's training data) is plain JPGs with no coordinate
reference system attached - there is nothing to check geospatially on this
dataset. The function below (check_geospatial_consistency) is a working stub
that operates on real Sentinel-2 GeoTIFFs (the kind you download from the
Copernicus browser) using rasterio - use it once you run this model on real
downloaded satellite imagery, not on the EuroSAT training/test images.

Usage:
    python src/evaluate.py --model_path models/srcnn_best.keras
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import model as model_module  # registers custom layers (e.g. PixelShuffle) before loading a saved model


def compute_metrics(hr_true, hr_pred):
    """PSNR, SSIM, RMSE averaged over a batch of images."""
    psnr_scores, ssim_scores, rmse_scores = [], [], []

    for true_img, pred_img in zip(hr_true, hr_pred):
        psnr_scores.append(psnr(true_img, pred_img, data_range=1.0))
        ssim_scores.append(ssim(true_img, pred_img, channel_axis=2, data_range=1.0))
        rmse_scores.append(np.sqrt(np.mean((true_img - pred_img) ** 2)))

    return {
        "psnr_mean": float(np.mean(psnr_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
    }


def check_spectral_consistency(hr_true, hr_pred):
    """
    Compares the per-channel (R/G/B) value distributions between ground truth
    and predicted images using histogram correlation. A score near 1.0 means
    the model preserved realistic color/spectral relationships rather than
    inventing implausible values - important because the problem statement
    requires preserving spectral consistency, not just visual sharpness.
    """
    channel_names = ["Red", "Green", "Blue"]
    results = {}

    for c, name in enumerate(channel_names):
        true_hist = cv2.calcHist([hr_true[..., c].astype(np.float32)], [0], None, [64], [0, 1])
        pred_hist = cv2.calcHist([hr_pred[..., c].astype(np.float32)], [0], None, [64], [0, 1])
        correlation = cv2.compareHist(true_hist, pred_hist, cv2.HISTCMP_CORREL)
        results[f"{name}_histogram_correlation"] = float(correlation)

    return results


def check_geospatial_consistency(geotiff_lr_path, geotiff_sr_path):
    """
    STUB for use with real Sentinel-2 GeoTIFFs (not the EuroSAT training data).
    Verifies the super-resolved output still aligns to the same geographic
    footprint and coordinate reference system as the input - a real SR model
    must not shift or distort pixel geolocation.

    Requires: pip install rasterio
    """
    import rasterio

    with rasterio.open(geotiff_lr_path) as lr_src, rasterio.open(geotiff_sr_path) as sr_src:
        same_crs = lr_src.crs == sr_src.crs
        bounds_match = (
            abs(lr_src.bounds.left - sr_src.bounds.left) < 1e-3
            and abs(lr_src.bounds.top - sr_src.bounds.top) < 1e-3
        )
        return {"same_crs": same_crs, "bounds_match": bounds_match}


def monte_carlo_uncertainty(model, lr_image, n_passes=20):
    """
    Uncertainty mapping via Monte Carlo Dropout: if the model has Dropout
    layers, running it multiple times with dropout still active (training=True)
    gives slightly different outputs each time. Pixel-wise variance across
    those runs is a genuine, defensible uncertainty estimate - it highlights
    which reconstructed regions the model is less confident about
    (per the problem statement's requirement to manage/communicate uncertainty).

    NOTE: the default srcnn/deep_resnet models in model.py don't include
    Dropout layers (kept simple for a fast first build). If you want real
    MC-Dropout uncertainty, add a layers.Dropout(0.1) after a couple of conv
    layers in model.py and retrain. Until then, this function will return a
    near-zero variance map — which is itself an honest result to report:
    "no stochastic uncertainty estimate without dropout in the architecture".
    """
    batch = np.expand_dims(lr_image, axis=0)
    predictions = np.stack([model(batch, training=True).numpy()[0] for _ in range(n_passes)])
    mean_pred = predictions.mean(axis=0)
    variance_map = predictions.var(axis=0)
    return mean_pred, variance_map


def save_sample_visuals(lr, hr_true, hr_pred, variance_map, out_dir, index=0):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(lr)
    axes[0].set_title("Input (bicubic LR)")
    axes[1].imshow(hr_pred)
    axes[1].set_title("Model output (SR)")
    axes[2].imshow(hr_true)
    axes[2].set_title("Ground truth (HR)")
    im = axes[3].imshow(variance_map.mean(axis=-1), cmap="inferno")
    axes[3].set_title("Uncertainty map")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    out_path = os.path.join(out_dir, f"sample_{index}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained SR model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_samples_to_visualize", type=int, default=5)
    parser.add_argument("--log_to_db", action="store_true", help="Also log this run to PostgreSQL (see db.py)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, compile=False)

    lr_test = np.load(os.path.join(args.data_dir, "lr_test.npy"))
    hr_test = np.load(os.path.join(args.data_dir, "hr_test.npy"))
    print(f"Test set: {lr_test.shape[0]} images")
    print(f"  LR shape: {lr_test.shape[1:]}, HR shape: {hr_test.shape[1:]}")

    print("Running inference on test set...")
    hr_pred = model.predict(lr_test, verbose=1)

    if hr_pred.shape != hr_test.shape:
        print("\n" + "=" * 70)
        print("MISMATCH: the model's output size doesn't match this test set's")
        print("HR image size, so accuracy metrics can't be computed.")
        print(f"  Model produced:     {hr_pred.shape[1:]}")
        print(f"  Test set expects:   {hr_test.shape[1:]}")
        print()
        print("This almost always means --data_dir points at data prepared")
        print("for a DIFFERENT model type than the one you're evaluating:")
        print("  - srcnn / deep_resnet (same-size refinement) need data from")
        print("    data_pipeline.py or data_pipeline_geotiff.py")
        print("  - espcn (true upsampling) needs data from")
        print("    data_pipeline_upsample.py")
        print()
        print("Re-run data_pipeline_upsample.py for this model's --scale, or")
        print("point --data_dir at the correct existing processed folder,")
        print("then try evaluate.py again.")
        print("=" * 70)
        return

    print("Computing PSNR / SSIM / RMSE...")
    metrics = compute_metrics(hr_test, hr_pred)
    print(metrics)

    print("Checking spectral consistency...")
    spectral = check_spectral_consistency(hr_test, hr_pred)
    print(spectral)

    print(f"Saving {args.n_samples_to_visualize} sample visualizations with uncertainty maps...")
    for i in range(min(args.n_samples_to_visualize, lr_test.shape[0])):
        _, variance_map = monte_carlo_uncertainty(model, lr_test[i])
        path = save_sample_visuals(lr_test[i], hr_test[i], hr_pred[i], variance_map, args.results_dir, index=i)
        print(f"  Saved {path}")

    # Write metrics summary
    import csv
    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    all_metrics = {**metrics, **spectral, "model_path": args.model_path}
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(all_metrics)
    print(f"Appended metrics to {metrics_path}")

    if args.log_to_db:
        try:
            from db import insert_result
            insert_result(model_name=os.path.basename(args.model_path), **metrics)
            print("Logged run to PostgreSQL.")
        except Exception as e:
            print(f"Could not log to database (this is optional): {e}")

    print("\nDone. Reminder: geospatial consistency check requires real GeoTIFF")
    print("input (see check_geospatial_consistency in this file) - it does not")
    print("apply to the EuroSAT JPG test set used here.")


if __name__ == "__main__":
    main()
