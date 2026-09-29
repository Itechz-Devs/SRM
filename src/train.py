"""
train.py
--------
Trains the super-resolution CNN on the preprocessed LR/HR pairs.

YOU run this file - it is not run automatically. Training time depends on
your machine: with CPU only and the default "srcnn" model, expect roughly
1-3 minutes per epoch on the full ~24,000-image training set (faster if you
used --limit_per_class in data_pipeline.py for a smaller test run first,
which is recommended before committing to a full run).

Usage:
    python src/train.py --model srcnn --epochs 20
    python src/train.py --model deep_resnet --epochs 30 --batch_size 16
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoids needing a display - saves plots to file instead
import matplotlib.pyplot as plt

import tensorflow as tf

# Force TensorFlow to use only GPU 0 (your system has 2 detected GPUs via
# DirectML - by default TensorFlow may pick either/both; this pins it to
# GPU:0 specifically). This must run BEFORE any other TensorFlow operation.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Restricted TensorFlow to: {gpus[0]}")
    except RuntimeError as e:
        print(f"Could not restrict GPU (must be set before any TF op ran): {e}")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model import build_model


def ssim_metric(y_true, y_pred):
    """Higher is better (max 1.0) - tracked during training just to watch it,
    not used directly as the loss (see combined_loss below)."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def sobel_edges(img):
    """
    Computes Sobel edge maps (one per color channel) - this highlights
    boundaries where pixel values change sharply: building outlines, road
    edges, water/land boundaries, field boundaries. Smooth areas (open
    fields, forest canopy, calm water interiors) produce near-zero edges.
    tf.image.sobel_edges returns both x and y gradient directions per
    channel; we combine them into a single edge-strength map.
    """
    edges = tf.image.sobel_edges(img)  # shape: (batch, H, W, channels, 2)
    edge_magnitude = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1) + 1e-8)
    return edge_magnitude


def edge_loss(y_true, y_pred):
    """
    Penalizes the model more where GROUND-TRUTH edges are strong (i.e.
    real structural boundaries) and its prediction doesn't match them
    sharply. This is what pushes buildings/roads/water edges to come out
    sharper, without needing any labeled "this is a building" data - it
    only needs to know where edges naturally occur in the true image.
    """
    true_edges = sobel_edges(y_true)
    pred_edges = sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(true_edges - pred_edges))


def edge_sharpness_metric(y_true, y_pred):
    """Tracked during training to watch edge quality directly - higher
    true-edge-strength captured in the prediction is better. Not used as
    the loss itself, just a visibility metric."""
    return tf.reduce_mean(sobel_edges(y_pred))


def combined_loss(y_true, y_pred):
    """
    Three terms working together:

    1. MSE - plain pixel accuracy. Alone, this mathematically rewards
       blurry 'safe average' output (a well-documented SR failure mode).
    2. SSIM loss - rewards structural similarity, not just raw pixel
       distance - a standard fix for MSE's blur bias.
    3. Edge loss - specifically rewards matching sharp boundaries
       (buildings, roads, water edges, field boundaries) rather than
       smooth regions. This is what makes structures - the things a
       human eye actually focuses on in a satellite image - come out
       visibly sharper, without needing any object-labeled dataset.

    Weights: mostly MSE for overall accuracy, with SSIM and edge terms
    pulling toward sharper, more structurally faithful output. Raise
    edge_weight further (e.g. 0.3) for even more aggressive structure
    sharpening, at some risk of amplifying noise in smooth areas.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    edge = edge_loss(y_true, y_pred)

    mse_weight = 0.7
    ssim_weight = 0.15
    edge_weight = 0.15
    return mse_weight * mse + ssim_weight * ssim_loss + edge_weight * edge


_vgg_feature_extractor = None


def get_vgg_feature_extractor():
    """
    Loads a pretrained VGG16 (trained on ImageNet - ordinary photos, not
    satellite imagery) and exposes an early-to-mid layer's features. This
    requires internet access on first use to download the pretrained
    weights (~58MB) - only needed if you enable --perceptual_loss.
    """
    global _vgg_feature_extractor
    if _vgg_feature_extractor is None:
        vgg = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        vgg.trainable = False
        _vgg_feature_extractor = tf.keras.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block2_conv2").output
        )
    return _vgg_feature_extractor


def perceptual_loss(y_true, y_pred):
    """
    Compares deep features (from a network pretrained to recognize objects
    in ordinary photos) rather than raw pixels - this is what most modern
    photo-realistic super-resolution models use, since it rewards
    natural-looking TEXTURE rather than exact pixel matching. Real
    improvement over pixel-only losses, but: (a) needs internet access to
    download VGG16's weights on first use, (b) VGG was trained on ordinary
    photos, not satellite imagery, so its notion of "natural texture" is a
    reasonable but imperfect proxy here, and (c) it roughly doubles
    training time per step.
    """
    vgg = get_vgg_feature_extractor()
    true_prep = tf.keras.applications.vgg16.preprocess_input(y_true * 255.0)
    pred_prep = tf.keras.applications.vgg16.preprocess_input(y_pred * 255.0)
    true_features = vgg(true_prep)
    pred_features = vgg(pred_prep)
    return tf.reduce_mean(tf.square(true_features - pred_features))


def build_loss_fn(use_perceptual=False, perceptual_weight=0.1):
    """
    Returns the actual loss function to use, with perceptual loss as an
    OPT-IN addition on top of combined_loss - kept optional so a normal
    training run stays fast and doesn't require internet access unless
    you specifically ask for it via --perceptual_loss.
    """
    if not use_perceptual:
        return combined_loss

    def loss_with_perceptual(y_true, y_pred):
        base = combined_loss(y_true, y_pred)
        perceptual = perceptual_loss(y_true, y_pred)
        return base + perceptual_weight * perceptual

    return loss_with_perceptual


def load_processed_data(data_dir):
    lr_train = np.load(os.path.join(data_dir, "lr_train.npy"))
    hr_train = np.load(os.path.join(data_dir, "hr_train.npy"))
    lr_val = np.load(os.path.join(data_dir, "lr_val.npy"))
    hr_val = np.load(os.path.join(data_dir, "hr_val.npy"))
    return (lr_train, hr_train), (lr_val, hr_val)


def plot_training_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train")
    axes[1].plot(history.history["val_mae"], label="val")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the super-resolution CNN")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "deep_resnet", "espcn"])
    parser.add_argument("--scale", type=int, default=4, help="Upsampling factor - only used by --model espcn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--perceptual_loss", action="store_true",
                         help="Add VGG-based perceptual loss for more natural texture. "
                              "Needs internet access (downloads VGG16 weights, ~58MB, first "
                              "time) and roughly doubles training time per step.")
    parser.add_argument("--model_suffix", type=str, default=None,
                         help="Optional label added to the saved checkpoint name, e.g. "
                              "'ir', 'sar', 'thermal' - lets the app tell these apart from "
                              "the standard RGB model and offer them as separate options.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading preprocessed data...")
    (lr_train, hr_train), (lr_val, hr_val) = load_processed_data(args.data_dir)
    print(f"  Train pairs: {lr_train.shape[0]}, Val pairs: {lr_val.shape[0]}")
    channels = lr_train.shape[-1]

    # IMPORTANT: use (None, None, channels) here, NOT the concrete training
    # patch size (e.g. 64x64). Every model in model.py is fully
    # convolutional, so declaring a flexible input shape means the SAVED
    # model can run on an image of any size at inference time - matching
    # its exact training patch dimensions would silently lock inference to
    # that one fixed size, defeating the point.
    input_shape = (None, None, channels)

    # For espcn, auto-detect the scale this data was built for (written by
    # data_pipeline_upsample.py) so you don't have to remember to pass
    # --scale manually and risk it not matching the actual data.
    scale = args.scale
    scale_file = os.path.join(args.data_dir, "scale.txt")
    if args.model == "espcn" and os.path.exists(scale_file):
        with open(scale_file) as f:
            detected_scale = int(f.read().strip())
        if detected_scale != args.scale:
            print(f"Note: using scale={detected_scale} detected from {scale_file} (overriding --scale {args.scale})")
        scale = detected_scale

    model_label = f"{args.model}_x{scale}" if args.model == "espcn" else args.model
    if args.model_suffix:
        model_label += f"_{args.model_suffix}"
    if args.perceptual_loss:
        model_label += "_perceptual"
    print(f"Building model: {model_label}")
    model = build_model(args.model, input_shape=input_shape, scale=scale)
    if args.perceptual_loss:
        print("Perceptual loss ENABLED - downloading VGG16 weights if not cached (needs internet)...")
    loss_fn = build_loss_fn(use_perceptual=args.perceptual_loss)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss_fn, metrics=["mae", ssim_metric, edge_sharpness_metric])
    model.summary()

    checkpoint_path = os.path.join(args.out_dir, f"{model_label}_best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
        CSVLogger(os.path.join(args.out_dir, f"{model_label}_training_log.csv")),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]

    print("Starting training - this is the long step, let it run.")
    history = model.fit(
        lr_train, hr_train,
        validation_data=(lr_val, hr_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history, os.path.join(args.out_dir, f"{model_label}_training_curves.png"))
    print(f"\nDone. Best model saved to: {checkpoint_path}")
    print("Next step: python src/evaluate.py --model_path", checkpoint_path)


if __name__ == "__main__":
    main()
