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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model import build_model


def ssim_metric(y_true, y_pred):
    """Higher is better (max 1.0) - tracked during training just to watch it,
    not used directly as the loss (see combined_loss below)."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def combined_loss(y_true, y_pred):
    """
    Plain MSE (mean squared error) mathematically rewards a model for
    producing blurry, 'safe average' pixel values - this is THE most common
    cause of blurry super-resolution output, not a training mistake.

    Adding an SSIM term (which measures structural/edge similarity, not just
    raw pixel distance) pushes the model to preserve sharper structure
    instead of just minimizing average pixel error. This is a standard,
    well-documented fix used in real super-resolution research.

    alpha controls the balance - 0.8/0.2 (mostly MSE, some SSIM) is a solid
    starting point. Increase the SSIM weight (e.g. 0.6/0.4) for even sharper
    but potentially noisier results.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    alpha = 0.8
    return alpha * mse + (1.0 - alpha) * ssim_loss


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
    parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "deep_resnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading preprocessed data...")
    (lr_train, hr_train), (lr_val, hr_val) = load_processed_data(args.data_dir)
    print(f"  Train pairs: {lr_train.shape[0]}, Val pairs: {lr_val.shape[0]}")
    input_shape = lr_train.shape[1:]

    print(f"Building model: {args.model}")
    model = build_model(args.model, input_shape=input_shape)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=combined_loss, metrics=["mae", ssim_metric])
    model.summary()

    checkpoint_path = os.path.join(args.out_dir, f"{args.model}_best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
        CSVLogger(os.path.join(args.out_dir, f"{args.model}_training_log.csv")),
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

    plot_training_history(history, os.path.join(args.out_dir, f"{args.model}_training_curves.png"))
    print(f"\nDone. Best model saved to: {checkpoint_path}")
    print("Next step: python src/evaluate.py --model_path", checkpoint_path)


if __name__ == "__main__":
    main()
