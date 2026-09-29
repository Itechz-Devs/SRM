"""
app.py
------
Flask web app for the SRM project.

Pages:
  /               - Landing choice: register, log in, or continue as guest
                    (shown to anyone not yet signed in)
  /register       - Create an account with email + password -> sends OTP
  /verify-otp     - Enter the emailed code to activate the account
  /resend-otp     - Resends a fresh code
  /login          - Log in with email + password
  /guest          - Continue without an account (downloads disabled)
  /logout         - Clears the session
  /               - Dashboard (once signed in / guest): evaluation history
                    + sample results
  /enhance        - Upload an image, run it through the trained CNN, see
                    (and optionally download) the enhanced result
  /download/<..>  - Download an enhanced result (registered accounts only)

Run with:
    python src/app.py
Then open http://localhost:5000
"""

import os
import re
import csv
import uuid
import shutil
import functools
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from tensorflow.keras.models import load_model
import model as model_module  # registers custom layers (e.g. PixelShuffle) before loading a saved model

import auth_db
from email_utils import send_otp_email

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-this-before-sharing-publicly")

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
UPLOADS_DIR = os.path.join(BASE_DIR, "..", "static", "uploads")
CONTRIB_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "EuroSAT_RGB", "UserContributed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CONTRIB_DIR, exist_ok=True)

# Sentinel-2's true-color bands are natively 10m/pixel. This is the
# reference point every "resulting resolution" figure below is computed
# from (10 / scale). This assumption only holds for real Sentinel-2
# imagery - if someone uploads an unrelated photo, the resolution figures
# are still shown (for consistency) but are not physically meaningful for
# that image, and the UI says so.
NATIVE_RESOLUTION_M = 10.0

MAX_UPLOAD_DIMENSION = 512  # cap the longest side for speed; aspect ratio is always preserved, never squished into a square

_model_cache = {}  # keyed by model key, e.g. "espcn_x4" or "srcnn" -> (model, error)


IMAGERY_TYPE_CHANNELS = {"rgb": 3, "ir": 4, "sar": 2, "thermal": 1}
IMAGERY_TYPE_LABELS = {
    "rgb": "Standard (RGB)",
    "ir": "Infrared (RGB+NIR, real Sentinel-2 data)",
    "sar": "Radar / SAR (real Sentinel-1 data)",
    "thermal": "Thermal (real Landsat data)",
}


def discover_available_models():
    """
    Scans models/ for every trained checkpoint and returns a dict describing
    each one, so the enhance page can offer only options that actually have
    a real trained model behind them - never a fake/placeholder choice.

    Returns: {key: {"path", "kind", "scale", "label", "imagery_type", "channels"}}
    """
    available = {}
    if not os.path.isdir(MODELS_DIR):
        return available

    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith("_best.keras"):
            continue
        key = fname[: -len("_best.keras")]
        path = os.path.join(MODELS_DIR, fname)

        match = re.match(r"^espcn_x(\d+)(?:_(ir|sar|thermal))?(_perceptual)?$", key)
        if match:
            scale = int(match.group(1))
            imagery_type = match.group(2) or "rgb"
            is_perceptual = bool(match.group(3))
            result_res = NATIVE_RESOLUTION_M / scale
            label = f"{IMAGERY_TYPE_LABELS[imagery_type]} — {scale}x upsampling (~{result_res:.1f}m equivalent, from {NATIVE_RESOLUTION_M:.0f}m input)"
            if is_perceptual:
                label += " - perceptual loss"
            available[key] = {
                "path": path,
                "kind": "upsample",
                "scale": scale,
                "imagery_type": imagery_type,
                "channels": IMAGERY_TYPE_CHANNELS[imagery_type],
                "label": label,
            }
        else:
            base_key = key[: -len("_perceptual")] if key.endswith("_perceptual") else key
            if base_key in ("srcnn", "deep_resnet"):
                label = f"{base_key} (same-size refinement, no resolution change)"
                if key.endswith("_perceptual"):
                    label += " - perceptual loss"
                available[key] = {
                    "path": path,
                    "kind": "refine",
                    "scale": None,
                    "imagery_type": "rgb",
                    "channels": 3,
                    "label": label,
                }

    return available


def available_imagery_types(available_models):
    """Which imagery types have at least one real trained model - drives
    the Imagery Type dropdown so it never shows a fake/unavailable option."""
    types_present = {info["imagery_type"] for info in available_models.values()}
    return {t: IMAGERY_TYPE_LABELS[t] for t in ["rgb", "ir", "sar", "thermal"] if t in types_present}


def get_model(key):
    if key not in _model_cache:
        available = discover_available_models()
        info = available.get(key)
        if info is None:
            _model_cache[key] = (None, f"No trained model found for '{key}'.")
        else:
            try:
                loaded = load_model(info["path"], compile=False)
                _model_cache[key] = (loaded, None)
                print(f"Loaded model '{key}' from {info['path']}")
            except Exception as e:
                _model_cache[key] = (None, str(e))
                print(f"Could not load model '{key}' at {info['path']}: {e}")
    return _model_cache[key]


# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------

@app.context_processor
def inject_auth_state():
    return {
        "is_authenticated": bool(session.get("user_email")),
        "is_guest": bool(session.get("is_guest")),
        "user_email": session.get("user_email"),
    }


def login_required(view_func):
    """Allows either a logged-in user OR a guest session through; sends
    anyone else to the landing/choice page."""
    @functools.wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_email") and not session.get("is_guest"):
            return redirect(url_for("landing"))
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or "@" not in email:
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                auth_db.create_or_update_pending_user(email, password)
                otp_code = auth_db.generate_and_store_otp(email)
                send_otp_email(email, otp_code)
                session["pending_email"] = email
                return redirect(url_for("verify_otp_page"))
            except Exception as e:
                error = f"Could not create account right now: {e}"
    return render_template("register.html", error=error)


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp_page():
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("register"))

    error = None
    if request.method == "POST":
        submitted_code = request.form.get("otp", "")
        try:
            success, err = auth_db.verify_otp(email, submitted_code)
        except Exception as e:
            success, err = False, f"Could not verify right now: {e}"
        if success:
            session.pop("pending_email", None)
            session["user_email"] = email
            session["is_guest"] = False
            return redirect(url_for("enhance"))
        error = err
    return render_template("verify_otp.html", email=email, error=error)


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    email = request.form.get("email", "").strip().lower()
    if email:
        try:
            otp_code = auth_db.generate_and_store_otp(email)
            send_otp_email(email, otp_code)
            session["pending_email"] = email
        except Exception as e:
            print(f"Could not resend OTP: {e}")
    return redirect(url_for("verify_otp_page"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            success, err = auth_db.check_login(email, password)
        except Exception as e:
            success, err = False, f"Could not log in right now: {e}"
        if success:
            session["user_email"] = email
            session["is_guest"] = False
            return redirect(url_for("dashboard"))
        error = err
    return render_template("login.html", error=error)


@app.route("/guest")
def guest():
    session["is_guest"] = True
    session["user_email"] = None
    return redirect(url_for("enhance"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/welcome")
def landing():
    if session.get("user_email") or session.get("is_guest"):
        return redirect(url_for("dashboard"))
    return render_template("auth_choice.html")


# ---------------------------------------------------------------------
# Core feature: enhance an uploaded image
# ---------------------------------------------------------------------

def load_geotiff_multiband(file_bytes, num_channels):
    """
    Reads a GeoTIFF using rasterio and returns exactly num_channels bands,
    each independently percentile-stretched to a normal 0-255 range -
    works for a real Sentinel-2 GeoTIFF, a genuine multi-band IR file, a
    2-band SAR file, or a 1-band thermal file, not just 3-band RGB.

    Raises ValueError with a clear message if the file doesn't actually
    have enough bands for what was requested - this is a real, meaningful
    check: you cannot get 4-channel infrared data out of a 3-band RGB file,
    no matter how it's processed, because the NIR information simply isn't
    physically present in an RGB image.
    """
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.count < num_channels:
                raise ValueError(
                    f"This file has {src.count} band(s), but {num_channels} are needed "
                    f"for the selected imagery type. A standard RGB image or GeoTIFF "
                    f"cannot supply infrared/SAR/thermal data that isn't physically in it."
                )
            bands = src.read(list(range(1, num_channels + 1)))

    img = np.transpose(bands, (1, 2, 0)).astype(np.float32)

    def rescale(band, low=2, high=98):
        p_low, p_high = np.percentile(band, (low, high))
        if p_high - p_low < 1e-6:
            return np.zeros_like(band)
        return np.clip((band - p_low) / (p_high - p_low), 0, 1)

    img_rescaled = np.dstack([rescale(img[:, :, i]) for i in range(num_channels)])
    return (img_rescaled * 255).astype(np.uint8)


def to_displayable_rgb(img, imagery_type):
    """
    Converts any channel count into a normal 3-channel image for saving as
    a viewable PNG. Honest about what each conversion actually shows:
      - rgb (3ch): as-is
      - thermal (1ch): grayscale replicated across channels
      - sar (2ch): a false-color composite (R=VV, G=VH, B=average) - a
        common SAR visualization convention, clearly not "true color"
        since radar has no color at all
      - ir (4ch): shows just the RGB portion - the NIR band is used
        internally by the model but isn't shown as a 4th visible channel
    """
    channels = img.shape[-1]
    if channels == 3:
        return img
    if channels == 1:
        return np.repeat(img, 3, axis=-1)
    if channels == 2:
        vv, vh = img[..., 0].astype(np.uint16), img[..., 1].astype(np.uint16)
        avg = ((vv + vh) // 2).astype(np.uint8)
        return np.stack([vv.astype(np.uint8), vh.astype(np.uint8), avg], axis=-1)
    return img[..., :3]  # e.g. 4-channel IR - show the RGB portion


def resize_preserving_aspect(img_rgb, max_dimension):
    """
    Scales the image down so its longest side is at most max_dimension,
    WITHOUT changing its aspect ratio (no forced square, no distortion).
    If the image is already smaller than max_dimension, it's left as-is.
    """
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img_rgb
    scale = max_dimension / longest
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_uncertainty_map(model, input_batch, n_passes=15):
    """
    Monte Carlo Dropout: runs the model several times with dropout still
    ACTIVE (training=True) and measures how much the output varies pixel-
    by-pixel across those runs. High variance = the model is less certain
    about that region. This only produces meaningful (non-near-zero)
    results because model.py's architectures now include real Dropout
    layers - previously they had none, so this always returned ~0.
    """
    predictions = np.stack([model(input_batch, training=True).numpy()[0] for _ in range(n_passes)])
    mean_pred = predictions.mean(axis=0)
    variance_map = predictions.var(axis=0)
    return mean_pred, variance_map


def enhance_image(file_storage, model_key):
    """
    Runs an uploaded image through the selected trained model. Works with
    ANY of the discovered models (see discover_available_models()) - true
    upsampling (espcn_xN, any imagery type) or legacy same-size refinement
    (srcnn/deep_resnet, RGB only).

    The uploaded image's aspect ratio is always preserved - it is never
    forced into a square, since every model here is fully convolutional
    and genuinely works at any input size (verified during development).
    """
    model, error = get_model(model_key)
    if model is None:
        return None, error

    available = discover_available_models()
    info = available.get(model_key, {})
    expected_channels = info.get("channels", 3)
    imagery_type = info.get("imagery_type", "rgb")

    raw_bytes = file_storage.read()

    if expected_channels == 3:
        # Standard path: try a normal image decoder first (JPG/PNG/etc),
        # fall back to GeoTIFF if that fails - unchanged from before.
        file_bytes = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                img = load_geotiff_multiband(raw_bytes, 3)
            except Exception as e:
                return None, (
                    f"Could not read that file as a standard RGB image or GeoTIFF: {e}"
                )
    else:
        # IR/SAR/thermal genuinely need more (or different) bands than a
        # plain JPG/PNG can ever contain - only a real multi-band GeoTIFF
        # (from the matching download script) can supply this.
        try:
            img = load_geotiff_multiband(raw_bytes, expected_channels)
        except Exception as e:
            return None, (
                f"This imagery type ({IMAGERY_TYPE_LABELS.get(imagery_type, imagery_type)}) "
                f"needs a GeoTIFF with at least {expected_channels} band(s). {e}"
            )

    img = resize_preserving_aspect(img, MAX_UPLOAD_DIMENSION)

    if info.get("kind") == "refine":
        # Legacy same-size models expect a pre-blurred input matching how
        # they were trained (see data_pipeline.py) - apply the same
        # downsample-then-upsample degradation here, still at the image's
        # natural (non-square) size.
        h, w = img.shape[:2]
        degrade_scale = 4
        small = cv2.resize(img, (max(1, w // degrade_scale), max(1, h // degrade_scale)), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    img_norm = img.astype(np.float32) / 255.0
    input_batch = np.expand_dims(img_norm, axis=0)

    mean_pred, variance_map = compute_uncertainty_map(model, input_batch)
    output_raw = np.clip(mean_pred * 255.0, 0, 255).astype(np.uint8)
    input_raw = img

    input_display = to_displayable_rgb(input_raw, imagery_type)
    output_display = to_displayable_rgb(output_raw, imagery_type)

    run_id = uuid.uuid4().hex[:8]
    input_filename = f"input_{run_id}.png"
    output_filename = f"output_{run_id}.png"
    uncertainty_filename = f"uncertainty_{run_id}.png"

    cv2.imwrite(os.path.join(UPLOADS_DIR, input_filename), cv2.cvtColor(input_display, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(UPLOADS_DIR, output_filename), cv2.cvtColor(output_display, cv2.COLOR_RGB2BGR))

    # Render the uncertainty map as a heatmap image (matplotlib's "inferno"
    # colormap, same as evaluate.py uses) so it's viewable like any other result.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uncertainty_gray = variance_map.mean(axis=-1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(uncertainty_gray, cmap="inferno")
    ax.axis("off")
    fig.savefig(os.path.join(UPLOADS_DIR, uncertainty_filename), bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)

    scale = info.get("scale")
    resolution_info = {
        "input_res_m": NATIVE_RESOLUTION_M,
        "output_res_m": (NATIVE_RESOLUTION_M / scale) if scale else NATIVE_RESOLUTION_M,
        "scale": scale,
        "kind": info.get("kind"),
        "imagery_type": imagery_type,
        "input_dims": f"{input_raw.shape[1]}x{input_raw.shape[0]}",
        "output_dims": f"{output_raw.shape[1]}x{output_raw.shape[0]}",
    }

    contrib_frame = cv2.cvtColor(input_display, cv2.COLOR_RGB2BGR)

    result = {
        "input": input_filename,
        "output": output_filename,
        "uncertainty": uncertainty_filename,
        "resolution": resolution_info,
    }
    return (result, contrib_frame), None


def save_contribution(contrib_frame):
    """Saves the image into the same folder structure data_pipeline.py already
    reads from - so re-running preprocessing naturally picks it up next time.
    No IR/hyperspectral/SAR/thermal bands here since it's a plain RGB upload."""
    filename = f"contrib_{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(os.path.join(CONTRIB_DIR, filename), contrib_frame)
    return filename


@app.route("/enhance", methods=["GET", "POST"])
@login_required
def enhance():
    result = None
    error = None
    contributed = False
    available_models = discover_available_models()

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        model_key = request.form.get("model_key")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please choose an image file first."
        elif not model_key or model_key not in available_models:
            error = "Please select a valid target resolution/model."
        else:
            outcome, err = enhance_image(uploaded_file, model_key)
            if err:
                error = err
            else:
                result, contrib_frame = outcome
                result["original_name"] = uploaded_file.filename
                if request.form.get("contribute") == "yes" and contrib_frame is not None:
                    save_contribution(contrib_frame)
                    contributed = True

    model_ready = len(available_models) > 0
    trained_types = {info["imagery_type"] for info in available_models.values()}
    missing_imagery_types = [
        IMAGERY_TYPE_LABELS[t] for t in ["ir", "sar", "thermal"] if t not in trained_types
    ]
    return render_template(
        "enhance.html",
        result=result,
        error=error,
        model_ready=model_ready,
        contributed=contributed,
        available_models=available_models,
        missing_imagery_types=missing_imagery_types,
    )


@app.route("/download/<path:filename>")
@login_required
def download_result(filename):
    if session.get("is_guest"):
        return redirect(url_for("register"))
    return send_from_directory(UPLOADS_DIR, filename, as_attachment=True)


# ---------------------------------------------------------------------
# Dashboard (existing functionality, now behind login_required)
# ---------------------------------------------------------------------

def load_metrics_history():
    try:
        from db import fetch_all_results
        rows = fetch_all_results()
        return [{"run_time": r[0], "model_name": r[1], "psnr": r[2], "ssim": r[3], "rmse": r[4]} for r in rows]
    except Exception as e:
        print(f"(PostgreSQL not available, falling back to metrics.csv: {e})")
        csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
        if not os.path.exists(csv_path):
            return []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return [
                {
                    "run_time": "-",
                    "model_name": row.get("model_path", "-"),
                    "psnr": row.get("psnr_mean", "-"),
                    "ssim": row.get("ssim_mean", "-"),
                    "rmse": row.get("rmse_mean", "-"),
                }
                for row in reader
            ]


def list_sample_images():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith("sample_") and f.endswith(".png")])


@app.route("/")
@login_required
def dashboard():
    history = load_metrics_history()
    samples = list_sample_images()
    return render_template("dashboard.html", history=history, samples=samples)


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
