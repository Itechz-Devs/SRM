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
  /history        - Your own past enhance runs (registered: saved forever;
                    guest: saved only for this browser session)
  /download/<..>  - Download an enhanced result (registered accounts only)

Run with:
    python src/app.py
Then open http://localhost:5000
"""

import os
import csv
import uuid
import shutil
import functools
from datetime import datetime, timezone
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from tensorflow.keras.models import load_model

import auth_db
from email_utils import send_otp_email

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-this-before-sharing-publicly")

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
UPLOADS_DIR = os.path.join(BASE_DIR, "..", "static", "uploads")
CONTRIB_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "EuroSAT_RGB", "UserContributed")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CONTRIB_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(RESULTS_DIR, "history.csv")
HISTORY_FIELDS = ["timestamp", "owner", "original_name", "original_filename", "input_filename", "output_filename"]

MODEL_PATH = os.environ.get("SRM_MODEL_PATH", os.path.join(BASE_DIR, "..", "models", "srcnn_best.keras"))
MODEL_INPUT_SIZE = 64
# Must match the --scale value used when you ran data_pipeline.py (default 4).
# The model was trained to sharpen images degraded this exact way - feeding
# it anything else (e.g. a plain resize with no degradation) means it has
# little actual blur to correct, so the output looks nearly unchanged.
DEGRADE_SCALE = 4
DISPLAY_SIZE = 320

_model_cache = {"model": None, "error": None}


def get_model():
    if _model_cache["model"] is None and _model_cache["error"] is None:
        try:
            _model_cache["model"] = load_model(MODEL_PATH, compile=False)
            print(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            _model_cache["error"] = str(e)
            print(f"Could not load model at {MODEL_PATH}: {e}")
    return _model_cache["model"], _model_cache["error"]


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


def current_owner():
    """Identity key used for history rows: the account email for logged-in
    users, or a per-browser-session id for guests (so one guest never sees
    another guest's history, but doesn't need an account either)."""
    if session.get("user_email"):
        return session["user_email"]
    if not session.get("guest_id"):
        session["guest_id"] = uuid.uuid4().hex
    return f"guest:{session['guest_id']}"


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
    session["guest_id"] = session.get("guest_id") or uuid.uuid4().hex
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
# History (persisted to a CSV so it survives restarts without needing
# a database migration; if you already have Postgres wired up via db.py
# you can swap record_history/load_history_for to write there instead)
# ---------------------------------------------------------------------

def record_history(owner, original_name, original_filename, input_filename, output_filename):
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "owner": owner,
            "original_name": original_name,
            "original_filename": original_filename,
            "input_filename": input_filename,
            "output_filename": output_filename,
        })


def load_history_for(owner):
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("owner") == owner]
    rows.reverse()  # most recent first
    return rows


@app.route("/history")
@login_required
def history():
    rows = load_history_for(current_owner())
    return render_template("history.html", rows=rows)


# ---------------------------------------------------------------------
# Core feature: enhance an uploaded image
# ---------------------------------------------------------------------

def _center_crop_square(img):
    """Crop to a square around the center before resizing, instead of
    stretching a non-square image directly into a square. Skipping this
    was the cause of uploaded images looking geometrically distorted -
    a 1200x800 photo, for example, was being squashed 1:1 into 64x64."""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top:top + side, left:left + side]


def enhance_image(file_storage):
    """
    Runs an uploaded image through the trained CNN. See model.py/train.py
    for how the model was trained (64x64 satellite patches).

    Returns three images for display, all the same pixel size so they line
    up in the UI:
      - original_display: the user's real upload, aspect-preserved, resized
        only (no degradation) - "what you actually gave us"
      - input_display: that same crop, degraded the same way the training
        data was degraded - "what the model actually saw"
      - output_display: the model's reconstruction of the degraded input
    """
    model, error = get_model()
    if model is None:
        return None, None, None, None, error

    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None, None, None, "Could not read that file as an image. Try a .jpg or .png."

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    square = _center_crop_square(img_rgb)
    original_resized = cv2.resize(square, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # Match training-time preprocessing exactly: shrink then stretch back up.
    # This is not optional - skipping it means the model sees a much
    # sharper input than it was ever trained on, and its learned
    # sharpening has little actual blur left to correct.
    small = cv2.resize(
        original_resized,
        (MODEL_INPUT_SIZE // DEGRADE_SCALE, MODEL_INPUT_SIZE // DEGRADE_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    degraded = cv2.resize(small, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img_norm = degraded.astype(np.float32) / 255.0

    prediction = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0]
    output_uint8 = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

    # LANCZOS4 instead of the old NEAREST - NEAREST blows pixels up into
    # hard blocky squares, which reads as "just zoomed in" even when the
    # model output is genuinely different from the input.
    original_display = cv2.resize(original_resized, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
    input_display = cv2.resize(degraded, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
    output_display = cv2.resize(output_uint8, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)

    run_id = uuid.uuid4().hex[:8]
    original_filename = f"original_{run_id}.png"
    input_filename = f"input_{run_id}.png"
    output_filename = f"output_{run_id}.png"
    cv2.imwrite(os.path.join(UPLOADS_DIR, original_filename), cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(UPLOADS_DIR, input_filename), cv2.cvtColor(input_display, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(UPLOADS_DIR, output_filename), cv2.cvtColor(output_display, cv2.COLOR_RGB2BGR))

    # Keep the degraded 64x64 frame around in case the user opts to
    # contribute it to the training dataset below.
    contrib_frame = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)

    return original_filename, input_filename, output_filename, contrib_frame, None


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

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or uploaded_file.filename == "":
            error = "Please choose an image file first."
        else:
            original_filename, input_filename, output_filename, contrib_frame, err = enhance_image(uploaded_file)
            if err:
                error = err
            else:
                result = {
                    "original": original_filename,
                    "input": input_filename,
                    "output": output_filename,
                    "original_name": uploaded_file.filename,
                }
                record_history(current_owner(), uploaded_file.filename, original_filename, input_filename, output_filename)
                if request.form.get("contribute") == "yes" and contrib_frame is not None:
                    save_contribution(contrib_frame)
                    contributed = True

    model_ready = os.path.exists(MODEL_PATH)
    return render_template("enhance.html", result=result, error=error, model_ready=model_ready, contributed=contributed)


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
    history_rows = load_metrics_history()
    samples = list_sample_images()
    return render_template("dashboard.html", history=history_rows, samples=samples)


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
