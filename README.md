# SIH 2026 — Deep Learning Based Super Resolution Mapping (SRM)

A from-scratch CNN pipeline that takes medium-resolution Sentinel-2-style satellite
imagery (10m) and learns to reconstruct sharper, enhanced-resolution output, following
your project flowchart end to end:

```
Sentinel-2 Imagery → Preprocessing → LR-HR Dataset → CNN (SRCNN/Deep-ResNet)
→ SR Image Output → Metrics (PSNR/SSIM/RMSE) + Spectral Check + Geospatial Check
→ Uncertainty Mapping → Scientifically Reliable SR Product → Web Dashboard
```

## Important, read this first: about the training data

The `archive.zip` you provided contains **EuroSAT_RGB** — 27,000 images, 64×64 pixels,
RGB only, across 10 land-cover classes (crops, forest, highway, residential, river, etc).

**This dataset does NOT include infrared, hyperspectral, radar/SAR, or thermal bands.**
Those weren't in the file you gave me, so this build does not fabricate support for
them. What it *does* do: every script here is written to be **band-count agnostic** —
`data_pipeline.py` reads however many channels an image has, and the model architecture
in `model.py` takes its channel count from the data automatically. If you later obtain
multi-band Sentinel-2 GeoTIFFs (13 bands, including NIR) — for example via the
Copernicus Browser pipeline — you can swap in a rasterio-based loader and the rest of
the pipeline (LR/HR generation, training, evaluation) does not need to change.

Be upfront about this scope in your presentation — it's more credible than claiming
multispectral capability the current build doesn't have.

## How the training data works (since we don't have real paired LR/HR imagery)

There's no dataset of "the exact same location at both 10m and <4m resolution" readily
available. The standard approach (used by SRCNN, EDSR, and similar research) is:
treat your available imagery as **HR ground truth**, then **synthetically downsample**
it to create the **LR input**. The model learns to reverse that degradation. This is
exactly what `data_pipeline.py` does — it's a legitimate, widely-used technique, not a
shortcut, but say so explicitly when presenting: your validation is against a *proxy*
ground truth, not independently captured high-resolution imagery.

---

## Project structure

```
srm_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/EuroSAT_RGB/        ← extract archive.zip here
│   └── processed/              ← generated .npy LR/HR pairs (data_pipeline.py output)
├── models/                     ← trained .keras models + training curves (train.py output)
├── results/                    ← sample images + metrics.csv (evaluate.py output)
├── templates/
│   └── dashboard.html
└── src/
    ├── data_pipeline.py        ← OpenCV image loading + LR/HR pair generation
    ├── model.py                ← CNN architectures (SRCNN, Deep-ResNet)
    ├── train.py                ← training script (YOU run this)
    ├── evaluate.py             ← metrics, spectral check, uncertainty mapping
    ├── db.py                   ← PostgreSQL logging
    └── app.py                  ← Flask web dashboard
```

Every script in `src/` has been run and verified against your actual `archive.zip`
data during development — the pipeline is confirmed working end to end, not just
written from theory.

---

## Step 1 — Environment setup (Windows)

**Install Python 3.10+** if you don't have it: https://www.python.org/downloads/
(check "Add Python to PATH" during install).

Open Command Prompt or PowerShell in your project folder and create a virtual
environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

This installs TensorFlow, OpenCV, scikit-image, Flask, psycopg2, and rasterio — no
`basicsr`/`facexlib`/`realesrgan` needed anymore, which is what was causing your
earlier Jupyter install error. That problem is gone with this approach.

---

## Step 2 — Get the data in place

Extract `archive.zip` so the folder structure looks like:
```
data/raw/EuroSAT_RGB/AnnualCrop/...
data/raw/EuroSAT_RGB/Forest/...
... (10 class folders total)
```

---

## Step 3 — Preprocess: images → matrices → LR/HR pairs

```
python src/data_pipeline.py --data_dir data/raw/EuroSAT_RGB --out_dir data/processed
```

**Recommended for your first run:** use a smaller subset so you see the whole pipeline
work in under a minute, before committing to the full 27,000-image run:
```
python src/data_pipeline.py --data_dir data/raw/EuroSAT_RGB --out_dir data/processed --limit_per_class 300
```

What this does:
1. Loads every image with OpenCV (`cv2.imread`) into a numpy matrix.
2. Converts BGR→RGB and normalizes pixel values to [0, 1].
3. Downsamples each image then upsamples it back (bicubic) to create the LR input —
   this is your "medium-resolution" simulated input.
4. Splits into train/val/test sets and saves them as `.npy` files in `data/processed/`.

---

## Step 4 — Train the CNN

```
python src/train.py --model srcnn --epochs 20
```

Or the deeper residual model (matches the "EDSR" direction in your flowchart, more
capacity, slower):
```
python src/train.py --model deep_resnet --epochs 30 --batch_size 16
```

This is the step **you run and watch** — training time depends on your machine and how
much data you used in Step 3. Watch the `loss` value drop each epoch; it should
generally decrease. The best model (by validation loss) is saved automatically to
`models/<model_name>_best.keras`, along with a training curve plot and a CSV log.

If training seems too slow, reduce `--limit_per_class` in Step 3 or lower `--epochs`.

---

## Step 5 — Evaluate: metrics, spectral check, uncertainty map

```
python src/evaluate.py --model_path models/srcnn_best.keras
```

This produces, matching your flowchart exactly:
- **PSNR / SSIM / RMSE** — printed and saved to `results/metrics.csv`
- **Spectral consistency check** — per-channel (R/G/B) histogram correlation between
  predicted and ground-truth images
- **Uncertainty mapping** — via Monte Carlo Dropout. *Note: the default models don't
  include Dropout layers (kept simple for a fast first build). Add
  `layers.Dropout(0.1)` inside `model.py` and retrain if you want genuine stochastic
  uncertainty maps — until then this returns a near-zero variance map, which is an
  honest result to report, not a bug.*
- Sample before/after/uncertainty visual grids saved to `results/sample_0.png`, etc.

**On geospatial consistency:** EuroSAT JPGs have no coordinate reference system, so
there's nothing to check geospatially on this training/test data. `evaluate.py`
includes a working `check_geospatial_consistency()` function using `rasterio` for when
you run this model on real Sentinel-2 GeoTIFFs (e.g. downloaded via the Copernicus
Browser) — mention this honestly as a documented limitation of the current demo.

---

## Step 6 — Set up PostgreSQL (optional but requested)

**Install PostgreSQL on Windows:** download from https://www.postgresql.org/download/windows/
and run the installer (remember the password you set for the `postgres` superuser).mayur@0065

Open the SQL Shell (psql, installed alongside PostgreSQL) and run:
```sql
CREATE DATABASE srm_db;
CREATE USER srm_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE srm_db TO srm_user;
```

Set environment variables (Command Prompt):
```
set DB_HOST=localhost
set DB_NAME=srm_db
set DB_USER=srm_user
set DB_PASS=your_password_here
```

Then re-run evaluation with database logging:
```
python src/evaluate.py --model_path models/srcnn_best.keras --log_to_db
```

**If you skip this step, everything still works** — `app.py` automatically falls back
to reading `results/metrics.csv` if PostgreSQL isn't configured, so the dashboard
never breaks.

---

## Step 7 — Launch the dashboard

```
python src/app.py
```

Open http://localhost:5000 in your browser. You'll see your evaluation run history
(from PostgreSQL or the CSV fallback) and a gallery of before/after/uncertainty images.
This is what you present live.

---

## Step 8 — Git and GitHub

Initialize and push your project (do this once you have a version worth saving):
```
git init
git add .
git commit -m "Initial SRM pipeline: CNN training, evaluation, dashboard"
```

Create a new empty repository on https://github.com/new (don't initialize it with a
README there — you already have one), then:
```
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

The `.gitignore` already excludes the large data folder, trained model files, and
generated results — so your repo stays small and doesn't accidentally include your
database password.

---

## What to say in your presentation

- **Data:** EuroSAT_RGB (Sentinel-2 derived, RGB, 10 classes). No IR/SAR/hyperspectral
  bands in this build — architecture is band-agnostic and extensible.
- **Training pairs:** synthetically generated via downsample-then-upsample of real
  imagery — a standard, defensible SR training technique, not a shortcut.
- **Models:** SRCNN (fast baseline) and a deeper residual CNN (EDSR-inspired) — pick
  whichever trained better for your final demo.
- **Validation:** PSNR/SSIM/RMSE plus spectral consistency checking — validated against
  a proxy ground truth; true validation would need independently captured high-res
  reference imagery.
- **Uncertainty:** Monte Carlo Dropout variance mapping — mention whether you added
  Dropout layers for genuine uncertainty estimates.
- **Full stack:** OpenCV for data ingestion, TensorFlow/Keras for the model, PostgreSQL
  for run tracking, Flask for the presentation dashboard, Git/GitHub for version
  control — the complete pipeline your flowchart describes.
