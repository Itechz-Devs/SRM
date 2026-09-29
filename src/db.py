"""
db.py
-----
Minimal PostgreSQL logging for evaluation runs, so the dashboard (app.py) has
a history to display instead of just the latest run.

SETUP (one-time):
  1. Install PostgreSQL locally (see README.md for OS-specific instructions).
  2. Create a database and user, e.g.:
       psql -U postgres
       CREATE DATABASE srm_db;
       CREATE USER srm_user WITH PASSWORD 'your_password_here';
       GRANT ALL PRIVILEGES ON DATABASE srm_db TO srm_user;
  3. Set these environment variables (or edit the defaults below directly -
     fine for a hackathon demo, just don't commit real passwords to GitHub):
       set DB_HOST=localhost        (Windows: use `set`, Mac/Linux: `export`)
       set DB_NAME=srm_db
       set DB_USER=srm_user
       set DB_PASS=your_password_here

The table is created automatically the first time you call insert_result().
"""

import os
import datetime
import psycopg2


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        dbname=os.environ.get("DB_NAME", "srm_db"),
        user=os.environ.get("DB_USER", "srm_user"),
        password=os.environ.get("DB_PASS", "change_me"),
        port=os.environ.get("DB_PORT", "5432"),
    )


def ensure_table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS srm_results (
                id SERIAL PRIMARY KEY,
                run_time TIMESTAMP NOT NULL,
                model_name TEXT NOT NULL,
                psnr_mean REAL,
                ssim_mean REAL,
                rmse_mean REAL,
                notes TEXT
            );
        """)
    conn.commit()


def insert_result(model_name, psnr_mean=None, ssim_mean=None, rmse_mean=None, notes=None, **kwargs):
    """Logs one evaluation run. Extra kwargs (e.g. spectral scores) are ignored
    here but printed, so you can extend the table later if you want them stored."""
    conn = get_connection()
    try:
        ensure_table_exists(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO srm_results (run_time, model_name, psnr_mean, ssim_mean, rmse_mean, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (datetime.datetime.now(), model_name, psnr_mean, ssim_mean, rmse_mean, notes),
            )
        conn.commit()
        if kwargs:
            print(f"(Note: extra metrics {list(kwargs.keys())} were not stored - "
                  f"add columns to srm_results in db.py if you want to keep them.)")
    finally:
        conn.close()


def fetch_all_results():
    """Used by the dashboard (app.py) to list past runs."""
    conn = get_connection()
    try:
        ensure_table_exists(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT run_time, model_name, psnr_mean, ssim_mean, rmse_mean FROM srm_results ORDER BY run_time DESC;")
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick manual test: python src/db.py
    insert_result(model_name="test_model", psnr_mean=20.1, ssim_mean=0.8, rmse_mean=0.05, notes="manual test row")
    print(fetch_all_results())
