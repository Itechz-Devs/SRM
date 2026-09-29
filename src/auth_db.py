"""
auth_db.py
----------
User accounts and email-OTP verification, stored in PostgreSQL (same
database/connection as db.py - see that file for connection setup and
environment variables DB_HOST/DB_NAME/DB_USER/DB_PASS).

Two tables, created automatically the first time this is used:
  users - email, hashed password, verified status
  otps  - one-time codes sent by email, with an expiry time

Passwords are never stored in plain text - werkzeug's generate_password_hash
handles that (same library Flask itself depends on, so no extra install).
"""

import datetime
import random
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_connection

OTP_VALID_MINUTES = 10


def ensure_auth_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS otps (
                id SERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
    conn.commit()


def create_or_update_pending_user(email, password):
    """
    Registers a new account, or - if someone abandoned registration before
    verifying - resets it so they can try again. Always leaves the account
    as NOT verified until the OTP step succeeds.
    """
    password_hash = generate_password_hash(password)
    conn = get_connection()
    try:
        ensure_auth_tables(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, password_hash, is_verified)
                VALUES (%s, %s, FALSE)
                ON CONFLICT (email) DO UPDATE
                SET password_hash = EXCLUDED.password_hash, is_verified = FALSE;
            """, (email, password_hash))
        conn.commit()
    finally:
        conn.close()


def generate_and_store_otp(email):
    otp_code = f"{random.randint(0, 999999):06d}"
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=OTP_VALID_MINUTES)
    conn = get_connection()
    try:
        ensure_auth_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO otps (email, otp_code, expires_at) VALUES (%s, %s, %s);",
                (email, otp_code, expires_at),
            )
        conn.commit()
    finally:
        conn.close()
    return otp_code


def verify_otp(email, submitted_code):
    """Checks the most recent OTP for this email and confirms it matches and hasn't expired."""
    conn = get_connection()
    try:
        ensure_auth_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT otp_code, expires_at FROM otps WHERE email = %s ORDER BY created_at DESC LIMIT 1;",
                (email,),
            )
            row = cur.fetchone()
        if row is None:
            return False, "No verification code found - request a new one."
        otp_code, expires_at = row
        if datetime.datetime.now() > expires_at:
            return False, "That code has expired - request a new one."
        if submitted_code.strip() != otp_code:
            return False, "That code doesn't match."

        with conn.cursor() as cur:
            cur.execute("UPDATE users SET is_verified = TRUE WHERE email = %s;", (email,))
        conn.commit()
        return True, None
    finally:
        conn.close()


def check_login(email, password):
    """Returns (success, error_message)."""
    conn = get_connection()
    try:
        ensure_auth_tables(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash, is_verified FROM users WHERE email = %s;", (email,))
            row = cur.fetchone()
        if row is None:
            return False, "No account with that email - register first."
        password_hash, is_verified = row
        if not check_password_hash(password_hash, password):
            return False, "Incorrect password."
        if not is_verified:
            return False, "Please verify your email first - check for the code we sent."
        return True, None
    finally:
        conn.close()
