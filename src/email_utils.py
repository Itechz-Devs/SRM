"""
email_utils.py
--------------
Sends the OTP verification code by email using standard SMTP (works with
Gmail, Outlook, etc. once you configure it - see README for how to get a
Gmail "App Password").

DEV FALLBACK: if you haven't set up SMTP_HOST/SMTP_USER/SMTP_PASS yet, this
prints the OTP code to your terminal instead of failing - so you can test
and demo the whole registration flow before wiring up real email.
"""

import os
import smtplib
from email.mime.text import MIMEText


def send_otp_email(to_email, otp_code):
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = os.environ.get("SMTP_PORT", "587")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not all([smtp_host, smtp_user, smtp_pass]):
        print("=" * 50)
        print(f"[DEV MODE - no SMTP configured] OTP for {to_email}: {otp_code}")
        print("Set SMTP_HOST / SMTP_USER / SMTP_PASS to send real emails.")
        print("=" * 50)
        return True

    try:
        msg = MIMEText(
            f"Your SRM verification code is: {otp_code}\n\n"
            f"This code expires in 10 minutes. If you didn't request this, ignore this email."
        )
        msg["Subject"] = "Your SRM verification code"
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"Failed to send OTP email ({e}) - falling back to console:")
        print(f"[FALLBACK] OTP for {to_email}: {otp_code}")
        return False
