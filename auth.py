import random
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import session, redirect, url_for
from functools import wraps
from datetime import datetime, timedelta

# --------- In-memory OTP store ---------
# Format: { email: { 'otp': '123456', 'expires': datetime, 'attempts': 0 } }
_otp_store = {}

OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 5
MAX_OTP_ATTEMPTS = 5


def login_required(func):
    """Decorator that redirects unauthenticated users to the login page."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_email' in session:
            return func(*args, **kwargs)
        return redirect(url_for('login'))
    return wrapper


def generate_otp():
    """Generate a random 6-digit OTP."""
    return ''.join([str(random.randint(0, 9)) for _ in range(OTP_LENGTH)])


def store_otp(email, otp):
    """Store OTP with expiry timestamp."""
    _otp_store[email.lower().strip()] = {
        'otp': otp,
        'expires': datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES),
        'attempts': 0,
    }


def verify_otp(email, otp):
    """
    Verify the OTP for a given email.
    Returns: (success: bool, error_message: str or None)
    """
    email = email.lower().strip()

    if email not in _otp_store:
        return False, "No OTP was sent to this email. Please request a new one."

    stored = _otp_store[email]

    # Check expiry
    if datetime.now() > stored['expires']:
        del _otp_store[email]
        return False, "OTP has expired. Please request a new one."

    # Check max attempts
    stored['attempts'] += 1
    if stored['attempts'] > MAX_OTP_ATTEMPTS:
        del _otp_store[email]
        return False, "Too many failed attempts. Please request a new OTP."

    # Check match
    if stored['otp'] == otp.strip():
        del _otp_store[email]  # One-time use
        return True, None

    remaining = MAX_OTP_ATTEMPTS - stored['attempts']
    return False, f"Incorrect OTP. {remaining} attempt(s) remaining."


def send_otp_email(recipient_email, otp):
    """
    Send OTP via email using SMTP.
    Falls back to console printing if SMTP is not configured (dev mode).
    Returns True on success, False on failure.
    """
    smtp_email = os.environ.get('SMTP_EMAIL')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', '587'))

    # Dev mode fallback — print OTP to console
    if not smtp_email or not smtp_password:
        import sys
        print("=" * 50, flush=True)
        print(f"  [DEV MODE] OTP for {recipient_email}: {otp}", flush=True)
        print("=" * 50, flush=True)
        return True

    # Build the email
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'DermaScan AI — Your Login Code'
    msg['From'] = smtp_email
    msg['To'] = recipient_email

    html_body = f"""
    <html>
    <body style="margin:0; padding:0; font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9;">
      <table width="100%" cellpadding="0" cellspacing="0" style="padding: 40px 20px;">
        <tr><td align="center">
          <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff; border-radius:16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); overflow:hidden;">
            <!-- Header -->
            <tr>
              <td style="background: linear-gradient(135deg, #4361ee, #3f37c9); padding: 30px 40px; text-align:center;">
                <h1 style="margin:0; color:#ffffff; font-size:24px;">🔬 DermaScan AI</h1>
                <p style="margin:8px 0 0; color:rgba(255,255,255,0.85); font-size:14px;">Secure Login Verification</p>
              </td>
            </tr>
            <!-- Body -->
            <tr>
              <td style="padding: 35px 40px;">
                <p style="margin:0 0 20px; color:#333; font-size:15px; line-height:1.6;">
                  Use the following code to log in to your DermaScan AI account. This code expires in <strong>{OTP_EXPIRY_MINUTES} minutes</strong>.
                </p>
                <div style="text-align:center; margin: 25px 0;">
                  <div style="display:inline-block; background:#f0f4ff; border: 2px dashed #4361ee; border-radius:12px; padding: 18px 40px;">
                    <span style="font-size:36px; font-weight:700; letter-spacing:8px; color:#4361ee; font-family: 'Courier New', monospace;">{otp}</span>
                  </div>
                </div>
                <p style="margin:20px 0 0; color:#888; font-size:13px; line-height:1.5;">
                  If you didn't request this code, you can safely ignore this email.<br>
                  Do not share this code with anyone.
                </p>
              </td>
            </tr>
            <!-- Footer -->
            <tr>
              <td style="background:#f8fafc; padding: 20px 40px; text-align:center; border-top: 1px solid #eee;">
                <p style="margin:0; color:#aaa; font-size:12px;">&copy; DermaScan AI &mdash; AI-Powered Skin Health Analysis</p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, recipient_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send OTP to {recipient_email}: {e}")
        return False