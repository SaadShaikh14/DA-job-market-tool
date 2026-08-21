"""
auth.py
Handles user accounts (sign up / log in) and per-user activity logging,
backed by a PostgreSQL database.

Expects a connection string either in Streamlit secrets:

    # .streamlit/secrets.toml
    [postgres]
    url = "postgresql://user:password@host:port/dbname?sslmode=require"

or, for local dev without secrets.toml, in a DATABASE_URL env var
(e.g. in your .env file, alongside GROQ_API_KEY etc.).

Passwords are never stored in plain text — only a bcrypt hash.
"""

import os
import re
import ssl
import hashlib
import smtplib
import secrets as pysecrets
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone

import bcrypt
import psycopg2
import psycopg2.extras
import psycopg2.pool
import streamlit as st

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Exactly 10 digits - plain Indian mobile numbers only, no country code,
# no "+" prefix. Spaces/dashes are stripped before this is checked.
PHONE_RE = re.compile(r"^[0-9]{10}$")

OTP_LENGTH = 6
OTP_VALID_MINUTES = 10
OTP_MAX_ATTEMPTS = 5


def _get_connection_string():
    try:
        return st.secrets["postgres"]["url"]
    except Exception:
        pass
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database connection configured. Add a [postgres] url to "
            ".streamlit/secrets.toml (or DATABASE_URL to your .env)."
        )
    return url


@st.cache_resource
def _get_pool():
    """A small pool of connections, created once per server process and
    reused across reruns/users. Avoids paying a fresh TCP+TLS+auth
    handshake to Neon on every single login/signup/log_activity call -
    that per-call reconnect cost was the main source of repeated slowness
    beyond Neon's one-time cold-start wake-up."""
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=_get_connection_string(),
        connect_timeout=10,
    )


def _is_connection_alive(conn):
    """Cheap health probe. Neon (free tier) suspends its compute after a
    few minutes idle, which silently kills any connection sitting in the
    pool - the connection object looks fine until the next real query
    throws OperationalError. Catching that here means callers never see
    it; they just transparently get a fresh connection instead."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


class _PooledConnection:
    """Context-manager wrapper so callers can keep using
    `with get_connection() as conn:` unchanged, while the underlying
    connection is borrowed from (and returned to) the pool instead of
    being opened and closed from scratch each time."""

    def __enter__(self):
        self._pool = _get_pool()
        conn = self._pool.getconn()
        if not _is_connection_alive(conn):
            # Stale/dead connection (e.g. Neon's compute went to sleep
            # and woke back up) - discard it and grab a fresh one instead
            # of handing back something broken.
            self._pool.putconn(conn, close=True)
            conn = self._pool.getconn()
        self._conn = conn
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        # A connection that errored mid-transaction shouldn't be handed
        # back dirty - roll it back before returning it to the pool.
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        # A connection-level failure (vs. an application error like a
        # UniqueViolation) means the connection itself is broken - close
        # it instead of returning it to the pool for the next caller to
        # trip over the same dead connection.
        broken = exc_type is not None and issubclass(
            exc_type, (psycopg2.OperationalError, psycopg2.InterfaceError)
        )
        self._pool.putconn(self._conn, close=broken)


def get_connection():
    return _PooledConnection()


@st.cache_resource
def init_db():
    """Create the users/activity_log tables if they don't exist yet.
    Cached so this only runs once per server process, not on every rerun."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(30) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(20) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            # Safe to re-run against a users table created before this column
            # existed. NOT NULL/UNIQUE aren't added retroactively here since
            # that would fail against existing rows with no phone value -
            # only new installs get the constraint via CREATE TABLE above.
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(20);")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    activity_type VARCHAR(30) NOT NULL,
                    details JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_log(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_log(activity_type);")
        conn.commit()
    return True


def create_user(username, email, phone, password):
    """Sign up a new user. Returns (True, user_dict) on success,
    or (False, error_message) on failure."""
    username = username.strip()
    email = email.strip().lower()
    phone = phone.strip().replace(" ", "").replace("-", "")

    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, or underscore only."
    if not EMAIL_RE.match(email):
        return False, "Enter a valid email address."
    if not PHONE_RE.match(phone):
        return False, "Enter a valid 10-digit phone number."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(
                    "INSERT INTO users (username, email, phone, password_hash) "
                    "VALUES (%s, %s, %s, %s) RETURNING id, username, email, phone, created_at;",
                    (username, email, phone, password_hash),
                )
                user = cur.fetchone()
                conn.commit()
                return True, dict(user)
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return False, "That username, email, or phone number is already registered."


def check_availability(username, email, phone):
    """Validates format and checks username/email/phone aren't already
    taken - WITHOUT creating a user. Called before an OTP is sent so we
    don't burn an email on a signup that would fail anyway.
    Returns (True, None) or (False, error_message)."""
    username = username.strip()
    email = email.strip().lower()
    phone = phone.strip().replace(" ", "").replace("-", "")

    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, or underscore only."
    if not EMAIL_RE.match(email):
        return False, "Enter a valid email address."
    if not PHONE_RE.match(phone):
        return False, "Enter a valid 10-digit phone number."

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE username = %s OR email = %s OR phone = %s LIMIT 1;",
                (username, email, phone),
            )
            if cur.fetchone():
                return False, "That username, email, or phone number is already registered."
    return True, None


def _get_smtp_config():
    """Reads SMTP creds from Streamlit secrets (deployed) or env vars
    (local .env). Raises with a clear setup message if neither is set."""
    try:
        cfg = st.secrets["smtp"]
        return {
            "host": cfg.get("host", "smtp.gmail.com"),
            "port": int(cfg.get("port", 587)),
            "user": cfg["user"],
            "password": cfg["password"],
            "from_addr": cfg.get("from_addr", cfg["user"]),
        }
    except Exception:
        pass

    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "No SMTP configured. Add a [smtp] section (host/port/user/password) "
            "to .streamlit/secrets.toml, or set SMTP_USER/SMTP_PASSWORD "
            "(and optionally SMTP_HOST/SMTP_PORT/SMTP_FROM) in your .env."
        )
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from_addr": os.getenv("SMTP_FROM", user),
    }


def _generate_otp():
    """Cryptographically random numeric OTP (not random.random - that's
    not safe for anything security-adjacent)."""
    return "".join(str(pysecrets.randbelow(10)) for _ in range(OTP_LENGTH))


def _hash_otp(otp):
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def send_otp_email(to_email, otp):
    """Emails a one-time verification code. Raises on failure so the
    caller can surface a real error instead of silently pretending
    it sent."""
    msg = MIMEText(
        f"Your verification code is {otp}.\n\n"
        f"It expires in {OTP_VALID_MINUTES} minutes. "
        "If you didn't request this, you can ignore this email."
    )
    msg["Subject"] = "Your verification code - DA Job Market Tool"
    cfg = _get_smtp_config()
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
        server.starttls(context=context)
        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["from_addr"], [to_email], msg.as_string())


def make_otp_challenge(email):
    """Generates a fresh OTP, emails it, and returns a bundle to stash
    in st.session_state - only the hash + expiry + attempt count, never
    the raw code."""
    otp = _generate_otp()
    send_otp_email(email, otp)
    return {
        "otp_hash": _hash_otp(otp),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=OTP_VALID_MINUTES)).isoformat(),
        "attempts": 0,
    }


def verify_otp(challenge, entered_otp):
    """Checks an entered code against a challenge dict from
    make_otp_challenge. Mutates challenge['attempts'] in place on a
    wrong guess (the caller's session_state dict, since dicts are
    passed by reference). Returns (True, None) or (False, error_message)."""
    if challenge is None:
        return False, "No verification code on file - request a new one."
    if challenge["attempts"] >= OTP_MAX_ATTEMPTS:
        return False, "Too many incorrect attempts. Request a new code."
    if datetime.now(timezone.utc) > datetime.fromisoformat(challenge["expires_at"]):
        return False, "That code expired. Request a new one."
    if _hash_otp((entered_otp or "").strip()) != challenge["otp_hash"]:
        challenge["attempts"] += 1
        return False, "Incorrect code."
    return True, None


def authenticate_user(username, password):
    """Returns the user dict if credentials are valid, else None."""
    username = username.strip()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, email, password_hash FROM users WHERE username = %s;",
                (username,),
            )
            user = cur.fetchone()

    if not user:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return None
    return {"id": user["id"], "username": user["username"], "email": user["email"]}


def log_activity(user_id, activity_type, details=None):
    """Record one row of user activity. Never raises — a logging failure
    shouldn't break the app for the user, so errors are swallowed after
    a single attempt."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO activity_log (user_id, activity_type, details) "
                    "VALUES (%s, %s, %s);",
                    (user_id, activity_type, psycopg2.extras.Json(details or {})),
                )
            conn.commit()
    except Exception:
        pass
