import hashlib
import hmac
import os
import secrets
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import User


_ITERATIONS = 310_000
_TOKEN_SECRET = os.environ.get("NEIGHBORCONNECT_SECRET", "neighborconnect-local-demo-secret")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_access_token(user_id: int) -> str:
    payload = f"{user_id}.{int(time.time()) + 7 * 24 * 60 * 60}"
    signature = hmac.new(_TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return urlsafe_b64encode(f"{payload}.{signature}".encode()).decode().rstrip("=")


def get_current_user(
    authorization: str | None = Header(default=None), db: Session = Depends(get_db)
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    token = authorization[7:].strip()
    try:
        decoded = urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        user_id_text, expires_text, signature = decoded.split(".", 2)
        payload = f"{user_id_text}.{expires_text}"
        expected = hmac.new(_TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected) or int(expires_text) < int(time.time()):
            raise ValueError("Expired or invalid token")
        user_id = int(user_id_text)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Your session is invalid or has expired.") from None
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Your account is no longer available.")
    return user