import bcrypt
import hashlib

def _prehash(password: str) -> bytes:
    """Pre-hash password with SHA-256 to bypass bcrypt's 72 byte limit."""
    return hashlib.sha256(password.encode("utf-8")).digest()

def hash_password(password: str) -> str:
    # Hash the pre-hashed password string using bcrypt
    hashed = bcrypt.hashpw(_prehash(password), bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Verify the plain password matches the hashed password
    return bcrypt.checkpw(_prehash(plain_password), hashed_password.encode("utf-8"))
