# =============================================================================
#  auth.py  —  FinSight AI v2 · Authentication Module
# =============================================================================
import os, secrets
from functools  import wraps
from flask      import request, jsonify, session
import bcrypt
from database   import (create_user, get_user_by_email,
                         get_user_by_id, update_last_login)


# =============================================================================
#  Password helpers
# =============================================================================

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# =============================================================================
#  Token store (in-memory for simplicity; swap for Redis in prod)
# =============================================================================
_TOKENS: dict[str, int] = {}   # token → user_id


def issue_token(user_id: int) -> str:
    token = secrets.token_hex(32)
    _TOKENS[token] = user_id
    return token


def revoke_token(token: str):
    _TOKENS.pop(token, None)


def get_user_id_from_token(token: str) -> int | None:
    return _TOKENS.get(token)


# =============================================================================
#  Decorator: require valid Bearer token OR Flask session
# =============================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # 1. Bearer token (API / JS fetch)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            uid = get_user_id_from_token(auth.split(" ", 1)[1])
            if uid:
                request.current_user_id = uid
                return f(*args, **kwargs)

        # 2. Session cookie (browser)
        uid = session.get("user_id")
        if uid:
            request.current_user_id = uid
            return f(*args, **kwargs)

        return jsonify({"error": "Authentication required"}), 401
    return decorated


def optional_auth(f):
    """Sets request.current_user_id if authenticated, else None."""
    @wraps(f)
    def decorated(*args, **kwargs):
        request.current_user_id = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            uid = get_user_id_from_token(auth.split(" ", 1)[1])
            if uid:
                request.current_user_id = uid
        if not request.current_user_id:
            request.current_user_id = session.get("user_id")
        return f(*args, **kwargs)
    return decorated


# =============================================================================
#  Route registration helpers
# =============================================================================

def register_auth_routes(app):
    """Attach /api/auth/* routes to the Flask app."""

    @app.route("/api/auth/register", methods=["POST"])
    def register():
        body = request.get_json(silent=True) or {}
        name  = (body.get("name", "") or "").strip()
        email = (body.get("email", "") or "").strip().lower()
        pw    = (body.get("password", "") or "").strip()

        if not name or not email or not pw:
            return jsonify({"error": "Name, email and password are required"}), 400
        if len(pw) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400

        uid = create_user(name, email, hash_password(pw))
        if uid is None:
            return jsonify({"error": "Email already registered"}), 409

        token = issue_token(uid)
        session["user_id"] = uid
        return jsonify({"message": "Account created", "token": token,
                        "user": {"id": uid, "name": name, "email": email}}), 201


    @app.route("/api/auth/login", methods=["POST"])
    def login():
        body  = request.get_json(silent=True) or {}
        email = (body.get("email", "") or "").strip().lower()
        pw    = (body.get("password", "") or "").strip()

        user = get_user_by_email(email)
        if not user or not check_password(pw, user["password_hash"]):
            return jsonify({"error": "Invalid email or password"}), 401

        update_last_login(user["id"])
        token = issue_token(user["id"])
        session["user_id"] = user["id"]
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id":    user["id"],
                "name":  user["name"],
                "email": user["email"],
            }
        })


    @app.route("/api/auth/logout", methods=["POST"])
    def logout():
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            revoke_token(auth.split(" ", 1)[1])
        session.pop("user_id", None)
        return jsonify({"message": "Logged out"})


    @app.route("/api/auth/me")
    def me():
        uid = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            uid = get_user_id_from_token(auth.split(" ", 1)[1])
        if not uid:
            uid = session.get("user_id")
        if not uid:
            return jsonify({"error": "Not authenticated"}), 401
        user = get_user_by_id(uid)
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({
            "id":    user["id"],
            "name":  user["name"],
            "email": user["email"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
        })
