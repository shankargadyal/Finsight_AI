# =============================================================================
#  database.py  —  FinSight AI v2 · SQLite Database Layer
# =============================================================================
import sqlite3, json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("finsight.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            symbol     TEXT    NOT NULL,
            added_at   TEXT    DEFAULT (datetime('now')),
            UNIQUE(user_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS search_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            symbol     TEXT    NOT NULL,
            searched_at TEXT   DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS prediction_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            symbol          TEXT    NOT NULL,
            last_close      REAL,
            ensemble_d7     REAL,
            recommendation  TEXT,
            confidence      REAL,
            risk_score      REAL,
            risk_level      TEXT,
            sentiment_label TEXT,
            predicted_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
            session_id TEXT,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    print("[db] ✅ Database initialised")


# =============================================================================
#  User helpers
# =============================================================================

def create_user(name: str, email: str, password_hash: str) -> int | None:
    try:
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?,?,?)",
            (name, email, password_hash)
        )
        conn.commit()
        uid = cur.lastrowid
        conn.close()
        return uid
    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email: str) -> dict | None:
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(uid: int) -> dict | None:
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_last_login(uid: int):
    conn = get_db()
    conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.utcnow().isoformat(), uid))
    conn.commit()
    conn.close()


# =============================================================================
#  Watchlist helpers
# =============================================================================

def add_to_watchlist(user_id: int, symbol: str) -> bool:
    try:
        conn = get_db()
        conn.execute("INSERT OR IGNORE INTO watchlist (user_id, symbol) VALUES (?,?)", (user_id, symbol))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def remove_from_watchlist(user_id: int, symbol: str):
    conn = get_db()
    conn.execute("DELETE FROM watchlist WHERE user_id=? AND symbol=?", (user_id, symbol))
    conn.commit()
    conn.close()


def get_watchlist(user_id: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT symbol, added_at FROM watchlist WHERE user_id=? ORDER BY added_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
#  Search history helpers
# =============================================================================

def log_search(user_id: int, symbol: str):
    conn = get_db()
    conn.execute("INSERT INTO search_history (user_id, symbol) VALUES (?,?)", (user_id, symbol))
    conn.commit()
    conn.close()


def get_recent_searches(user_id: int, limit: int = 10) -> list:
    conn = get_db()
    rows = conn.execute(
        """SELECT symbol, MAX(searched_at) as last_searched
           FROM search_history WHERE user_id=?
           GROUP BY symbol ORDER BY last_searched DESC LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
#  Prediction history helpers
# =============================================================================

def save_prediction(user_id: int, data: dict):
    conn = get_db()
    conn.execute(
        """INSERT INTO prediction_history
           (user_id, symbol, last_close, ensemble_d7, recommendation,
            confidence, risk_score, risk_level, sentiment_label)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            user_id,
            data.get("symbol"),
            data.get("last_close"),
            data.get("ensemble_d7"),
            data.get("recommendation"),
            data.get("confidence"),
            data.get("risk_score"),
            data.get("risk_level"),
            data.get("sentiment_label"),
        )
    )
    conn.commit()
    conn.close()


def get_prediction_history(user_id: int, limit: int = 20) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM prediction_history WHERE user_id=? ORDER BY predicted_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
#  Chat history helpers
# =============================================================================

def save_chat_message(user_id: int | None, session_id: str, role: str, content: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_history (user_id, session_id, role, content) VALUES (?,?,?,?)",
        (user_id, session_id, role, content)
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str, limit: int = 30) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]
