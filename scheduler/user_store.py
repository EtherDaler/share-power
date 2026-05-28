"""Пользователи координатора (SQLite, тот же файл что и jobs)."""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


@dataclass
class User:
    id: str
    username: str
    created_at: float


class UserStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _validate_username(username: str) -> str:
        u = username.strip()
        if not _USERNAME_RE.match(u):
            raise ValueError(
                "Имя пользователя: 3–32 символа, латиница, цифры, _ . -"
            )
        return u

    @staticmethod
    def _validate_password(password: str) -> str:
        if len(password) < 8:
            raise ValueError("Пароль не короче 8 символов")
        return password

    def register(self, username: str, password: str) -> User:
        u = self._validate_username(username)
        self._validate_password(password)
        uid = uuid.uuid4().hex[:12]
        pw_hash = _pwd.hash(password)
        created = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (uid, u, pw_hash, created),
                )
                conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError("Пользователь с таким именем уже существует") from e
        return User(id=uid, username=u, created_at=created)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        u = self._validate_username(username)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users "
                "WHERE username = ?",
                (u,),
            ).fetchone()
        if not row or not _pwd.verify(password, row["password_hash"]):
            return None
        return User(
            id=row["id"],
            username=row["username"],
            created_at=float(row["created_at"]),
        )

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            created_at=float(row["created_at"]),
        )
