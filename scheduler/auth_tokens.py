"""JWT для сессий браузера."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import jwt
from fastapi import HTTPException

from .user_store import User, UserStore

JWT_ALG = "HS256"
JWT_TTL_SEC = 60 * 60 * 24 * 7  # 7 дней


def jwt_secret() -> str:
    secret = os.getenv("DISTGPU_JWT_SECRET", "").strip()
    if not secret:
        secret = os.getenv("WORKER_TOKEN") or os.getenv("TOKEN") or ""
    if len(secret) < 16:
        raise RuntimeError(
            "Задайте DISTGPU_JWT_SECRET (≥16 символов) в .env на координаторе"
        )
    return secret


def create_access_token(user: User) -> str:
    now = int(time.time())
    payload = {
        "sub": user.id,
        "username": user.username,
        "iat": now,
        "exp": now + JWT_TTL_SEC,
    }
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALG)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, jwt_secret(), algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Сессия истекла") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Неверный токен") from e


def user_from_token(token: str, store: UserStore) -> User:
    payload = decode_access_token(token)
    uid = payload.get("sub")
    if not isinstance(uid, str) or not uid:
        raise HTTPException(status_code=401, detail="Неверный токен")
    user = store.get_by_id(uid)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user
