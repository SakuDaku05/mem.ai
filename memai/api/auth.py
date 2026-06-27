"""
Authentication — API key + JWT OAuth.

Supports two auth modes:
  1. Bearer API key   (Authorization: Bearer sk-memai-...)
  2. JWT OAuth token  (POST /auth/token -> returns JWT)

Usage (FastAPI dependency injection):
    @router.get("/memory/{id}")
    async def get_memory(agent_id: str = Depends(require_api_key)):
        ...
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer

from memai.api.config import get_settings

logger = logging.getLogger(__name__)

# In-memory key store (production: replace with DB)
# Maps api_key -> agent_id
_api_keys: dict[str, str] = {}

bearer_scheme = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token", auto_error=False)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_jwt(data: dict, expires_minutes: int) -> str:
    from jose import jwt as jose_jwt
    settings = get_settings()
    payload = {
        **data,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jose_jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_jwt(token: str) -> dict:
    from jose import JWTError, jwt as jose_jwt
    settings = get_settings()
    try:
        return jose_jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def register_api_key(api_key: str, agent_id: str) -> None:
    """Register an API key -> agent_id mapping."""
    _api_keys[api_key] = agent_id


def create_api_key(agent_id: str) -> str:
    """Generate and register a new API key for an agent."""
    import secrets
    key = f"sk-memai-{secrets.token_urlsafe(24)}"
    _api_keys[key] = agent_id
    return key


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    return _api_keys.pop(api_key, None) is not None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_agent(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    oauth_token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """
    Resolve agent_id from either API key or JWT token.
    Raises 401 if neither is valid.
    """
    settings = get_settings()
    token = None

    if bearer:
        token = bearer.credentials
    elif oauth_token:
        token = oauth_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Bearer API key or JWT token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if it's a raw API key
    if token == settings.master_api_key:
        return "__master__"

    if token in _api_keys:
        return _api_keys[token]

    # Try to decode as JWT
    try:
        payload = _decode_jwt(token)
        agent_id = payload.get("sub")
        if agent_id:
            return agent_id
    except HTTPException:
        pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key or token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Optional dependency — doesn't raise, returns None if unauth
async def get_optional_agent(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    try:
        return await get_current_agent(bearer=bearer, oauth_token=None)
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# Auth router (token endpoint)
# ---------------------------------------------------------------------------

from fastapi import APIRouter
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

auth_router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    agent_id: str


class APIKeyRequest(BaseModel):
    agent_id: str


class APIKeyResponse(BaseModel):
    api_key: str
    agent_id: str


@auth_router.post("/token", response_model=TokenResponse, summary="Get JWT token")
async def login_for_token(form: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange API key (as password) for a JWT token.
    Username = agent_id, Password = API key.
    """
    settings = get_settings()
    api_key = form.password
    agent_id = form.username

    if api_key == settings.master_api_key or api_key in _api_keys:
        resolved_agent = _api_keys.get(api_key, agent_id)
        token = _create_jwt({"sub": resolved_agent}, settings.jwt_expire_minutes)
        return TokenResponse(
            access_token=token,
            expires_in=settings.jwt_expire_minutes * 60,
            agent_id=resolved_agent,
        )
    raise HTTPException(status_code=401, detail="Invalid API key")


@auth_router.post("/keys", response_model=APIKeyResponse, summary="Create API key")
async def create_key(
    request: APIKeyRequest,
    agent_id: str = Depends(get_current_agent),
):
    """Create a new API key for an agent. Requires master key."""
    settings = get_settings()
    # Only master can create keys for other agents
    if agent_id != "__master__" and agent_id != request.agent_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    key = create_api_key(request.agent_id)
    return APIKeyResponse(api_key=key, agent_id=request.agent_id)


@auth_router.get("/me", summary="Get current agent info")
async def get_me(agent_id: str = Depends(get_current_agent)):
    return {"agent_id": agent_id, "authenticated": True}
