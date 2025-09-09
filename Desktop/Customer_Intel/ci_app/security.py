# ci_app/security.py
from __future__ import annotations

import os
import secrets
from functools import wraps
from typing import Callable, Iterable

from flask import (
    abort,
    current_app,
    g,
    jsonify,
    redirect,
    request,
    url_for,
)
from flask_login import current_user


# ──────────────────────────────────────────────────────────────────────────────
# RBAC decorators
#   - roles_required(...): user must have ANY of the listed roles (OR)
#   - roles_required_all(...): user must have ALL of the listed roles (AND)
#   - role_required("admin") convenience alias for a single role
# They return JSON (401/403) for API calls and redirect to login for HTML.
# ──────────────────────────────────────────────────────────────────────────────

def _wants_json() -> bool:
    # JSON for /api/* or explicit Accept header / X-Requested-With
    if request.path.startswith("/api/"):
        return True
    accepts = request.headers.get("Accept", "")
    xr = request.headers.get("X-Requested-With", "")
    return "application/json" in accepts or xr.lower() == "xmlhttprequest"


def _handle_unauthorized() :
    if _wants_json():
        return jsonify({"error": "unauthorized"}), 401
    # HTML: send to login page with next=…
    login_endpoint = current_app.config.get("LOGIN_VIEW", "auth.login")
    return redirect(url_for(login_endpoint, next=request.url))


def _handle_forbidden():
    if _wants_json():
        return jsonify({"error": "forbidden"}), 403
    abort(403)


def roles_required(*role_names: str) -> Callable:
    """
    Require the user to have ANY of the given roles.
    Example: @roles_required("admin", "analyst")
    """
    if not role_names:
        raise ValueError("roles_required needs at least one role name")

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*a, **k):
            if not current_user.is_authenticated:
                return _handle_unauthorized()
            ok = any(current_user.has_role(r) for r in role_names)
            if not ok:
                return _handle_forbidden()
            return fn(*a, **k)
        return wrapper
    return deco


def roles_required_all(*role_names: str) -> Callable:
    """
    Require the user to have ALL of the given roles (AND).
    Example: @roles_required_all("admin", "billing")
    """
    if not role_names:
        raise ValueError("roles_required_all needs at least one role name")

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*a, **k):
            if not current_user.is_authenticated:
                return _handle_unauthorized()
            ok = all(current_user.has_role(r) for r in role_names)
            if not ok:
                return _handle_forbidden()
            return fn(*a, **k)
        return wrapper
    return deco


def role_required(name: str) -> Callable:
    """Convenience for a single role."""
    return roles_required(name)


# ──────────────────────────────────────────────────────────────────────────────
# Security headers
#   - Use nonce-based CSP by default
#   - Optional CDN allowances via app.config (CSP_* lists)
#   - HSTS when over HTTPS
#   - COOP/CORP/Referrer-Policy, X-Frame-Options, etc.
# Put `security_before_request(app)` and `add_security_headers` in your factory.
# ──────────────────────────────────────────────────────────────────────────────

def _list_from_config(key: str, default: Iterable[str]) -> list[str]:
    val = current_app.config.get(key)
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple, set)):
        return list(val)
    # comma-separated string support
    return [x.strip() for x in str(val).split(",") if x.strip()]


def _compute_csp(nonce: str | None) -> str:
    """
    Build a tight, configurable Content-Security-Policy.
    Add your CDNs in config:
      CSP_SCRIPT_SRC = ["'self'", "https://cdn.jsdelivr.net"]
      CSP_STYLE_SRC  = ["'self'", "https://fonts.googleapis.com"]
      CSP_FONT_SRC   = ["'self'", "https://fonts.gstatic.com", "data:"]
      CSP_IMG_SRC    = ["'self'", "data:"]
      CSP_CONNECT_SRC= ["'self'"]
      CSP_FRAME_SRC  = ["'self'"]
    Set CSP_ALLOW_UNSAFE_INLINE=True only if you absolutely need it.
    """
    allow_inline = bool(current_app.config.get("CSP_ALLOW_UNSAFE_INLINE", False))

    script_src = _list_from_config("CSP_SCRIPT_SRC", ["'self'"])
    style_src  = _list_from_config("CSP_STYLE_SRC",  ["'self'"])
    img_src    = _list_from_config("CSP_IMG_SRC",    ["'self'", "data:"])
    font_src   = _list_from_config("CSP_FONT_SRC",   ["'self'", "data:"])
    connect_src= _list_from_config("CSP_CONNECT_SRC",["'self'"])
    frame_src  = _list_from_config("CSP_FRAME_SRC",  ["'self'"])

    if nonce:
        script_src += [f"'nonce-{nonce}'", "'strict-dynamic'"]
        # Many libraries (e.g., Tailwind CDN) inject runtime <style> without nonce.
        # If explicitly allowed in config, also permit unsafe-inline styles in addition
        # to the nonce so the UI renders correctly under strict CSP.
        if allow_inline and "'unsafe-inline'" not in style_src:
            style_src += ["'unsafe-inline'"]
        # Several templates include inline <script> blocks for charts; when enabled,
        # allow unsafe-inline scripts alongside the nonce to avoid blocking them.
        if allow_inline and "'unsafe-inline'" not in script_src:
            script_src += ["'unsafe-inline'"]
    elif allow_inline:
        # Fallback for legacy pages – prefer nonces instead
        script_src += ["'unsafe-inline'"]
        style_src  += ["'unsafe-inline'"]

    # Disallow eval by default; can be extended via config if needed
    script_src += ["'unsafe-eval'"] if current_app.config.get("CSP_ALLOW_UNSAFE_EVAL") else []

    # Assemble policy
    parts = [
        "default-src 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        f"img-src {' '.join(dict.fromkeys(img_src))}",
        f"style-src {' '.join(dict.fromkeys(style_src))}",
        # Allow safe use of element style attributes (needed by some UI libs like Choices.js)
        "style-src-attr 'unsafe-inline'",
        f"script-src {' '.join(dict.fromkeys(script_src))}",
        f"font-src {' '.join(dict.fromkeys(font_src))}",
        f"connect-src {' '.join(dict.fromkeys(connect_src))}",
        f"frame-src {' '.join(dict.fromkeys(frame_src))}",
    ]
    # Only upgrade insecure requests when desired; skip on localhost to avoid HTTPS errors in dev
    try:
        from flask import request as _rq  # type: ignore
        host = (_rq.host or "").split(":")[0]
        is_local = host in ("127.0.0.1", "localhost")
    except Exception:
        is_local = False
    if current_app.config.get("CSP_UPGRADE_INSECURE_REQUESTS", True) and not is_local:
        parts.append("upgrade-insecure-requests")
    return "; ".join(parts)


def add_security_headers(resp):
    """Attach strong, production security headers to every response."""
    # Core headers
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("X-XSS-Protection", "0")  # deprecated; rely on CSP
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy",
                            "geolocation=(), microphone=(), camera=(), browsing-topics=()")

    # HSTS (only when HTTPS)
    # Respect reverse proxy headers if configured
    is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
    if is_secure:
        max_age = int(current_app.config.get("HSTS_SECONDS", 31536000))  # 1 year
        include_sub = "; includeSubDomains" if current_app.config.get("HSTS_INCLUDE_SUBDOMAINS", True) else ""
        preload = "; preload" if current_app.config.get("HSTS_PRELOAD", False) else ""
        resp.headers.setdefault("Strict-Transport-Security", f"max-age={max_age}{include_sub}{preload}")

    # COOP/CORP (safe defaults that won't break typical CDNs)
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")

    # CSP with per-request nonce (honor CSP_ENABLE toggle)
    if current_app.config.get("CSP_ENABLE", True):
        nonce = getattr(g, "csp_nonce", None)
        csp_override = current_app.config.get("CSP_OVERRIDE")
        resp.headers["Content-Security-Policy"] = csp_override or _compute_csp(nonce)

    return resp


# ──────────────────────────────────────────────────────────────────────────────
# App factory helpers
#   Use these in create_app():
#     security_before_request(app)
#     @app.after_request -> add_security_headers
#     enforce_https(app)  (optional)
# ──────────────────────────────────────────────────────────────────────────────

def security_before_request(app):
    """Register a before_request hook to create a CSP nonce each request."""
    @app.before_request
    def _set_nonce():
        # Use a nonce only when running in stricter modes. In debug/testing or when
        # CSP_ALLOW_UNSAFE_INLINE is enabled, skip the nonce so 'unsafe-inline' remains effective
        # and legacy inline handlers keep working during local development.
        try:
            allow_inline = bool(app.config.get("CSP_ALLOW_UNSAFE_INLINE", False))
            if app.debug or app.testing or allow_inline:
                g.csp_nonce = None
            else:
                g.csp_nonce = secrets.token_urlsafe(16)
        except Exception:
            g.csp_nonce = None


def enforce_https(app):
    """
    Optionally redirect HTTP -> HTTPS in production.
    Set SECURE_SSL_REDIRECT=true and (optionally) SECURE_PROXY_SSL_HEADER.
    """
    redirect_enabled = bool(app.config.get("SECURE_SSL_REDIRECT", False))

    if not redirect_enabled:
        return

    @app.before_request
    def _https_redirect():
        # Skip in debug/testing and for local dev hosts
        if app.debug or app.testing:
            return
        # Already secure?
        if request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https":
            return
        # Only GET/HEAD are safe to redirect automatically
        if request.method in ("GET", "HEAD"):
            target = request.url.replace("http://", "https://", 1)
            return redirect(target, code=308)
        # For other verbs, refuse (prevents credential leakage)
        return _handle_unauthorized()
