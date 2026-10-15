
from __future__ import annotations

from urllib.parse import urljoin, urlsplit, urlunsplit


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    """
    Normalize a URL for deduplication.

    Returns None for unsupported or malformed URLs.
    Preserves meaningful query parameters and trailing slashes.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()

    # Resolve relative links when a base URL is supplied.
    if base_url:
        url = urljoin(base_url, url)

    try:
        parts = urlsplit(url)

        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"}:
            return None

        # Reject URLs containing embedded credentials.
        if parts.username is not None or parts.password is not None:
            return None

        hostname = parts.hostname
        if not hostname:
            return None

        hostname = hostname.rstrip(".").lower()
        if not hostname:
            return None

        # Validate the port and omit default ports.
        port = parts.port

        if port is not None and not 1 <= port <= 65535:
            return None

        if ":" in hostname:  # IPv6 literal
            host = f"[{hostname}]"
        else:
            host = hostname.encode("idna").decode("ascii")

        if port is not None and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            host = f"{host}:{port}"

        # Remove common tracking parameters without reordering
        # or rewriting the meaningful query parameters.
        kept_params = []

        for item in parts.query.split("&") if parts.query else []:
            key = item.partition("=")[0]
            key_lower = key.lower()

            if key_lower.startswith("utm_"):
                continue

            if key_lower in TRACKING_PARAMS:
                continue

            kept_params.append(item)

        query = "&".join(kept_params)

        # Keep the path and trailing slash as supplied.
        # URL fragments identify locations within a page, not new pages.
        path = parts.path or "/"

        return urlunsplit(
            (scheme, host, path, query, "")
        )

    except (ValueError, UnicodeError):
        return None


def is_within_scope(url: str, root_url: str) -> bool:
    """
    Check whether a URL stays within the root's origin and path scope.

    A root ending in '/' defines that directory as its scope.
    Otherwise, the root's containing directory is used.
    """
    candidate = normalize_url(url)
    root = normalize_url(root_url)

    if candidate is None or root is None:
        return False

    candidate_parts = urlsplit(candidate)
    root_parts = urlsplit(root)

    candidate_origin = (
        candidate_parts.scheme,
        candidate_parts.hostname,
        candidate_parts.port,
    )
    root_origin = (
        root_parts.scheme,
        root_parts.hostname,
        root_parts.port,
    )

    if candidate_origin != root_origin:
        return False

    root_path = root_parts.path

    if not root_path.endswith("/"):
        root_path = root_path.rsplit("/", 1)[0] + "/"

    return candidate_parts.path.startswith(root_path)
