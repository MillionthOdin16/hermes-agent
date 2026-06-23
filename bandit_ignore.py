import urllib.request

def _http_ok(url: str, timeout: float) -> bool:
    try:
        if not url.startswith(("http://", "https://")):
            return False
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False
