import urllib.request

def _http_ok(url: str, timeout: float) -> bool:
    if not url.startswith(("http://", "https://")):
        return False

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False
