1. **Identify and validate urlopen calls in `gateway/relay/__init__.py` and `tui_gateway/server.py`:**
   - In `gateway/relay/__init__.py` inside `_post_provisioning()`, add a check before making the request: `if not provision_url.startswith(("http://", "https://")): raise ValueError(...)`. Also, append `# nosec B310` to the `urllib.request.urlopen` call on line 420.
   - In `gateway/relay/__init__.py` inside `_post_policy()`, add a check before making the request: `if not policy_url.startswith(("http://", "https://")): raise ValueError(...)`. Also, append `# nosec B310` to the `urllib.request.urlopen` call on line 603.
   - In `tui_gateway/server.py` inside `_http_ok()`, add a check before making the request: `if not url.startswith(("http://", "https://")): return False`. Also, append `# nosec B310` to the `urllib.request.urlopen` call on line 12910.

2. **Update `.jules/sentinel.md` with critical learning:**
   - Create `.jules/sentinel.md` (or append to it) with the finding that `urllib.request.urlopen` permits `file://` schemes by default, creating a Server-Side Request Forgery (SSRF) and local file read vulnerability if the URL is not strictly validated to begin with `http://` or `https://`.

3. **Run all tests and linters:**
   - Run workspace-wide linting: `uv run ruff check .`
   - Run testing suite: `uv run scripts/run_tests.sh .`
   - Run Bandit security scanner to confirm the issues are fixed: `uv run --with bandit bandit -r agent/ gateway/ tui_gateway/ cron/ -ll`

4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**

5. **Create a Pull Request:**
   - Use the `submit` tool to create a PR.
   - Branch: `sentinel-ssrf-urlopen-fix`
   - Title: `🛡️ Sentinel: [MEDIUM] Fix SSRF vulnerabilities in urlopen`
   - Description:
     ```markdown
     🚨 Severity: MEDIUM
     💡 Vulnerability: Server-Side Request Forgery (SSRF) and local file read via `urllib.request.urlopen` allowing the `file://` scheme by default.
     🎯 Impact: If user input reaches these URL parameters, attackers could read arbitrary local files or access internal network services.
     🔧 Fix: Explicitly validated that URLs start with `http://` or `https://` before calling `urlopen` and added `# nosec B310` to suppress Bandit warnings.
     ✅ Verification: Run `uv run --with bandit bandit -r agent/ gateway/ tui_gateway/ cron/ -ll` to verify B310 is resolved.
     ```
