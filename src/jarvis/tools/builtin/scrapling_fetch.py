"""Optional escalation to a locally-installed Scrapling CLI.

Jarvis's normal page fetch is plain ``requests`` + BeautifulSoup: fast, cheap,
and offline-friendly, but blind to JavaScript-rendered pages and helpless
against anti-bot walls. When the user opts in (``scrapling_fetch_enabled``),
this helper retries a single URL through the ``scrapling`` CLI, walking the
ladder the official skill recommends:

    extract get  ->  extract fetch  ->  extract stealthy-fetch

``get`` is a TLS-impersonating HTTP request; ``fetch`` renders the page in a
headless browser; ``stealthy-fetch`` adds anti-bot evasion (and, optionally,
Cloudflare solving). We stop at the first stage that returns content.

Design guarantees that keep this safe on a privacy-first, latency-sensitive
voice assistant:

* **Opt-in**: a no-op unless ``cfg.scrapling_fetch_enabled`` is true, so the
  default install never spawns a browser and needs no extra dependency.
* **SSRF-guarded**: the URL is validated with the same ``_is_public_url`` guard
  the web-search cascade uses before any process is spawned.
* **Bounded**: each stage is capped by the remaining wall-clock budget; once
  the budget falls below a floor the ladder stops rather than firing a doomed
  browser launch.
* **AI-targeted**: every invocation passes ``--ai-targeted`` so Scrapling
  returns main-content Markdown with hidden elements sanitised (a
  prompt-injection mitigation) and ads stripped (token saving).
* **Fail-soft**: a missing binary, a crash, or a timeout degrades to ``None``;
  callers treat ``None`` as "no escalation happened" and fall back to their
  existing behaviour.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Callable, Optional

from ...debug import debug_log
from .web_search import _is_public_url

# Default per-stage wall-clock when no chain deadline is supplied. Browser
# stages are heavy, so this is generous; callers on the voice fast path pass a
# tighter ``deadline`` instead.
_DEFAULT_STAGE_TIMEOUT_SEC = 20.0
# Below this remaining budget a browser launch cannot realistically finish, so
# the ladder stops instead of firing a request doomed to time out.
_MIN_STAGE_TIMEOUT_SEC = 3.0
# Grace added to the subprocess timeout over the in-tool timeout so the browser
# has room to shut down cleanly before we kill it.
_SUBPROCESS_GRACE_SEC = 5.0

# Stages in escalation order. The bool marks whether Cloudflare solving is even
# applicable to the stage (only the stealth fetcher can solve challenges).
_STAGES = (("get", False), ("fetch", False), ("stealthy-fetch", True))


def _stage_timeout(deadline: Optional[float]) -> Optional[float]:
    """Return the per-stage timeout in seconds, or ``None`` if out of budget."""
    if deadline is None:
        return _DEFAULT_STAGE_TIMEOUT_SEC
    remaining = deadline - time.monotonic()
    if remaining < _MIN_STAGE_TIMEOUT_SEC:
        return None
    return min(remaining, _DEFAULT_STAGE_TIMEOUT_SEC)


def _run_scrapling(
    subcmd: str,
    url: str,
    *,
    binary: str,
    timeout_sec: float,
    solve_cloudflare: bool = False,
) -> Optional[str]:
    """Run a single ``scrapling extract <subcmd>`` and return its output text.

    Scrapling writes to a file path (it has no stdout mode for extraction), so
    we hand it a temp ``.md`` target, read it back, and delete it. Returns the
    stripped content, or ``None`` on any failure or empty result.
    """
    fd, out_path = tempfile.mkstemp(suffix=".md", prefix="jarvis_scrapling_")
    os.close(fd)
    try:
        # Command layout is positional-stable: callers and tests rely on
        # argv[2] == subcmd and argv[4] == out_path.
        argv = [binary, "extract", subcmd, url, out_path, "--ai-targeted"]
        # ``get`` takes a timeout in seconds; the browser stages take
        # milliseconds. Match each so the cap actually means what we intend.
        if subcmd == "get":
            argv += ["--timeout", str(int(timeout_sec))]
        else:
            argv += ["--timeout", str(int(timeout_sec * 1000))]
        if subcmd == "stealthy-fetch" and solve_cloudflare:
            argv.append("--solve-cloudflare")

        subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec + _SUBPROCESS_GRACE_SEC,
        )
        with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read().strip()
        return content or None
    except FileNotFoundError:
        debug_log(
            f"scrapling binary '{binary}' not found; cannot escalate", "web",
        )
        return None
    except subprocess.TimeoutExpired:
        debug_log(f"scrapling {subcmd} timed out for {url}", "web")
        return None
    except Exception as e:  # pragma: no cover (safety net)
        debug_log(f"scrapling {subcmd} failed for {url}: {e}", "web")
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def scrapling_fetch(
    url: str,
    *,
    cfg,
    deadline: Optional[float] = None,
    query: Optional[str] = None,
    user_print: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Escalate a single URL through the Scrapling CLI when the user opts in.

    Args:
        url: The page to fetch.
        cfg: Settings object; reads ``scrapling_fetch_enabled``,
            ``scrapling_binary``, ``scrapling_solve_cloudflare``.
        deadline: Optional monotonic timestamp bounding total time spent here.
            When omitted, each stage uses ``_DEFAULT_STAGE_TIMEOUT_SEC``.
        query: Unused today; reserved so the web-search cascade can pass the
            user's query for future relevance-aware extraction.
        user_print: Optional ephemeral progress sink (emoji-prefixed).

    Returns:
        Extracted Markdown/text, or ``None`` when disabled, unsafe, out of
        budget, or unable to fetch.
    """
    if not getattr(cfg, "scrapling_fetch_enabled", False):
        return None
    if deadline is not None and (deadline - time.monotonic()) <= 0:
        return None
    if not _is_public_url(url):
        debug_log(f"scrapling escalation refused non-public URL: {url}", "web")
        return None

    binary = (getattr(cfg, "scrapling_binary", "scrapling") or "scrapling").strip()
    solve = bool(getattr(cfg, "scrapling_solve_cloudflare", False))
    if user_print:
        user_print("🕷️ Escalating to Scrapling…")
    debug_log(f"scrapling escalation starting for {url}", "web")

    for subcmd, solvable in _STAGES:
        timeout_sec = _stage_timeout(deadline)
        if timeout_sec is None:
            debug_log("scrapling escalation out of budget; stopping ladder", "web")
            break
        content = _run_scrapling(
            subcmd,
            url,
            binary=binary,
            timeout_sec=timeout_sec,
            solve_cloudflare=solve and solvable,
        )
        if content:
            debug_log(
                f"scrapling {subcmd} returned {len(content)} chars for {url}",
                "web",
            )
            if user_print:
                user_print("✅ Scrapling fetched the page.")
            return content

    debug_log(f"scrapling escalation exhausted all stages for {url}", "web")
    return None
