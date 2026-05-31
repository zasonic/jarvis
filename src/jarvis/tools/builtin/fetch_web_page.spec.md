## Fetch Web Page Tool Spec

Fetches a single URL the user (or the reply LLM) named and returns its text so
the assistant can summarise or quote it. Distinct from `webSearch`: that tool
discovers pages, this one reads a page already in hand.

### Default pipeline (always available, offline-friendly)

1. Normalise the URL (prepend `https://` when no scheme is given).
2. `requests.get` with a desktop Chrome User-Agent, 15s timeout, redirects
   followed. The connection is released via a `with` block so a mid-parse
   exception cannot leak it.
3. Parse with BeautifulSoup: strip `script` / `style` / `meta` / `link` /
   `noscript`, take the title, collapse the text to non-trivial unique lines,
   cap at 500 lines.
4. Optionally extract up to 20 links when `include_links` is true (relative
   hrefs resolved to absolute).
5. Assemble `**Title** / **URL** / **Content**`, truncated to 50,000 chars.

This path is unchanged from its original behaviour and requires no extra
dependency. When BeautifulSoup is unavailable it falls back to raw text.

### Optional Scrapling escalation (opt-in)

The default path is blind to JavaScript-rendered pages and helpless against
anti-bot walls. When `scrapling_fetch_enabled` is true, the tool escalates to a
locally-installed `scrapling` CLI in two situations:

- **Thin content**: the cleaned extract is shorter than `_THIN_CONTENT_CHARS`
  (200) — the signature of a JavaScript shell that rendered to nothing.
- **Transport failure**: the `requests` call raised (timeout, 403, blocked).

Escalation is delegated to `scrapling_fetch.scrapling_fetch` (see
`scrapling_fetch.py`), which:

- is a **no-op unless the flag is set**, so the default install never spawns a
  browser and needs no extra dependency;
- validates the URL with the web-search `_is_public_url` SSRF guard before
  spawning anything;
- walks `extract get → extract fetch → extract stealthy-fetch`, stopping at the
  first non-empty result, with `--solve-cloudflare` on the stealth stage only
  when `scrapling_solve_cloudflare` is set;
- always passes `--ai-targeted` so the returned Markdown is main-content only,
  with hidden elements sanitised (a prompt-injection mitigation) and ads
  stripped;
- degrades to `None` on a missing binary, crash, or timeout.

On a non-`None` return the tool emits the escalated Markdown as the Content
block (same Title/URL/Content envelope, same 50,000-char cap). On `None` the
tool keeps its existing behaviour: the thin content is returned as-is, or the
original transport failure is reported.

### Configuration

- `scrapling_fetch_enabled` (bool, default `false`): master opt-in for the
  escalation. Off keeps the tool pure-`requests` and dependency-free.
- `scrapling_binary` (str, default `"scrapling"`): path or bare name of the CLI.
- `scrapling_solve_cloudflare` (bool, default `false`): allow the stealth stage
  to solve Cloudflare challenges (slower, more conspicuous).

### Concurrency

`fetchWebPage` declares `parallel_safe = True`: a read-only network fetch
with no shared-DB writes, so the planner's direct-exec path may dispatch it
concurrently with other parallel-safe steps in one turn (see
`src/jarvis/reply/planner.spec.md` → "Parallel batch execution").

### Behavioural guarantees for tests

1. **Flag off**: byte-for-byte identical to the pre-escalation behaviour;
   `scrapling_fetch` is never called.
2. **Thin content + flag on**: a near-empty extract escalates and, on a
   non-`None` return, the reply carries Scrapling's text.
3. **Transport failure + flag on**: escalation is attempted before the failure
   is reported; a non-`None` return turns the failure into a success.
4. **Escalation declined**: when `scrapling_fetch` returns `None`, the original
   thin content or failure stands.
5. **Rich content**: a page that already yields ample text never escalates.

### Non-goals

- Crawling / multi-page flows — that is the `scrapling mcp` server's job (see
  README MCP integrations), not this single-URL tool.
- Bundling Scrapling — it stays an optional, user-installed dependency.
