# Supermemory Backend (opt-in cloud memory)

An optional long-term memory backend backed by the hosted (or self-hosted) [supermemory](https://supermemory.ai) service. It is **off by default**. Jarvis's local three-tier memory (hot dialogue window, SQLite diary, knowledge graph) remains the default and the source of truth, and nothing leaves the device unless the user explicitly opts in with an API key.

This module is the single, contained seam through which Jarvis talks to supermemory. Everything else in the codebase calls it through one feature gate.

## Scope

- File: `src/jarvis/memory/supermemory_backend.py`.
- Spec sibling: this file.
- Write callers: `update_daily_conversation_summary` and `update_diary_from_dialogue_memory` in `src/jarvis/memory/conversation.py` (diary flush path), reached from `daemon.py` which passes the live `cfg`.
- Read callers: the Step 4 memory-enrichment block in `src/jarvis/reply/engine.py` (diary merge in Step 4a, profile merge in Step 4b).
- Optional dependency: the `supermemory` PyPI package (see `requirements.txt`). Not a hard requirement.

## The single gate

`is_enabled(cfg)` returns true only when **both** `cfg.supermemory_enabled` is true **and** `cfg.supermemory_api_key` is non-empty. Every public function returns immediately when it is false. This guarantees the offline-first invariant:

- **Disabled ⇒ no import.** The `from supermemory import Supermemory` line lives inside `_get_client`, which is never reached when disabled. A stock install without the package keeps working.
- **Disabled ⇒ no network.** No client is constructed and no request is made.
- **Disabled ⇒ no behaviour change.** Read functions return `[]` and write functions return `None`, so enrichment and the diary flush are byte-identical to the pre-feature path.

## Fail-open everywhere

Jarvis is offline-first, so a remote failure must be indistinguishable from "no remote results" and must never break a turn:

- `_get_client` returns `None` (never raises) when the package is missing, the key is absent, or construction fails. A missing package is logged once via `debug_log`.
- The client is built with a short timeout and `max_retries=0` (guarded against SDK versions that don't accept those kwargs) so a hung or flaky call can't stall a reply.
- Every public function wraps its body in `try/except Exception → debug_log(..., "memory")` and returns an empty result.

## Model reconciliation

supermemory has no notion of Jarvis's tree structure or its USER/DIRECTIVES/WORLD branches, and Jarvis has no notion of supermemory's server-side contradiction resolution or forgetting. We do not round-trip structure. supermemory is treated as an additive, parallel episodic + profile store; local memory stays the source of truth and remote hits are merged in, never substituted.

| Jarvis | supermemory | Reconciliation |
|--------|-------------|----------------|
| Per-install single user | `container_tag` namespace | One stable tag; `cfg.supermemory_container_tag` or `"jarvis_default_user"`. |
| Diary daily summary | `client.add(content=...)` | `mirror_diary_summary` sends the scrubbed summary with `metadata={type: diary, date, topics}`. |
| Knowledge-graph fact | `client.add(content=...)` | `mirror_graph_fact` sends each newly stored fact with `metadata={type: fact, node}`. |
| Diary hybrid search → `List[str]` | `client.search.memories(q=...)` | `search_memories` returns `[YYYY-MM-DD] text` strings matching the local diary format. |
| Graph question lookup / warm profile | `client.profile(container_tag, q=...)` | `fetch_profile_facts` returns static + dynamic facts as `[profile] ...` strings. |

## Read path

- **Diary merge (engine Step 4a).** After the local keyword search returns its `List[str]`, `search_memories` is queried with the same keywords and `merge_memory_results` combines the two: dedupe by exact text (local wins on a tie), sort newest-first by the `[YYYY-MM-DD]` prefix, cap to `cfg.memory_enrichment_max_results`. The result flows through the existing small-model digest unchanged.
- **Profile merge (engine Step 4b).** When the extractor produced implicit `questions`, `fetch_profile_facts` is appended to `graph_parts` and enters the same "Information the user has shared with you in prior conversations" block and digest. This runs even when the local `<2 content words` skip fires, so remote facts can still surface.
- Both merges are gated by `is_enabled(cfg)` and only run when the planner requested a memory search (the `searchMemory` directive / recall gate), so the feature never adds unsolicited network calls.

## Write path

`cfg` is threaded as an optional `cfg=None` kwarg from `daemon.py` → `update_diary_from_dialogue_memory` → `update_daily_conversation_summary`. The default `None` preserves every existing caller and test (no mirroring). When `cfg` is present and `cfg.supermemory_mirror_writes` is true:

- the scrubbed daily summary is mirrored right after the local `upsert_conversation_summary` + embedding write;
- each newly stored graph fact is mirrored inside the existing non-fatal graph block.

Re-mirroring is safe: each `add` carries a stable `custom_id` (`jarvis-diary-{date}` for the cumulative daily summary, a content hash for facts) so a re-mirror updates one logical document rather than accumulating partial copies. SDK versions that reject `custom_id` fall back to a plain `add`.

## Startup validation

Because the backend fails open, a misconfiguration (missing package, wrong key, unreachable host, or an SDK whose surface differs from what we call) would otherwise present as a silent per-turn no-op. `startup_check(cfg)` is called once from `daemon.py` startup: when enabled, it makes one bounded `profile` probe and prints either `🌐 Supermemory connected: <endpoint> (container: <tag>)` or a `⚠️` warning with the error, so the problem surfaces in the console. It never raises and never blocks longer than the client's short timeout. When disabled it is silent and makes no network call.

The live smoke test `tests/test_supermemory_live.py` (skipped unless `SUPERMEMORY_API_KEY` is set) exercises the real SDK end-to-end against a throwaway container tag, which is the one thing the fail-open unit tests cannot prove.

## Privacy

Per the project's "data privacy comes first" rule, only **already-scrubbed** text leaves the device: the diary summary has passed redaction (`redact`) and the deterministic deflection scrub (`scrub_deflection_sentences`), and graph facts are extracted from that same scrubbed summary. Raw transcripts and the hot dialogue window are never sent. Enabling the backend is explicit consent (it requires a non-empty key), and `supermemory_base_url` lets a user point at a self-hosted instance so nothing reaches a third party.

## Config

- `supermemory_enabled` (bool, default false)
- `supermemory_api_key` (str, default ""; also read from the `SUPERMEMORY_API_KEY` env var)
- `supermemory_base_url` (str, default ""; empty = hosted `https://api.supermemory.ai`)
- `supermemory_container_tag` (str, default ""; empty = `"jarvis_default_user"`)
- `supermemory_mirror_writes` (bool, default true; off = recall-only)

A silent config migration (v2) introduces these keys with safe defaults.
