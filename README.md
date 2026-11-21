# Free AI Model Aggregator — make.fun entry

Competition: `free-model-aggregator-01` — "The best free AI model aggregator" ($500).
One Node 22 server (`main.ts`, standard library only) that serves an
OpenAI- and Anthropic-compatible API over ten simulated free-tier providers.

## How it works

- **Startup:** reads the brief (`MAKEFUN_PROMPT_FILE`) for the provider
  directory, the quality table and the workload manifest. Probes every
  provider's models route to build the catalog it reports (falls back to
  the brief's quality table if a probe fails). Registers the port + catalog
  at `$MAKEFUN_ENV_URL/register`, then serves requests from `GET /work`
  until the workload is done and exits 0.
- **Routing:** for each request, derives requirements from the body
  (tools / vision / JSON mode) plus the manifest entry by arrival order
  (min_context; a >2000-char user text is treated as long-context as a
  backup). Candidates are the free models meeting the requirements,
  sorted by rating desc (ties: provider, model). Per-provider soft caps
  sit below each provider's sealed budget low end, so a provider is never
  burned out; gemini/zai caps stay low because they are the only
  tools+vision lifeline. The paid model
  `anthropic/claude-opus-5.5` (openrouter) is never picked.
- **Failover:** a 429/402 (or an unparseable answer) moves to the next
  candidate; a provider is blacklisted after 2 refusals (401 blacklists at
  once; 400 drops just that model). Exactly one successful provider call
  per request is used.
- **Wire fidelity:** OpenAI↔OpenAI passthrough (adds `stream_options`
  `include_usage` for streams); OpenAI→Anthropic and Anthropic→OpenAI
  conversions (system, tools, stops, images, max_tokens); both →Gemini
  (`:generateContent` / `:streamGenerateContent?alt=sse`). JSON-mode
  requests are never routed to the Anthropic-wire provider. Responses are
  rebuilt from the provider's echo with the provider's exact usage and a
  same-family stop reason. Streams are re-chunked in 13-char deltas with a
  final usage chunk (OpenAI) / message_delta (Anthropic).
- **Catalog:** `GET /v1/models` lists exactly the free model ids under
  both Bearer and x-api-key auth. `node main.ts --list` prints the
  catalog document byte-for-byte (empty `{"providers":[]}` outside a trial).

## Local testing (all three practice trials, kit harness)

```
score 99 (quality 99, served 100, wire 100, failover 100, catalog 100)
trial-0: 20/20 valid, 233/233 checks, 27 upstream calls
trial-1: 20/20 valid, 31 upstream calls
trial-2: 20/20 valid, 29 upstream calls
```

Run locally with the kit's env + scorer (see `/tmp/agg-test.sh`, scratch).
Two bugs found and fixed during testing: the brief-JSON extractor
landed inside a document instead of at its opening brace, and a
`const text` shadowed the `text` parameter in the Anthropic answer
parser (temporal-dead-zone ReferenceError, swallowed by the catch —
Anthropic-wire answers were silently rejected).

## Notes for submit

- Do NOT commit/bundle/check/submit from this build session — the parent
  does that. Repo is ready to commit as-is.
- Entry is deterministic (no wall clock, no unseeded randomness) and
  exits by itself ~165 s safety cap; the trial pod allows 180 s.
