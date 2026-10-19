# loci-litellm

LiteLLM proxy callbacks that read from and write to a Qdrant-backed memory store:

| Callback | Module | What it does |
|---|---|---|
| grounding hook | `grounding_hook.loci_grounding_hook` | Before a call, retrieves matching records from Qdrant and puts them in the request's instruction slot. |
| memory writer | `memory_writer.loci_memory_writer` | After a successful call, writes a prompt/answer preview into a Qdrant collection. |
| circuit monitor | `circuit_monitor.loci_circuit_monitor` | Counts consecutive transient failures per model group and logs when a circuit opens. Advisory: it does not change routing. |
| cost ledger | `cost_ledger.loci_cost_ledger` | Appends one metadata-only JSON line per call to a file. `python -m loci_litellm.cost_report` summarises it (offline: it does not need LiteLLM installed). |
| capability gate | `capability_gate.loci_capability_gate` | Adds `!<tag>` request tags to requests that carry tools or images, so the router skips deployments that lack the capability. Does nothing until you configure the tags (see below). |
| governor | `governor.loci_governor` | Spend guard: reserves a worst-case cost before each call against day, rolling-24h, hourly and per-client caps, with STOP/PAUSE files and a hash-chained audit log. Observe-only unless the owner envelope turns enforcement on (see "Governor" below). |

`loci_litellm.__version__` is read from `pyproject.toml` in a source checkout, else from installed metadata. It and
the package path are logged once at INFO on import.

## Requirements and install

- Python 3.11+, `httpx`, `pyyaml`, and a LiteLLM that provides `CustomLogger`. The `x-loci-grounding` response header
  needs LiteLLM to call `async_post_call_response_headers_hook`; on a LiteLLM that does not, grounding still runs
  and the header is not sent. Other LiteLLM versions have not been verified.
- Install with `pip install ./litellm-loci`, or put the directory on `PYTHONPATH`.
- Register callbacks in the proxy `config.yaml`:

```yaml
litellm_settings:
  callbacks:
    - loci_litellm.grounding_hook.loci_grounding_hook
    - loci_litellm.memory_writer.loci_memory_writer
    - loci_litellm.circuit_monitor.loci_circuit_monitor
    - loci_litellm.cost_ledger.loci_cost_ledger
    - loci_litellm.capability_gate.loci_capability_gate
```

## What happens when something fails

No callback raises into the request. Failures surface as follows.

| Situation | Request | Where you see it |
|---|---|---|
| Grounding: no hit above the thresholds | continues, no context | status `miss`, reason `no_matching_context`; notice injected; WARNING |
| Grounding: no hit, and some collection searches failed | continues, no context | status `miss`, reason `no_matching_context_some_searches_failed` (part of the store was never searched); notice injected; WARNING with `N of M searches failed` |
| Grounding: Qdrant or embedder unconfigured, unreachable, or every collection search failed | continues, no context | status `unavailable` + reason (`qdrant_unconfigured`, `embed_unconfigured`, `embedding_failed`, `all_collections_failed`); notice injected; WARNING |
| Grounding: some collections failed, others returned hits | continues with the hits | WARNING per failed collection; block header says `N searches failed` |
| Grounding: unexpected exception | continues, no context | status `error` + exception class; notice injected; WARNING |
| Grounding skipped (disabled, internal call, unsupported call type, no user text, slash command, client or earlier hop already grounded) | unchanged | `x-loci-grounding: skipped` (or `client_grounded` / `already_grounded`) with `x-loci-grounding-reason` |
| Memory write fails (no Qdrant URL, no embed endpoint, embed or upsert error) | unaffected | WARNING with reason (first per reason, then every 100th); `loci_memory_writer.failure_counts` |
| Cost ledger cannot write | unaffected | WARNING (first, then every 100th) |
| Embedder failing repeatedly | continues without context; memory write skipped | breaker opens: one WARNING, then grounding `unavailable` with reason `embedder_circuit_open` and counted writer skips until a probe succeeds |

So with nothing configured the grounding hook is not a no-op: every request of a grounded call type gets a
`LOCI_GROUNDING_STATUS: unavailable` system notice and a WARNING. Set `LOCI_GROUNDING_DISABLED=1` to turn it off.

Behaviour to be aware of:

- Retrieval writes. Each returned `loci_memory` point gets its `pheromone` field bumped (read-modify-write, not
  atomic, not capped) unless `LOCI_PHERO_DEPOSIT_DISABLED=1`. The deposits run concurrently and add up to one 0.5 s
  timeout to the request.
- The memory writer stores the first 300 characters of the prompt and the first `LOCI_MEM_MAX_CHARS` of the answer
  with no redaction, labelled `record_type=inferred, confidence=low, provenance=model_output`. Writes use
  `wait=false`, so success is not confirmed. Grounding searches `loci_memory`, so what one caller sent can be
  injected into another caller's request; collections are not filtered per tenant or key.
- The `loci_internal_call` marker (`metadata.loci_internal_call: true`) exempts a request from grounding and from
  write-back. Any authenticated caller can set it.
- A client marker in `LOCI_CLIENT_RAG_OK_MARKERS`, in any message role, makes grounding skip the request. Nothing
  verifies the client's claim. Unset by default.
- `LOCI_INVESTIGATION_ID` is process-wide: it boosts matching records in retrieval and is stamped on every record
  the writer stores, whoever made the request.
- Streams are written once from the assembled response LiteLLM passes to the success event; a stream LiteLLM cannot
  assemble is not recorded.
- Circuit state and failure counts are per process and in memory.
- Numeric variables read at import: in `grounding_hook` and `circuit_monitor` a non-numeric value makes the import fail
  (the proxy reports it at startup). `memory_writer` and `cost_report` fall back to the default with a WARNING.
- The delivery audit reports "context LOST" for provider body shapes it does not recognise.

## Environment variables

Read once at import (restart to change) unless marked "per call". Defaults are the values in the code.

| Variable | Default | Effect |
|---|---|---|
| `LOCI_CONFIG` | `~/.loci/backends.toml` | Path of the private TOML config (sections below). Read once. An unreadable file gives an empty config and a WARNING. |
| `LOCI_EMBED_API_BASE`, `MNEMOSYNE_EMBEDDING_API_URL` | `[embed] api_base`, else unset | Base URL of any OpenAI-compatible embeddings API (OpenAI, a LiteLLM proxy, vLLM, TEI, Ollama's `/v1`, a provider gateway). Requests go to `<base><LOCI_EMBED_API_PATH>`. Unset makes grounding `unavailable` (`embed_unconfigured`) and the writer warn. Per call. |
| `LOCI_EMBED_MODEL`, `MNEMOSYNE_EMBEDDING_MODEL` | `[embed] model`, else unset | Embedding model name sent in the request body. Required: there is no default. Per call. |
| `LOCI_EMBED_API_KEY`, `EMBED_API_KEY` | `[embed] api_key`, else empty | Key for the embeddings endpoint, sent as `<LOCI_EMBED_AUTH_HEADER>: <LOCI_EMBED_AUTH_SCHEME> <key>`. Per call. |
| `LOCI_EMBED_API_PATH` | `[embed] api_path`, else `/embeddings` | Appended to the base. Set it to `""` when the base is already the full URL, or to a path with a query string. Per call. |
| `LOCI_EMBED_AUTH_HEADER` | `[embed] auth_header`, else `Authorization` | Header that carries the key (for example `api-key`, `x-api-key`). Per call. |
| `LOCI_EMBED_AUTH_SCHEME` | `[embed] auth_scheme`, else `Bearer` | Prefix before the key; an empty string sends the raw key. Per call. |
| `LOCI_EMBED_EXTRA_HEADERS` | `[embed.extra_headers]`, else none | JSON object of additional request headers. Invalid JSON is ignored with a WARNING. Per call. |
| `LOCI_EMBED_EXTRA_BODY` | `[embed.extra_body]`, else none | JSON object merged into the request body (for example `{"input_type": "search_query", "encoding_format": "float"}`). `model` and `input` always win. Per call. |
| `LOCI_EMBED_DIMENSIONS` | `[embed] dimensions`, else unset | Integer sent as `dimensions`. Per call. |
| `LOCI_EMBED_QUERY_PREFIX`, `LOCI_EMBED_DOCUMENT_PREFIX` | `[embed] query_prefix` / `document_prefix`, else empty | Text placed before a retrieval query and before a stored record, for models that expect task prefixes (`search_query: `, `query: `, `passage: `). Per call. |
| `LOCI_QDRANT_URL`, `QDRANT_URL` | `[qdrant] url`, else empty | Qdrant endpoint. Empty makes grounding `unavailable` and the writer warn. Per call. |
| `LOCI_QDRANT_API_KEY`, `QDRANT_API_KEY` | `[qdrant] api_key`, else empty | Qdrant API key. Per call. |
| `LOCI_GROUNDING_DISABLED` | unset | `1`/`true`/`yes`/`on` skips grounding (status `skipped`). Per call. |
| `LOCI_GROUNDING_EXTRA_COLLECTIONS` | empty | Comma-separated collections searched in addition to `mnemosyne`, `loci_sessions`, `loci_memory`. Read from payload field `text` with the named vector, except `ecc_skills` and `agent_core_chunks`, which have built-in mappings. |
| `LOCI_RECALL_TOP_K` | `3` | Records injected. |
| `LOCI_RECALL_MIN_SCORE` | `0.55` | Minimum Qdrant similarity score. |
| `LOCI_RECALL_MIN_IMPORTANCE` | `0.2` | Minimum importance (an explicit 0 stays 0 and is filtered). |
| `LOCI_RECALL_VECTOR_NAME` | `dense` | Named vector searched. |
| `LOCI_RECALL_MAX_CHARS` | `200` | Characters of each record shown. |
| `LOCI_EMBED_TIMEOUT` | `3.0` | Seconds for the embedding request. |
| `LOCI_EMBED_BREAKER_THRESHOLD` | `3` | Consecutive embedding failures that open the embedder circuit breaker (shared by grounding and the memory writer; per process, per endpoint). While open, grounding reports reason `embedder_circuit_open` and the writer skips embedding, without sending requests. Read when the breaker is created. |
| `LOCI_EMBED_BREAKER_COOLDOWN_S` | `15` | First cooldown in seconds. After it one probe request is allowed; a failed probe reopens with a doubled cooldown. |
| `LOCI_EMBED_BREAKER_MAX_COOLDOWN_S` | `300` | Cooldown cap in seconds. |
| `LOCI_EMBED_BREAKER_DISABLED` | unset | `1` never opens the breaker. |
| `LOCI_QDRANT_TIMEOUT` | `2.0` | Seconds per Qdrant search. |
| `LOCI_INVESTIGATION_BOOST` | `1.5` | Relevance multiplier for records whose investigation matches `LOCI_INVESTIGATION_ID`. |
| `LOCI_INVESTIGATION_ID` | empty | Per call. See above; also stamped on written records. |
| `LOCI_RANKER_W_RELEVANCE`, `_RECENCY`, `_TRUST`, `_TYPE` | `0.50`, `0.20`, `0.15`, `0.15` | Weights of a weighted sum. A WARNING is logged at import if they do not sum to 1.0. |
| `LOCI_RECENCY_HALFLIFE_DAYS` | `7` | Recency decay half-life. |
| `LOCI_MMR_LAMBDA` | `0.75` | MMR relevance/diversity trade-off. |
| `LOCI_RECALL_EXPLORE_EPSILON` | `0.05` | Probability that the last slot is a random remaining hit, so retrieval is not deterministic by default. `0` makes it deterministic. |
| `LOCI_PHERO_BETA` | `0.08` | Weight of the logarithmic pheromone term in the score. |
| `LOCI_PHERO_HALFLIFE_H` | `24` | Pheromone decay half-life, hours. |
| `LOCI_PHERO_DEPOSIT` | `1.0` | Amount added to a returned record's pheromone. |
| `LOCI_PHERO_DEPOSIT_DISABLED` | unset | `1` makes grounding read-only. Per call. |
| `LOCI_CLIENT_RAG_OK_MARKERS` | empty (or `[grounding] client_rag_ok_markers`) | `\|\|`-separated strings; a request containing one is not grounded. Per call. |
| `LOCI_TRUST_CLIENT_RAG` | `1` | `0` ignores those markers. Per call. |
| `LOCI_HONOR_INTERNAL_CALL` | `1` | `0` treats `loci_internal_call` requests as ordinary traffic. Per call. |
| `LOCI_GROUNDING_AUDIT_FILE` | unset | Append a metadata-only delivered/lost row per grounded request. Unset: log line only. Per call. |
| `LOCI_GROUNDING_AUDIT_MAX_BYTES` | `10000000` | At this size the audit file is renamed to `<file>.1` (replacing an older one) and a new one starts. `0` disables rotation. Per call. |
| `LOCI_MEMORY_WRITER_DISABLED` | unset | `1` stops the memory writer. Per call. |
| `LOCI_MEM_REDACT_DISABLED` | unset | `1` stores and embeds prompt and answer text as received. By default common credential shapes (API keys, bearer tokens, cloud and VCS tokens, JWTs, private keys, `key=value` secrets) are replaced with `[REDACTED...]` first. Best effort, not a data-loss-prevention system. Per call. |
| `LOCI_MEMORY_WRITE_COLLECTION` | `[memory] write_collection`, else `loci_memory` | Collection written to. Grounding does not search a non-default write collection unless it is also in `LOCI_GROUNDING_EXTRA_COLLECTIONS`. Per call. |
| `LOCI_MEMORY_VECTOR_NAME` | `dense` | Named vector the writer sets. Per call. |
| `LOCI_MEM_MIN_PROMPT_LEN` | `20` | Prompts shorter than this are not written. Per call. |
| `LOCI_MEM_MAX_CHARS` | `500` | Answer characters stored. Per call. |
| `LOCI_MEM_EMBED_TIMEOUT` | `3.0` | Seconds for the writer's embedding request. Per call. |
| `LOCI_MEM_QDRANT_TIMEOUT` | `2.0` | Seconds for the writer's upsert. Per call. |
| `LOCI_CB_FAILURE_THRESHOLD` | `5` | Consecutive transient failures before a model group's circuit opens (no time window). |
| `LOCI_COST_LEDGER_FILE` | `[cost] ledger_file`, else `~/.loci/litellm_cost_ledger.jsonl` | Ledger path. Rows hold `agent` and `key_alias`; treat as sensitive. Per call. |
| `LOCI_COST_LEDGER_MAX_BYTES` | `50000000` | At this size the ledger is renamed to `<file>.1` (replacing an older one) and a new one starts. `0` disables rotation. Per call. |
| `LOCI_COST_LEDGER_DISABLED` | unset | `1` stops the ledger once registered. Per call. |
| `LOCI_CAPABILITY_GATE_DISABLED` | unset | `1` (or `true`/`yes`/`on`) stops the gate from tagging requests. Per call. |
| `LOCI_GATE_TOOLS_EXCLUDE_TAGS` | empty | Comma-separated deployment tags to exclude (`!<tag>`) for requests with tool definitions or tool turns. Empty: tools are not gated. Per call. |
| `LOCI_GATE_IMAGES_EXCLUDE_TAGS` | empty | Same for requests with image content parts. Per call. |
| `LOCI_LOCAL_HOSTS` | empty | Comma-separated hostnames counted as `local` in the ledger. Without it only loopback, private, link-local and `100.64.0.0/10` addresses and `localhost` are local; any other hostname counts as cloud. Per call. |
| `LOCI_LOCAL_PROVIDERS` | empty | Comma-separated LiteLLM provider prefixes (for example `ollama,vllm`) whose deployments count as `local` in the ledger whatever their host name. Per call. |
| `LOCI_LOCAL_SLO_SUCCESS`, `_P95_S`, `_MIN_CALLS` | `0.98`, `20`, `50` | Thresholds used by `cost_report`. |

Config file sections (all optional): `[embed]` (`api_base`, `model`, `api_key`, `api_path`, `auth_header`, `auth_scheme`,
`dimensions`, `query_prefix`, `document_prefix`, and tables `extra_headers`, `extra_body`), `[qdrant] url` and `api_key`,
`[memory] write_collection`, `[cost] ledger_file`, `[grounding] client_rag_ok_markers`. The file lives outside any
repository and is not managed by git; it may hold an API key.

## Deployment invariants

`deploy/config.example.yaml` is a reference config. `deploy/check_deployment.py` checks a real
`config.yaml` (and optionally a `.env`) against the generic proxy rules below. It does not check that the Loci
callbacks are registered or that any `LOCI_*` setting exists. Exit status 0 means no violations. The tests in
`tests/test_deploy_check.py` run it against the example and a set of mutations.

```
python deploy/check_deployment.py --config config.yaml --env .env
```

Run it with the proxy's interpreter and import path (so LiteLLM's cost map resolves). Without `--env` the
environment half of `no-ip-literal` is not evaluated and the output says so.

| Rule | What is enforced |
|---|---|
| `embedding-group` | A model group named `embedding` (`--embedding-group`) exists; it is not in `router_settings.fallbacks`, `context_window_fallbacks` or `content_policy_fallbacks`; it is not a value in a deployment's `complexity_router_config.tiers`. Not checked: `litellm_settings.fallbacks`, `default_fallbacks`, aliases pointing at it. |
| `priced` | Every deployment has `input_cost_per_token` and `output_cost_per_token`, or LiteLLM's cost map knows the model. Exempt: the embedding group, auto-router deployments, and deployments whose `api_base` is `os.environ/NAME` with `NAME` beginning with a `--local-env-prefix` (default `LOCAL`, `OLLAMA`, `VLLM`, `LLAMA`). The exemption goes by that variable name only. |
| `no-ip-literal` | Skipped with `--allow-ip-literals`. No literal `api_base` is an IP address. With `--env`: no variable named `*_URL`, `*_BASE`, `*_BASE_URL`, `*_ENDPOINT` or `*_HOST` (names containing `BIND` or `LISTEN` excepted), and no variable used as `api_base` through `os.environ/NAME`, holds an IP address. |
| `alias-targets` | Every `model_group_alias` target is a model group. |
| `trusted-proxies` | Warning only: `general_settings.trusted_proxy_ranges` is set. |

Notes:

- The plugin takes its embedding endpoint only from configuration (see the `LOCI_EMBED_*` variables); it does not
  probe for a local server and has no default model.
- The auth-failure path treats a missing `prisma` package as "not a database error", so a missing key is a 401
  rather than a 500 on a proxy without a database.
- Invented model names are deliberately not aliased to `embedding`, so an unknown model fails loudly.

## Tests

Run from this directory with `HOME` and `LOCI_CONFIG` pointing at an empty temporary location so the tests do not
read a real config:

```
python -m pytest -q
```

## Capability gate

Some deployments cannot serve some requests: a small local model may loop on tool results, or a text-only model
cannot take images. The gate expresses that as router tags instead of hardcoding which deployments those are.

1. Give the deployments that lack a capability a tag in `config.yaml` (`litellm_params.tags: ["local"]`).
2. Turn on `router_settings.enable_tag_filtering: true`.
3. Set `LOCI_GATE_TOOLS_EXCLUDE_TAGS` and/or `LOCI_GATE_IMAGES_EXCLUDE_TAGS` to those tags.

A request with `tools`/`functions`, a `tool`/`function` role message or `tool_calls` then gets `!<tag>` for every tag in
the tools list; a request with an `image_url`, `input_image` or `image` content part gets the images list. The gate adds
to `metadata.tags` and sets `metadata.loci_capability_gate` to the capabilities that matched. It never changes the model
or the request body, and an internal error is logged at WARNING and ignored. It writes only to `metadata`; whether
routes that carry `litellm_metadata` instead (for example Anthropic `/v1/messages`) honour the tags has not been
verified.

## Governor

`governor.loci_governor` is a spend guard. It is **observe-only by default**: it books spend and records what it
would have refused (`would_block` events in the log and the audit chain) but never blocks, downgrades or edits a
request. Registering it is therefore safe; nothing is read or written until the first request.

State lives in `LOCI_GOVERNOR_DIR` (default `~/.loci/governor`): `counter.json` (the Governor's own spend counter, under
4 KiB, written through a temp file and a rename, seeded from the cost ledger and its `.1` and never lowered by a ledger
rotation), `STOP`, `PAUSE`, `envelope.toml`, `envelope.pin` and `audit/`.

| Variable | Meaning |
|---|---|
| `LOCI_GOVERNOR` | `observe` (default; anything other than `enforce` is observe) or `enforce`. `enforce` takes effect only if the owner envelope is present, pinned and has `[governor] enforce = true`. |
| `LOCI_GOVERNOR_DIR` | State directory (above). |
| `LOCI_GOVERNOR_LEDGER` | Cost ledger used to seed the counter (default: the cost ledger's own path). |
| `LOCI_GOVERNOR_OWNER_KEY_FILE` | File with the owner's HMAC key; a matching signed `envelope_pin` entry in the audit chain accepts an envelope whose hash differs from the pin file. |

The envelope (`envelope.toml`, or `.json`) holds the caps, the enforce flag, the day timezone and the client table.
Its sha256 is compared with the pin on every re-check (60 s); one changed byte, a missing, unpinned or malformed file
means the built-in defaults, level `L3` and an alarm (`caps_hash_mismatch`, `envelope_missing`, `envelope_unpinned`,
`envelope_invalid`). Which accounts may write the files is an operating-system matter; the module only detects. Built-in
caps (dollars): local-day soft 30 and hard 50, rolling 24 h 60, hourly 10 (downgrade) and 25 (reject non-interactive),
interactive absolute stop 75. The day is the machine's local calendar day (DST-aware) unless `day_timezone` names a zone.

With enforcement on, a tool-carrying request over a hard cap gets HTTP 429 with `reason` and `cap`; a tool-free one goes
to the cheapest adequate arm if that fits under the cap; interactive callers are downgraded, never rejected, until the
absolute stop. The spend path fails closed (503 `budget_gate_error`); only optimisation steps fail open, counted in
`stats()`, and more than five such events in 60 s make the Governor fail closed (`governor_degraded`). `STOP` answers 503
`stopped` (health and ZERO-tier calls pass); `PAUSE` only switches off learning and exploration. Remove the callback or set
`LOCI_GOVERNOR=observe` to roll back.

## License

MIT. See `LICENSE`.
