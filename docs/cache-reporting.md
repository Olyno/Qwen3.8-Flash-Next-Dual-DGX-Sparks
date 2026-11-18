# Per-request cache reporting

The launcher enables `--enable-prompt-tokens-details` so OpenAI-compatible
clients can read `usage.prompt_tokens_details.cached_tokens` in Chat
Completions responses. For streaming requests, send
`"stream_options": {"include_usage": true}` and read the final usage chunk.
A cold request may report zero; enabling reporting does not guarantee a hit.

This is response metadata only: it does not enable or retune prefix caching,
change cache retention, or change the KV pool. Prometheus cache metrics at
`/metrics` are separate and do not depend on this flag.

Existing servers only pick up this launcher change when the operator next
recreates/restarts them. The accompanying CPU-only regression checks the
launch arguments; it does not start a server or prove live cache reuse.

## Readiness and failure logs

When the v030 launcher is configured with PLE offload, a healthy `/health`
response is not sufficient: the container log must also contain a PLE table
engagement marker. The launcher drains the complete log before deciding, so a
marker near the start of a large log cannot cause a false failure from a
broken pipe. A missing marker or a failed `docker logs` command still fails
the launch, even if a partial log already contained a marker.

On launch failure, the full container log is archived under `logs/` before
the original failure is reported. Error-line excerpts are best effort: a
benign log with no matching error lines does not interrupt diagnostics or
prevent the worker log from being archived on dual-node launches.

CPU-only regressions in `tests/test_launch_readiness.py` use fake Docker and
health responses to cover large logs, missing markers, log retrieval failures,
PLE gating, and diagnostic preservation. They do not start a server or prove
live PLE engagement.

## Loader page-cache eviction

`load_drop_cache: true` mounts a patched `weight_utils.py`. The safetensors
iterator advises `DONTNEED` after each consumed shard and sweeps checkpoint
shards and sidecars at load end. `runai_streamer` uses a separate iterator:
it now runs the same end sweep after the streamer's context releases its
readers. There is no per-shard RunAI hook because that tensor stream does not
publish shard-completion boundaries. Previously RunAI bypassed both hooks.

This is advisory clean-page eviction, not a hard loading-time memory bound,
and it does not release live tensors or streamer-owned buffers. Keep the
memory watchdog active during loading. The threaded loader is not covered.
CPU regressions execute the patched iterator from an extracted upstream
source, checking tensor delivery, gate-off behavior, and eviction only after
streamer release. They do not certify GPU loading or memory headroom.
