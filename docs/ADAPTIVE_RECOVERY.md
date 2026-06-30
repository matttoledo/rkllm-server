# Adaptive recovery — why `/process` re-inits on bad output

## TL;DR

`rknn-llm` v1.2.1 has a KV-cache state issue: when the same handle processes
multiple requests with `keep_history=0`, the cache state silently degrades
after a few turns. Subsequent requests start producing empty output or
malformed JSON, even with the same system prompt.

`/process` works around this by **validating the output and re-initializing
the handle if it's broken**. The cost: one ~2-second re-init every few
requests on the bad path. The gain: 100% success rate vs ~20% with a naive
shared-handle implementation.

This is implemented in `process_with_recovery()` in `src/main.cpp`.

## How we discovered it

Running a single Qwen2.5-1.5B handle across 5 structured-extraction prompts
with `keep_history=0` and `rkllm_clear_kv_cache` between calls, we saw:

| Variant | Success | Avg latency |
|---|---|---|
| v1: shared handle + `clear_kv_cache` between calls | 1/5 (20%) | 3.4s |
| v2: re-init handle per request | 5/5 (100%) | 6.3s |
| v3: shared handle, no clear, just `keep_history=0` | 3/5 (60%) | 2.6s |
| **v4: adaptive — re-init only on invalid output** | **5/5 (100%)** | **3.94s** |

The KV cache state has a "drift" that builds up regardless of whether you
call `rkllm_clear_kv_cache` or rely on `keep_history=0`. The exact trigger
isn't documented; we suspect attention masks accumulating across runs.

## How `/process` handles it

```cpp
bool process_with_recovery(const std::string& user_text, ProcessOutcome& outcome) {
    InferResult ir;
    run_inference(user_text, ir);

    if (output_is_valid_json(ir.raw_output)) {
        outcome = parsed_result;
        return true;
    }

    // Bad output — destroy + re-init the handle, retry once.
    destroy_model();
    init_model(g_model_path);

    InferResult ir2;
    run_inference(user_text, ir2);

    if (output_is_valid_json(ir2.raw_output)) {
        outcome = parsed_result;
        outcome.reinit_used = true;
        return true;
    }

    return false;
}
```

A single retry is enough — we've never seen a second consecutive failure
after a re-init.

## What "valid" means today

`/process` validates the output is:
- A JSON object (not an array, primitive, or non-JSON text)
- Has an `intent` field that is a non-empty string

This is intentionally permissive so the endpoint stays useful when you
swap the system prompt via `--prompt-file`. For stricter checks (enum-bound
`intent`, required nested fields, numeric ranges), fork
`is_valid_intent_response()` in `src/main.cpp`. A configurable JSON-Schema
validator is on the roadmap.

If your task produces an output shape that doesn't include an `intent`
field at all — use `/generate` instead. It has no validation and no recovery,
just raw passthrough.

## When NOT to use `/process`

- Your prompt produces output that doesn't have an `intent` field at all.
  Use `/generate`.
- You want raw passthrough (no JSON parsing). Use `/generate`.
- Your task is creative generation (chat, summarization) — no expected
  structure to validate against. Use `/generate`.

## Cost analysis

For the structured-extraction workload we benchmarked:
- 80% of requests succeed first try (~3.0–3.8s)
- 20% trigger re-init + retry (~6.3s)
- Weighted average: **3.94s/req**

If your prompt is well-tuned (good few-shot, clear schema), the bad-path rate
drops below 5%. If it's not tuned, you'll see frequent re-inits — fix the
prompt before fixing the recovery threshold.

## Future work

- **Configurable JSON Schema validator** so `/process` enforces your domain
  shape, not just the loose `{intent: string}` check.
- **Optional max-retries** flag (currently always 1).
- **Telemetry** — emit a counter per re-init so you can monitor prompt quality
  over time.
- **Streaming with recovery** — currently /process buffers the full output before
  validating. With SSE we'd need to either buffer-then-stream or commit to
  best-effort streaming without recovery.
