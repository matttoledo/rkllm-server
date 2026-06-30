# rkllm-server

Minimal HTTP wrapper over the Rockchip [`rknn-llm`](https://github.com/airockchip/rknn-llm)
runtime. Exposes a quantized `.rkllm` model (Qwen, Llama, Phi, Gemma, …) running
on the **RK3588 / RK3576 NPU** via two HTTP endpoints, so any service on your LAN
or Docker cluster can talk to the NPU without linking against `librkllmrt` directly.

> **Targets:** Orange Pi 5 / 5 Plus / 5 Pro, Radxa ROCK 5B / 5C, NanoPi R6S, and other
> RK3588/RK3576 SBCs. Tested with Ubuntu 24.04 + RKNPU driver 0.9.7 + rknn-llm v1.2.1.

## Why

The bundled C demos in `airockchip/rknn-llm` are interactive (read from stdin)
and assume a single-process workflow. [`RKLLama`](https://github.com/NotPunchnox/rkllama)
gives you an Ollama-compatible REST API but is heavier and has been stagnant since
March 2025.

This server is in between: ~400 lines of C++, two endpoints, single binary, single
systemd unit. Built for production-leaning hobby setups (home automation, smart
shop assistants, on-device chatbots) where you want the NPU as a quiet HTTP
service that just works.

## Highlights

- **Single binary**, no Python deps. ~400 LoC C++17 + [cpp-httplib](https://github.com/yhirose/cpp-httplib)
  + [nlohmann/json](https://github.com/nlohmann/json) (header-only, vendored).
- **Two endpoints**: opinionated `/process` (with adaptive recovery) +
  generic `/generate` (raw passthrough).
- **Adaptive recovery** for the [rknn-llm v1.2.1 KV-cache state issue](docs/ADAPTIVE_RECOVERY.md):
  invalid output triggers a handle reset + retry. In the wild, we saw shared
  handles silently corrupt after a few turns; this brings success rate from
  20% → 100% on the same prompt set.
- **Parametrizable system prompt** via `--prompt-file` — ship your own task
  without rebuilding.
- **systemd unit example** for `Restart=on-failure` resilience.

## Benchmarks

Qwen2.5-1.5B w8a8 on Orange Pi 5 (RK3588, 6 TOPS NPU), 1.6 GB peak RAM:

| Stage | Throughput |
|---|---|
| Decode | **14.34 tok/s** (avg over 8 prompts) |
| Prefill | 244 tok/s (system + user, ~270 tokens) |
| TTFT (warm) | **180 ms** |
| Init / cold start | 2.3 s |

For reference, the same model on the CPU (Ollama Q4_K_M, 8× Cortex-A76 + A55)
gives ~7 tok/s decode — so NPU is ~2× faster *and* leaves the CPU free.

## Build

You need the rknn-llm runtime headers + shared library:

```bash
# 1. Get the runtime (pin v1.2.1 — v1.3.0 had a SIGSEGV on RK3588)
git clone --depth 1 --branch release-v1.2.1 https://github.com/airockchip/rknn-llm.git ~/rknn-llm

# 2. Get a converted model (or convert your own via rkllm-toolkit)
mkdir -p ~/models && cd ~/models
wget https://huggingface.co/Azurastar2903/Qwen2.5-1.5B-Instruct-rk3588-1.2.1/resolve/main/Qwen2.5-1.5B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm \
    -O Qwen2.5-1.5B-Instruct.rkllm

# 3. Build (native aarch64, on the SBC itself)
git clone https://github.com/matttoledo/rkllm-server.git
cd rkllm-server
RKLLM_RUNTIME_DIR=~/rknn-llm/rkllm-runtime/Linux/librkllm_api ./build.sh
```

Output: `build/rkllm_server`.

## Run

```bash
# With the bundled vidraçaria (Portuguese glass-shop) example prompt
./build/rkllm_server ~/models/Qwen2.5-1.5B-Instruct.rkllm

# With your own prompt
./build/rkllm_server ~/models/Qwen2.5-1.5B-Instruct.rkllm \
    --prompt-file examples/prompts/customer_support.txt \
    --port 18080
```

For best performance, lock NPU + CPU frequencies before starting (the
`rknn-llm` repo ships `scripts/fix_freq_rk3588.sh` — needs root).

## API

### `GET /health`

```bash
curl http://localhost:18080/health
```

```json
{
  "status": "ok",
  "uptime_s": 1234,
  "model_path": "/home/user/models/Qwen2.5-1.5B-Instruct.rkllm",
  "decode_tps_avg": 14.34,
  "inferences_completed": 87
}
```

### `POST /generate` — generic passthrough (recommended for custom prompts)

Caller provides the system prompt + user message. Server returns raw output.
No JSON validation. Chat template is restored after each call so a parallel
`/process` keeps working.

```bash
curl -X POST http://localhost:18080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a sentiment classifier. Respond with one of: positive | negative | neutral.",
    "user_text": "this is the best thing I bought all year"
  }'
```

```json
{
  "ok": true,
  "raw_output": "positive",
  "total_latency_ms": 412,
  "decode_tps": 13.9,
  "prefill_tps": 218.3,
  "decode_tokens": 2,
  "prefill_tokens": 38
}
```

### `POST /process` — opinionated extractor (uses the configured system prompt)

Designed for structured extraction with JSON output. Includes **adaptive recovery**:
if the model returns invalid JSON, the handle is destroyed + reinitialized + the
request is retried once. The default validator expects the bundled vidraçaria
schema (`{intent, product?}`).

```bash
curl -X POST http://localhost:18080/process \
  -H "Content-Type: application/json" \
  -d '{"text":"vidro temperado 1,20x0,80"}'
```

```json
{
  "intent": "quote_request",
  "product": {
    "type": "FIXO",
    "width_cm": 120,
    "height_cm": 80,
    "sheets": 1,
    "color": "INCOLOR"
  },
  "total_latency_ms": 3204,
  "decode_tps": 14.3,
  "reinit_used": false,
  "raw_output": "..."
}
```

If you change the prompt with `--prompt-file`, use `/generate` instead —
`/process` will keep trying to parse the vidraçaria schema and re-initing on every
request. A configurable JSON-Schema validator is on the roadmap.

## systemd

```bash
sudo cp systemd/rkllm-server.service.example /etc/systemd/system/rkllm-server.service
# edit User= / paths
sudo systemctl daemon-reload
sudo systemctl enable --now rkllm-server
journalctl -u rkllm-server -f
```

## Network access from Docker containers

If you run other services in Docker that need to call the NPU, the host firewall
(UFW typically) blocks the overlay subnets by default. Allow them:

```bash
sudo ufw allow from 10.0.0.0/8 to any port 18080 proto tcp comment 'rkllm-server from docker overlays'
sudo ufw allow from 172.16.0.0/12 to any port 18080 proto tcp comment 'rkllm-server from docker bridges'
```

Inside a Docker Swarm service, reach it via `host.docker.internal:18080` after
adding `extra_hosts: ["host.docker.internal:host-gateway"]` to your service spec.

## What's NOT in this server

- **No streaming** — responses are buffered until generation finishes. The rknn-llm
  callback emits tokens incrementally; SSE/WebSocket support is on the roadmap if
  there's demand.
- **No multi-instance / multi-model** — one model per process. Rockchip's runtime is
  single-handle; run multiple processes on different ports if you need it.
- **No conversation memory** — `/process` and `/generate` are stateless per-request.
  Track conversation history caller-side and replay it in `user_text`.
- **No auth** — bind to LAN only or put a reverse proxy (Nginx/NPM/Caddy) in front.
- **No model download** — bring your own `.rkllm` file.

## Known limitations

- **Pin rknn-llm v1.2.1.** v1.3.0 has a [SIGSEGV on RK3588](https://github.com/airockchip/rknn-llm/issues/509)
  during `rkllm_run`. The CMakeLists looks for headers compatible with v1.2.1.
- **NPU is single-threaded.** Concurrent requests are serialized via mutex. For a
  vidraçaria-volume workload (~dozens of msgs/day) this is fine; for higher
  throughput, run multiple instances behind a load balancer.
- **w8a8 quant occasionally hallucinates small numbers.** "60x40" can become
  "600x400" if your prompt doesn't have a negative example. See
  `examples/prompts/vidracaria.txt` for the workaround.

## Comparison

| | rkllm-server | [RKLLama](https://github.com/NotPunchnox/rkllama) | Raw `llm_demo` from airockchip |
|---|---|---|---|
| Lines of code | ~400 C++ | ~5k Python | ~200 C++ |
| Footprint | single binary + .so | Python venv + deps | single binary |
| API style | Plain HTTP / JSON | Ollama-compatible REST | stdin (interactive) |
| Streaming | ❌ (planned) | ✅ | ✅ |
| Multi-model | ❌ | ✅ | ❌ |
| Function calling | ❌ (use prompt + parse) | ✅ Qwen native | ❌ |
| Adaptive recovery | ✅ | partial | ❌ |
| Last release | active | March 2025 | ongoing |

Pick `rkllm-server` if you want the NPU as a dumb HTTP service. Pick RKLLama if
you want OpenAI/Ollama compatibility and native function-calling support.

## Acknowledgments

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — the runtime this
  whole project depends on.
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) — the HTTP server
  (MIT, vendored in `third_party/`).
- [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing (MIT, vendored).

## License

Apache-2.0 — same as upstream `airockchip/rknn-llm`. See [LICENSE](LICENSE).
