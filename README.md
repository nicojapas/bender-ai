# Bender AI

[![CI](https://github.com/nicojapas/bender-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/nicojapas/bender-ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.31-blueviolet)](https://github.com/vllm-project/vllm)
[![Modal](https://img.shields.io/badge/Modal-1.6-00C853)](https://modal.com)

A configurable template for self-hosting an open LLM on [Modal](https://modal.com) with [vLLM](https://github.com/vllm-project/vllm). It exposes an OpenAI-compatible API, scales to zero when idle, and is capped so a traffic spike can't run up the bill.

Default model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) on one NVIDIA L4.

## Quick start

```bash
pip install -r requirements.txt
modal setup

modal run openai_server.py::download_model   # cache weights in a Volume, once per model/revision
modal deploy openai_server.py                # prints the server URL
```

Create a proxy auth token at [modal.com/settings/proxy-auth-tokens](https://modal.com/settings/proxy-auth-tokens), then copy `.env.example` to `.env` and fill it in.

Smoke test (starts a replica, waits for `/health`, sends one chat request):

```bash
modal run openai_server.py
```

## Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="<server-url>/v1",
    api_key="unused",
    default_headers={"Modal-Key": "wk-...", "Modal-Secret": "ws-..."},
    max_retries=10,
)

response = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Explain transformers in one sentence."}],
)
```

The model is served as both `llm` and its Hugging Face name, so you can swap models without changing clients.

**Cold starts.** With `MIN_CONTAINERS=0`, a request to an idle server returns `503` while a replica boots (about 1–2 minutes). Clients must retry; the OpenAI SDK does this with `max_retries`. Set `MIN_CONTAINERS=1` to keep one replica warm, at the cost of paying for an idle GPU.

## Configuration

Everything is set with environment variables at deploy time:

```bash
GPU=A10G MAX_CONTAINERS=4 MIN_CONTAINERS=1 modal deploy openai_server.py
```

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `bender-ai` | Modal app name (changes the URL) |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | Hugging Face model |
| `MODEL_REVISION` | pinned commit | Hugging Face commit, so the model can't change under you |
| `SERVED_MODEL_NAME` | `llm` | Model alias for clients |
| `GPU` | `L4` | Modal GPU type, e.g. `A10G`, `L40S`, `H100`, `H100:2` |
| `MAX_MODEL_LEN` | `8192` | Context length |
| `MIN_CONTAINERS` | `0` | Warm replicas (`0` = scale to zero) |
| `MAX_CONTAINERS` | `2` | Hard cap on replicas: your cost ceiling |
| `TARGET_CONCURRENCY` | `32` | Requests per replica before autoscaling adds one |
| `MAX_CONCURRENCY` | `64` | Hard per-replica limit; extra requests get `429` |
| `SCALEDOWN_MINUTES` | `5` | Idle time before a replica shuts down |
| `FAST_BOOT` | `true` | Skip CUDA graphs/compilation: faster cold start, slower generation |
| `ENABLE_THINKING` | `false` | Qwen3 thinking mode default (per request: `chat_template_kwargs`) |
| `REASONING_PARSER` | `qwen3` | Puts thinking in `reasoning_content`; empty to disable |
| `TOOL_CALL_PARSER` | `hermes` | Enables function calling; empty to disable |
| `EXTRA_VLLM_ARGS` | | Extra `vllm serve` flags, e.g. `--gpu-memory-utilization 0.85` |
| `PROXY_AUTH` | `true` | Require `Modal-Key`/`Modal-Secret` headers |
| `API_KEY_SECRET` | | Name of a Modal Secret holding `VLLM_API_KEY`, for `Authorization: Bearer` clients |

When changing `MODEL_NAME`, also set `MODEL_REVISION`, and check the model's reasoning and tool-call parsers in the [vLLM docs](https://docs.vllm.ai/).

### Auth

By default, Modal's proxy rejects any request without a valid proxy token. If a client can only send a standard API key:

```bash
modal secret create bender-api-key VLLM_API_KEY=sk-your-key
API_KEY_SECRET=bender-api-key PROXY_AUTH=false modal deploy openai_server.py
```

Keep `PROXY_AUTH=true` as well if you want both checks.

## Load testing

```bash
python load_test.py <server-url> --requests 200 --concurrency 32
```

Reports success rate, p50/p95 latency, time to first token and output tokens/s. Use it to tune `TARGET_CONCURRENCY` and `MAX_CONCURRENCY` for your model and GPU.

## Monitoring

- vLLM logs throughput and KV-cache usage every 10 seconds (visible with `modal app logs bender-ai`).
- Prometheus metrics are served at `/metrics`.
- Modal's dashboard shows replicas, GPU usage and cost.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```

CI runs the same checks on every push.
