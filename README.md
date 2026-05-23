# Bender AI

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.13-blueviolet)](https://github.com/vllm-project/vllm)
[![Modal](https://img.shields.io/badge/Modal-Serverless_GPU-00C853)](https://modal.com)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI_Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

Production-ready serverless LLM inference API with OpenAI-compatible endpoints, deployed on GPU infrastructure with automatic scaling.

## Features

- **Serverless GPU Compute** — Auto-scales from zero, pay only for inference time
- **High-Performance Inference** — vLLM engine with PagedAttention for optimized throughput
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI client libraries
- **100 Concurrent Requests** — Handles high throughput with request batching
- **Secure by Default** — Token-based proxy authentication

## Quick Start

```bash
pip install modal
modal setup
modal deploy openai_server.py
```

## Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-workspace>--vllm-openai-server-serve.modal.run/v1",
    api_key="unused",
    default_headers={"Modal-Key": "<token-id>", "Modal-Secret": "<token-secret>"},
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "Explain transformers in one sentence."}],
)
```

## Architecture

| Component | Choice |
|-----------|--------|
| Inference Engine | vLLM with continuous batching |
| GPU | NVIDIA L4 (24GB VRAM) |
| Model | Qwen3-1.7B |
| Infrastructure | Modal serverless |
| Auth | Modal proxy tokens |
