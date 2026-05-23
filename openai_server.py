import modal

MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("vllm-openai-server")

# Qwen3-1.7B is small, fast, and works well on L4
MODEL_NAME = "Qwen/Qwen3-1.7B"
VLLM_PORT = 8000

# Cache volumes for faster cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="L4",  # ~$0.80/hr, better than T4 for inference
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES, requires_proxy_auth=True)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "4096",
    ]

    subprocess.Popen(cmd)


@app.local_entrypoint()
async def main():
    import urllib.request
    import json

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    # Test the server
    data = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(result["choices"][0]["message"]["content"])
