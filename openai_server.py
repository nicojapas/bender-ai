import json
import os

import modal

MINUTES = 60
VLLM_PORT = 8000


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


CONFIG = {
    "APP_NAME": _env("APP_NAME", "bender-ai"),
    "MODEL_NAME": _env("MODEL_NAME", "Qwen/Qwen3-1.7B"),
    "MODEL_REVISION": _env("MODEL_REVISION", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "SERVED_MODEL_NAME": _env("SERVED_MODEL_NAME", "llm"),
    "GPU": _env("GPU", "L4"),
    "MAX_MODEL_LEN": _env("MAX_MODEL_LEN", "8192"),
    "MIN_CONTAINERS": _env("MIN_CONTAINERS", "0"),
    "MAX_CONTAINERS": _env("MAX_CONTAINERS", "2"),
    "TARGET_CONCURRENCY": _env("TARGET_CONCURRENCY", "32"),
    "MAX_CONCURRENCY": _env("MAX_CONCURRENCY", "64"),
    "SCALEDOWN_MINUTES": _env("SCALEDOWN_MINUTES", "5"),
    "FAST_BOOT": _env("FAST_BOOT", "true"),
    "ENABLE_THINKING": _env("ENABLE_THINKING", "false"),
    "REASONING_PARSER": _env("REASONING_PARSER", "qwen3"),
    "TOOL_CALL_PARSER": _env("TOOL_CALL_PARSER", "hermes"),
    "EXTRA_VLLM_ARGS": _env("EXTRA_VLLM_ARGS", ""),
    "PROXY_AUTH": _env("PROXY_AUTH", "true"),
    "API_KEY_SECRET": _env("API_KEY_SECRET", ""),
}


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def vllm_command(config: dict) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        config["MODEL_NAME"],
        "--revision",
        config["MODEL_REVISION"],
        "--served-model-name",
        config["SERVED_MODEL_NAME"],
        config["MODEL_NAME"],
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        config["MAX_MODEL_LEN"],
        "--uvicorn-log-level",
        "info",
        "--enforce-eager" if _flag(config["FAST_BOOT"]) else "--no-enforce-eager",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": _flag(config["ENABLE_THINKING"])}),
    ]
    if config["REASONING_PARSER"]:
        cmd += ["--reasoning-parser", config["REASONING_PARSER"]]
    if config["TOOL_CALL_PARSER"]:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", config["TOOL_CALL_PARSER"]]
    cmd += config["EXTRA_VLLM_ARGS"].split()
    return cmd


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0", "huggingface-hub[hf_xet]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/vllm": vllm_cache_vol,
}

app = modal.App(CONFIG["APP_NAME"])

secrets = [modal.Secret.from_name(CONFIG["API_KEY_SECRET"])] if CONFIG["API_KEY_SECRET"] else []


@app.function(
    image=vllm_image, volumes={"/root/.cache/huggingface": hf_cache_vol}, timeout=30 * MINUTES
)
def download_model():
    from huggingface_hub import snapshot_download

    path = snapshot_download(CONFIG["MODEL_NAME"], revision=CONFIG["MODEL_REVISION"])
    hf_cache_vol.commit()
    print(f"Model cached at {path}")


@app.server(
    image=vllm_image,
    gpu=CONFIG["GPU"],
    env=CONFIG,
    secrets=secrets,
    volumes=volumes,
    port=VLLM_PORT,
    startup_timeout=10 * MINUTES,
    scaledown_window=int(CONFIG["SCALEDOWN_MINUTES"]) * MINUTES,
    min_containers=int(CONFIG["MIN_CONTAINERS"]),
    max_containers=int(CONFIG["MAX_CONTAINERS"]),
    target_concurrency=int(CONFIG["TARGET_CONCURRENCY"]),
    max_concurrency=int(CONFIG["MAX_CONCURRENCY"]),
    exit_grace_period=30,
    unauthenticated=not _flag(CONFIG["PROXY_AUTH"]),
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = vllm_command(CONFIG)
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


def auth_headers() -> dict[str, str]:
    from dotenv import load_dotenv

    load_dotenv()
    headers = {}
    if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
        headers["Modal-Key"] = os.environ["MODAL_KEY"]
        headers["Modal-Secret"] = os.environ["MODAL_SECRET"]
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
    return headers


async def wait_until_healthy(session, timeout: float) -> None:
    import asyncio
    import time

    import aiohttp

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get("/health", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return
                if resp.status in (401, 403):
                    raise RuntimeError(
                        f"Auth rejected (HTTP {resp.status}). Check MODAL_KEY/MODAL_SECRET."
                    )
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"Server not healthy after {timeout:.0f}s")


@app.local_entrypoint()
async def test(prompt: str = "Explain transformers in one sentence.", timeout: int = 10 * MINUTES):
    import aiohttp

    url = await Server.get_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url, headers=auth_headers()) as session:
        await wait_until_healthy(session, timeout)
        print("Server healthy")

        payload = {
            "model": CONFIG["SERVED_MODEL_NAME"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        }
        async with session.post("/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

    message = result["choices"][0]["message"]
    if message.get("reasoning_content") or message.get("reasoning"):
        print("Reasoning:", message.get("reasoning_content") or message.get("reasoning"))
    print("Answer:", message["content"])
    print("Usage:", result["usage"])
