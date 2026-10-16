import json

from openai_server import CONFIG, auth_headers, vllm_command


def flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_default_command_pins_model_revision():
    cmd = vllm_command(CONFIG)
    assert cmd[:3] == ["vllm", "serve", CONFIG["MODEL_NAME"]]
    assert flag_value(cmd, "--revision") == CONFIG["MODEL_REVISION"]


def test_served_under_alias_and_model_name():
    cmd = vllm_command(CONFIG)
    i = cmd.index("--served-model-name")
    assert cmd[i + 1 : i + 3] == ["llm", CONFIG["MODEL_NAME"]]


def test_thinking_disabled_by_default():
    cmd = vllm_command(CONFIG)
    assert json.loads(flag_value(cmd, "--default-chat-template-kwargs")) == {
        "enable_thinking": False
    }


def test_thinking_can_be_enabled():
    cmd = vllm_command({**CONFIG, "ENABLE_THINKING": "true"})
    assert json.loads(flag_value(cmd, "--default-chat-template-kwargs")) == {
        "enable_thinking": True
    }


def test_fast_boot_toggles_eager_mode():
    assert "--enforce-eager" in vllm_command({**CONFIG, "FAST_BOOT": "true"})
    assert "--no-enforce-eager" in vllm_command({**CONFIG, "FAST_BOOT": "false"})


def test_parsers_can_be_disabled():
    cmd = vllm_command({**CONFIG, "REASONING_PARSER": "", "TOOL_CALL_PARSER": ""})
    assert "--reasoning-parser" not in cmd
    assert "--enable-auto-tool-choice" not in cmd


def test_extra_args_are_appended():
    cmd = vllm_command({**CONFIG, "EXTRA_VLLM_ARGS": "--gpu-memory-utilization 0.85"})
    assert cmd[-2:] == ["--gpu-memory-utilization", "0.85"]


def test_auth_headers_from_env(monkeypatch):
    monkeypatch.setenv("MODAL_KEY", "wk-id")
    monkeypatch.setenv("MODAL_SECRET", "ws-secret")
    monkeypatch.setenv("VLLM_API_KEY", "sk-test")
    assert auth_headers() == {
        "Modal-Key": "wk-id",
        "Modal-Secret": "ws-secret",
        "Authorization": "Bearer sk-test",
    }
