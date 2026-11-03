"""SglangMusicClient 单元测试（httpx.MockTransport 模拟服务，不依赖真实 sglang）。

覆盖：/generate 基础模式、/v1/chat/completions（OpenAI 兼容 +
thinking 分离 + budget 控制）两种适配，以及请求体结构与错误处理。
"""
import json

import httpx
import pytest

from audio_style_workflow.api.base import SongAnalysisError, create_client
from audio_style_workflow.api.sglang_client import (
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_SGLANG_BASE,
    DEFAULT_THINKING_PROCESSOR,
    SglangMusicClient,
)
from audio_style_workflow.style_parser import StyleParser


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "song.flac"
    f.write_bytes(b"fake-flac-bytes")
    return f


class RecordingTransport(httpx.MockTransport):
    """记录请求体并回放固定 JSON 的 MockTransport。"""

    def __init__(self, response_json, status_code=200):
        self.requests = []
        self.response_json = response_json
        self.status_code = status_code
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        self.requests.append({"url": request.url.path, "headers": dict(request.headers), "body": body})
        return httpx.Response(self.status_code, json=self.response_json)


# ----------------------------------------------------------------------
# 构造与工厂
# ----------------------------------------------------------------------
def test_invalid_api_mode_rejected():
    with pytest.raises(ValueError):
        SglangMusicClient(api_mode="oops")
    with pytest.raises(ValueError):
        SglangMusicClient(endpoint="ftp://x")


def test_endpoint_trailing_slash_normalized():
    client = SglangMusicClient("http://host:30000/")
    assert client.endpoint == "http://host:30000"
    assert client._url_generate == "http://host:30000/generate"
    assert client._url_chat == "http://host:30000/v1/chat/completions"


def test_factory_creates_sglang_client():
    client = create_client("sglang", endpoint="http://gpu:30000")
    assert isinstance(client, SglangMusicClient)
    assert client.endpoint == "http://gpu:30000"
    assert client.api_mode == "generate"


def test_thinking_budget_enables_processor_default():
    client = SglangMusicClient(thinking_budget=50)
    assert client.custom_logit_processor == DEFAULT_THINKING_PROCESSOR
    # 未设置 budget 时默认不启用
    plain = SglangMusicClient()
    assert plain.thinking_budget is None


# ----------------------------------------------------------------------
# generate 模式（基础音乐描述）
# ----------------------------------------------------------------------
def test_generate_mode_request_and_parse(audio_file):
    model_text = "Pop, R&B\nA groovy neo-soul tune with warm Rhodes chords."
    transport = RecordingTransport({"text": model_text, "meta_info": {"finish_reason": "stop"}})
    client = SglangMusicClient("http://localhost:30000", max_new_tokens=1024, temperature=0.0,
                               transport=transport)

    payload = client.analyze(audio_file)

    # 请求结构与官方 curl 示例一致
    req = transport.requests[0]
    assert req["url"] == "/generate"
    body = req["body"]
    assert body["text"] == DEFAULT_ANALYSIS_PROMPT
    assert body["audio_data"] == str(audio_file.resolve())
    assert body["sampling_params"] == {"max_new_tokens": 1024, "temperature": 0.0}

    # 响应契约
    assert payload["status"] == "ok"
    assert payload["styles"] == ["Pop", "R&B"]
    assert payload["engine"] == "sglang-music"
    assert payload["api_mode"] == "generate"
    assert "Rhodes" in payload["description"]

    # 直接喂给 StyleParser 可确定性提取
    styles, source, _ = StyleParser().parse(payload)
    assert styles == ["Pop", "R&B"] and source == "json"


def test_generate_mode_missing_text_field(audio_file):
    transport = RecordingTransport({"meta_info": {}})  # 无 text
    client = SglangMusicClient(transport=transport)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "text 字段" in str(err.value)


def test_generate_mode_http_error(audio_file):
    transport = RecordingTransport({"detail": "model not loaded"}, status_code=500)
    client = SglangMusicClient(transport=transport)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "HTTP 500" in str(err.value)


def test_generate_mode_non_json_response(audio_file):
    bad = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>gateway</html>"))
    client = SglangMusicClient(transport=bad)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "JSON" in str(err.value)


# ----------------------------------------------------------------------
# chat 模式（OpenAI 兼容 + thinking 分离 + budget 控制）
# ----------------------------------------------------------------------
def test_chat_mode_minimal_request_and_parse(audio_file):
    content = "Jazz, Lofi Hip-Hop\nBrushed drums and mellow piano over vinyl crackle."
    transport = RecordingTransport({
        "choices": [{"message": {"role": "assistant", "content": content}}]
    })
    client = SglangMusicClient(api_mode="chat", transport=transport)

    payload = client.analyze(audio_file)

    req = transport.requests[0]
    assert req["url"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "default"
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.0
    messages = body["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "audio_url", "audio_url": {"url": str(audio_file.resolve())}}
    assert messages[0]["content"][1]["type"] == "text"
    # 未开 thinking budget 时不应附带 custom 字段
    assert "separate_reasoning" not in body
    assert "custom_logit_processor" not in body
    assert "custom_params" not in body

    assert payload["styles"] == ["Jazz", "Lofi Hip-Hop"]


def test_chat_mode_with_thinking_budget(audio_file):
    """完整模式：separate_reasoning + custom_logit_processor + custom_params。"""
    content = "Rock, Grunge\nDetuned guitars with a raw 90s production aesthetic."
    transport = RecordingTransport({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": "Let me analyze the spectrogram first...",
            }
        }]
    })
    client = SglangMusicClient(
        api_mode="chat",
        thinking_budget=50,
        model_name="default",
        transport=transport,
    )

    payload = client.analyze(audio_file)

    body = transport.requests[0]["body"]
    assert body["separate_reasoning"] is True
    assert body["custom_logit_processor"] == DEFAULT_THINKING_PROCESSOR
    assert body["custom_params"] == {"thinking_budget": 50}

    # reasoning_content 分离后不影响风格提取（只用 content）
    assert payload["styles"] == ["Rock", "Grunge"]
    assert "reasoning" not in payload["description"].lower()


def test_chat_mode_custom_processor_override(audio_file):
    transport = RecordingTransport({"choices": [{"message": {"content": "Pop\nx"}}]})
    client = SglangMusicClient(
        api_mode="chat",
        thinking_budget=64,
        custom_logit_processor="MyCustomProcessor:v2",
        transport=transport,
    )
    client.analyze(audio_file)
    body = transport.requests[0]["body"]
    assert body["custom_logit_processor"] == "MyCustomProcessor:v2"
    assert body["custom_params"] == {"thinking_budget": 64}


def test_chat_mode_missing_choices(audio_file):
    transport = RecordingTransport({"error": {"message": "invalid request"}})
    client = SglangMusicClient(api_mode="chat", transport=transport)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "choices" in str(err.value)


def test_chat_mode_empty_content(audio_file):
    transport = RecordingTransport({"choices": [{"message": {"content": ""}}]})
    client = SglangMusicClient(api_mode="chat", transport=transport)
    with pytest.raises(SongAnalysisError):
        client.analyze(audio_file)


# ----------------------------------------------------------------------
# 通用行为
# ----------------------------------------------------------------------
def test_missing_audio_file(audio_file):
    client = SglangMusicClient()
    with pytest.raises(SongAnalysisError):
        client.analyze(audio_file.parent / "nope.flac")


def test_network_failure_wrapped(audio_file):
    def raiser(request):
        raise httpx.ConnectError("connection refused")

    client = SglangMusicClient(transport=httpx.MockTransport(raiser))
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "请求 sglang 服务失败" in str(err.value)


def test_no_label_line_keeps_description(audio_file):
    text = "This song is a mid-tempo pop ballad, featuring soft vocals and a gentle groove."
    transport = RecordingTransport({"text": text})
    client = SglangMusicClient(transport=transport)

    payload = client.analyze(audio_file)
    assert "styles" not in payload
    assert payload["description"] == text
    assert "description" in payload["message"]


def test_api_key_header_sent(audio_file):
    transport = RecordingTransport({"text": "Pop\nx"})
    client = SglangMusicClient(api_key="secret-token", transport=transport)
    client.analyze(audio_file)
    auth = transport.requests[0]["headers"].get("authorization")
    assert auth == "Bearer secret-token"


def test_prompt_customizable(audio_file):
    transport = RecordingTransport({"text": "Pop\nx"})
    client = SglangMusicClient(prompt="Summarize the genre briefly.", transport=transport)
    client.analyze(audio_file)
    assert transport.requests[0]["body"]["text"] == "Summarize the genre briefly."


# ----------------------------------------------------------------------
# CLI 集成
# ----------------------------------------------------------------------
def _parse_cli(tmp_path, *extra):
    from audio_style_workflow.main import build_arg_parser

    return build_arg_parser().parse_args([str(tmp_path), *extra])


def test_cli_sglang_defaults(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang")
    client = build_client(args)
    assert isinstance(client, SglangMusicClient)
    assert client.endpoint == DEFAULT_SGLANG_BASE
    assert client.api_mode == "generate"


def test_cli_sglang_custom(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang", "http://gpu-box:30000", "--sglang-api", "chat",
                      "--sglang-thinking-budget", "50", "--max-new-tokens", "1024")
    client = build_client(args)
    assert client.endpoint == "http://gpu-box:30000"
    assert client.api_mode == "chat"
    assert client.thinking_budget == 50
    assert client.max_new_tokens == 1024


def test_cli_sglang_conflicts_with_local_and_mock(tmp_path):
    from audio_style_workflow.main import build_client

    for extra in (["--sglang", "--local"], ["--sglang", "--mock"], ["--local", "--mock"]):
        args = _parse_cli(tmp_path, *extra)
        with pytest.raises(ValueError):
            build_client(args)
