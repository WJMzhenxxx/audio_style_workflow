"""SglangMusicClient 单元测试（httpx.MockTransport 模拟服务，不依赖真实 sglang）。

覆盖：/generate 基础模式、/v1/chat/completions（OpenAI 兼容 +
thinking 分离 + budget 控制）两种适配，以及请求体结构与错误处理。
"""
import json
import logging
import os

import httpx
import pytest

from audio_style_workflow.api.base import SongAnalysisError, create_client
from audio_style_workflow.api.sglang_client import (
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_SGLANG_BASE,
    ENV_CUSTOM_LOGIT_PROCESSOR,
    ProcessorResolveError,
    SglangMusicClient,
    resolve_thinking_processor,
    validate_processor_string,
)
from audio_style_workflow.style_parser import StyleParser

#: 合法的 processor 序列化串（sglang 服务端会 orjson.loads 成 dict）
VALID_PROCESSOR = json.dumps({"callable": "deadbeef"})


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


def test_thinking_budget_requires_valid_processor():
    # 未提供序列化串且本机无 sglang 时，必须报错（绝不能把类名发给服务端）
    with pytest.raises(ProcessorResolveError):
        SglangMusicClient(thinking_budget=50)
    # 明确传合法 JSON 时可用
    client = SglangMusicClient(thinking_budget=50, custom_logit_processor=VALID_PROCESSOR)
    assert client.custom_logit_processor == VALID_PROCESSOR
    # 未设置 budget 时不启用
    plain = SglangMusicClient()
    assert plain.thinking_budget is None
    assert plain.custom_logit_processor is None


def test_bare_class_name_rejected_before_sending():
    """回归：曾把类名当 processor 发给 sglang，导致服务端 orjson 崩溃。"""
    with pytest.raises(ProcessorResolveError) as err:
        SglangMusicClient(
            thinking_budget=50,
            custom_logit_processor="Qwen3InstructionInjectionThinkingBudgetLogitProcessor",
        )
    assert "JSON" in str(err.value)


def test_processor_accepts_at_file(tmp_path):
    f = tmp_path / "processor.json"
    f.write_text(VALID_PROCESSOR, encoding="utf-8")
    assert validate_processor_string(f"@{f}") == VALID_PROCESSOR
    client = SglangMusicClient(thinking_budget=10, custom_logit_processor=f"@{f}")
    assert client.custom_logit_processor == VALID_PROCESSOR


def test_processor_env_var(monkeypatch):
    monkeypatch.setenv(ENV_CUSTOM_LOGIT_PROCESSOR, VALID_PROCESSOR)
    assert resolve_thinking_processor() == VALID_PROCESSOR
    client = SglangMusicClient(thinking_budget=20)
    assert client.custom_logit_processor == VALID_PROCESSOR


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
        custom_logit_processor=VALID_PROCESSOR,
        transport=transport,
    )

    payload = client.analyze(audio_file)

    body = transport.requests[0]["body"]
    assert body["separate_reasoning"] is True
    assert body["custom_logit_processor"] == VALID_PROCESSOR
    assert body["custom_params"] == {"thinking_budget": 50}

    # reasoning_content 分离后不影响风格提取（只用 content）
    assert payload["styles"] == ["Rock", "Grunge"]
    assert "reasoning" not in payload["description"].lower()


def test_chat_mode_custom_processor_override(audio_file):
    other = json.dumps({"callable": "cafebabe"})
    transport = RecordingTransport({"choices": [{"message": {"content": "Pop\nx"}}]})
    client = SglangMusicClient(
        api_mode="chat",
        thinking_budget=64,
        custom_logit_processor=other,
        transport=transport,
    )
    client.analyze(audio_file)
    body = transport.requests[0]["body"]
    assert body["custom_logit_processor"] == other
    assert body["custom_params"] == {"thinking_budget": 64}


def test_chat_mode_missing_choices(audio_file):
    transport = RecordingTransport({"error": {"message": "invalid request"}})
    client = SglangMusicClient(api_mode="chat", transport=transport)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "choices" in str(err.value)


def test_chat_mode_falls_back_to_reasoning_content(audio_file):
    """回归：思考未闭合时 content=null，答案在 reasoning_content，不能直接判失败。"""
    transport = RecordingTransport({"choices": [{"index": 0, "message": {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Genre: Pop, R&B\n\nTempo: 慢板\n\nKey: C major",
    }}]})
    client = SglangMusicClient(api_mode="chat", transport=transport)
    payload = client.analyze(audio_file)

    assert payload["content_source"] == "reasoning_content"
    assert payload["styles"] == ["Pop", "R&B"]
    assert payload["description"].startswith("Genre: Pop, R&B")
    # 下游 StyleParser 同样能解析
    styles, source, _ = StyleParser().parse(payload)
    assert styles == ["Pop", "R&B"] and source == "json"


def test_chat_mode_prefers_content_over_reasoning(audio_file):
    transport = RecordingTransport({"choices": [{"message": {
        "content": "Jazz, Lofi\nanswer",
        "reasoning_content": "internal thoughts",
    }}]})
    payload = SglangMusicClient(api_mode="chat", transport=transport).analyze(audio_file)
    assert payload["content_source"] == "content"
    assert payload["styles"] == ["Jazz", "Lofi"]


def test_chat_mode_auto_degrades_when_server_rejects_processor(audio_file):
    """服务端未开 --enable-custom-logit-processor 时，自动去掉 budget 字段重试。"""
    calls = []

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if "custom_logit_processor" in body:
            return httpx.Response(
                400, json={"detail": "The server does not support custom logit processor"}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "Jazz\nok"}}]})

    transport = httpx.MockTransport(handler)
    client = SglangMusicClient(
        api_mode="chat", thinking_budget=50,
        custom_logit_processor=VALID_PROCESSOR, transport=transport,
    )
    payload = client.analyze(audio_file)

    assert len(calls) == 2, calls
    assert "custom_logit_processor" in calls[0] and "custom_params" in calls[0]
    assert "custom_logit_processor" not in calls[1]
    assert "separate_reasoning" not in calls[1]
    assert client.thinking_budget_applied is False
    assert payload["thinking_budget_applied"] is False
    assert payload["styles"] == ["Jazz"]

    # 之后不再带着被拒字段重试
    client.analyze(audio_file)
    assert "custom_logit_processor" not in calls[-1]


def test_chat_mode_unrelated_error_is_not_degraded(audio_file):
    transport = RecordingTransport({"detail": "CUDA out of memory"}, status_code=500)
    client = SglangMusicClient(
        api_mode="chat", thinking_budget=50,
        custom_logit_processor=VALID_PROCESSOR, transport=transport,
    )
    with pytest.raises(SongAnalysisError):
        client.analyze(audio_file)
    assert client.thinking_budget_applied is True


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


def test_cli_sglang_defaults(tmp_path, monkeypatch):
    from audio_style_workflow.main import build_client

    monkeypatch.delenv("SGLANG_BASE", raising=False)
    args = _parse_cli(tmp_path, "--sglang")
    client = build_client(args)
    assert isinstance(client, SglangMusicClient)
    assert client.endpoint == DEFAULT_SGLANG_BASE
    assert client.api_mode == "generate"


def test_cli_sglang_env_default(tmp_path, monkeypatch):
    from audio_style_workflow.main import build_client

    monkeypatch.setenv("SGLANG_BASE", "http://env-sglang:30000")
    args = _parse_cli(tmp_path, "--sglang")
    client = build_client(args)
    assert client.endpoint == "http://env-sglang:30000"


def test_cli_sglang_base_option(tmp_path, monkeypatch):
    from audio_style_workflow.main import build_client

    monkeypatch.delenv("SGLANG_BASE", raising=False)
    args = _parse_cli(tmp_path, "--sglang-base", "http://explicit-host:30000",
                      "--sglang-thinking-budget", "50",
                      "--sglang-custom-logit-processor", VALID_PROCESSOR)
    client = build_client(args)
    assert client.endpoint == "http://explicit-host:30000"
    assert client.api_mode == "generate"      # CLI 固定 generate
    assert client.thinking_budget == 50


def test_cli_sglang_base_overrides_env(tmp_path, monkeypatch):
    from audio_style_workflow.main import build_client

    monkeypatch.setenv("SGLANG_BASE", "http://env-sglang:30000")
    args = _parse_cli(tmp_path, "--sglang-base", "http://cli:9000")
    assert build_client(args).endpoint == "http://cli:9000"


def test_cli_sglang_custom(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang", "http://gpu-box:30000",
                      "--sglang-thinking-budget", "50", "--max-new-tokens", "1024",
                      "--sglang-custom-logit-processor", VALID_PROCESSOR)
    client = build_client(args)
    assert client.endpoint == "http://gpu-box:30000"
    assert client.api_mode == "generate"
    assert client.thinking_budget == 50
    assert client.max_new_tokens == 1024
    assert client.custom_logit_processor == VALID_PROCESSOR


def test_cli_sglang_conflicts_with_local_and_mock(tmp_path):
    from audio_style_workflow.main import build_client

    for extra in (["--sglang", "--local"], ["--sglang", "--mock"], ["--local", "--mock"],
                  ["--sglang-base", "http://x:1", "--local"],
                  ["--sglang-base", "http://x:1", "--mock"]):
        args = _parse_cli(tmp_path, *extra)
        with pytest.raises(ValueError):
            build_client(args)


# ----------------------------------------------------------------------
# 本地 -> 服务端路径映射（跨机部署）
# ----------------------------------------------------------------------
def test_server_path_map_translates(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    audio = music / "album" / "a.flac"
    audio.parent.mkdir()
    audio.write_bytes(b"x")

    client = SglangMusicClient(
        api_mode="chat", audio_path_map=[(str(music), "/data/music")]
    )
    assert client._server_path(audio) == "/data/music/album/a.flac"
    assert client._server_path(music / "b.mp3") == "/data/music/b.mp3"


def test_server_path_without_map_is_absolute(tmp_path):
    audio = tmp_path / "b.mp3"
    audio.write_bytes(b"x")
    assert SglangMusicClient()._server_path(audio) == str(audio.resolve())


def test_server_path_unmatched_prefix_untouched(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    audio = music / "x.flac"
    audio.write_bytes(b"x")
    client = SglangMusicClient(api_mode="chat", audio_path_map=[(str(tmp_path / "other"), "/nope")])
    assert client._server_path(audio) == str(audio.resolve())


def test_cli_sglang_path_map(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang", "http://h:1",
                      "--sglang-path-map", r"C:\music=/data/music")
    client = build_client(args)
    assert client.audio_path_map == [(os.path.normpath(r"C:\music"), "/data/music")]


def test_cli_sglang_path_map_invalid(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang", "http://h:1", "--sglang-path-map", "no-equals")
    with pytest.raises(ValueError):
        build_client(args)


def test_identical_responses_trigger_warning(audio_file, caplog):
    """连续返回相同文本 -> 提示音频可能没被服务端读到。"""
    same = {"choices": [{"message": {"content": None, "reasoning_content": "固定文本"}}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=same))
    client = SglangMusicClient(api_mode="chat", transport=transport)

    with caplog.at_level(logging.WARNING, logger="audio_style_workflow.api.sglang_client"):
        client.analyze(audio_file)
        client.analyze(audio_file)

    assert any("完全相同" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# base64 内联（不依赖共享文件系统）
# ----------------------------------------------------------------------
def test_audio_base64_sends_data_url(audio_file):
    import base64

    calls = []

    def handler(request):
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Pop\nok"}}]})

    client = SglangMusicClient(
        api_mode="chat", audio_base64=True, transport=httpx.MockTransport(handler)
    )
    client.analyze(audio_file)

    content = calls[0]["messages"][0]["content"]
    url = next(x for x in content if x["type"] == "audio_url")["audio_url"]["url"]
    assert url.startswith("data:audio/flac;base64,")
    assert base64.b64decode(url.split(",", 1)[1], validate=True) == audio_file.read_bytes()


def test_audio_path_mode_is_default(audio_file):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Pop\nok"}}]})

    SglangMusicClient(api_mode="chat", transport=httpx.MockTransport(handler)).analyze(audio_file)
    url = calls[0]["messages"][0]["content"][0]["audio_url"]["url"]
    assert url == str(audio_file.resolve())


def test_audio_base64_generate_mode(audio_file):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"text": "Pop\nok"})

    client = SglangMusicClient(audio_base64=True, transport=httpx.MockTransport(handler))
    client.analyze(audio_file)
    assert calls[0]["audio_data"].startswith("data:audio/flac;base64,")


def test_describe_audio_hides_payload():
    client = SglangMusicClient(audio_base64=True)
    short = client._describe_audio("data:audio/mpeg;base64," + "A" * 5000)
    assert short.startswith("data:audio/mpeg;base64,<")
    assert len(short) < 60


def test_cli_sglang_audio_base64(tmp_path):
    from audio_style_workflow.main import build_client

    args = _parse_cli(tmp_path, "--sglang", "http://h:1", "--sglang-audio-base64")
    assert build_client(args).audio_base64 is True
    assert build_client(_parse_cli(tmp_path, "--sglang", "http://h:1")).audio_base64 is False


def test_finish_reason_recorded(audio_file):
    transport = RecordingTransport({
        "choices": [{"finish_reason": "length",
                     "message": {"content": None, "reasoning_content": "x" * 10}}]
    })
    payload = SglangMusicClient(api_mode="chat", transport=transport).analyze(audio_file)
    assert payload["finish_reason"] == "length"
    assert payload["content_source"] == "reasoning_content"


def test_warns_when_prompt_tokens_too_small(audio_file, caplog):
    """prompt_tokens 只有纯文本规模 -> 说明音频没进模型。"""
    transport = RecordingTransport({
        "choices": [{"finish_reason": "stop",
                     "message": {"content": "Pop\nok"}}],
        "usage": {"prompt_tokens": 105, "completion_tokens": 10},
    })
    client = SglangMusicClient(api_mode="chat", audio_base64=True, transport=transport)
    with caplog.at_level(logging.WARNING, logger="audio_style_workflow.api.sglang_client"):
        payload = client.analyze(audio_file)

    assert payload["prompt_tokens"] == 105
    assert any("音频很可能没有被服务端解析" in r.message for r in caplog.records)


def test_no_warning_when_prompt_tokens_large(audio_file, caplog):
    transport = RecordingTransport({
        "choices": [{"finish_reason": "stop", "message": {"content": "Pop\nok"}}],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 10},
    })
    client = SglangMusicClient(api_mode="chat", transport=transport)
    with caplog.at_level(logging.WARNING, logger="audio_style_workflow.api.sglang_client"):
        client.analyze(audio_file)
    assert not any("音频很可能没有被服务端解析" in r.message for r in caplog.records)


def test_chat_without_audio_falls_back_to_generate(audio_file):
    """该服务的 chat 端点不吃 audio_url 时，自动改用 /generate 并记住。"""
    calls = []

    def handler(request):
        path = request.url.path
        calls.append(path)
        if path.endswith("/v1/chat/completions"):
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {"content": "Pop\nx"}}],
                "usage": {"prompt_tokens": 105},
            })
        return httpx.Response(200, json={
            "text": "list: Dream Pop, Synth Pop\nBPM: 100\nTempo: moderate\nKey: F# major",
            "meta_info": {"prompt_tokens": 5126},
        })

    client = SglangMusicClient(api_mode="chat", transport=httpx.MockTransport(handler))
    payload = client.analyze(audio_file)

    assert calls == ["/v1/chat/completions", "/generate"]
    assert payload["api_mode"] == "generate"
    assert payload["styles"] == ["Dream Pop", "Synth Pop"]
    assert payload["bpm"] == 100
    assert payload["key"] == "F# major"
    assert payload["prompt_tokens"] == 5126

    # 后续歌曲不再浪费一次 chat 请求
    client.analyze(audio_file)
    assert calls[-1] == "/generate"


def test_chat_with_audio_keeps_chat(audio_file):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "Jazz\nok"}}],
            "usage": {"prompt_tokens": 2000},
        })

    client = SglangMusicClient(api_mode="chat", transport=httpx.MockTransport(handler))
    payload = client.analyze(audio_file)
    assert payload["api_mode"] == "chat"
    assert calls == ["/v1/chat/completions"]


def test_cli_prompt_lang(tmp_path):
    from audio_style_workflow.main import build_client

    zh_client = build_client(_parse_cli(tmp_path, "--sglang", "http://h:1"))
    assert "中文" in zh_client.prompt          # 默认 zh

    en_client = build_client(_parse_cli(tmp_path, "--sglang", "http://h:1", "--prompt-lang", "en"))
    assert "Listen to this track" in en_client.prompt


def test_cli_per_song_flag(tmp_path):
    from audio_style_workflow.main import build_arg_parser

    assert build_arg_parser().parse_args([str(tmp_path), "--per-song"]).per_song is True
    assert build_arg_parser().parse_args([str(tmp_path)]).per_song is False


def test_clip_audio_payload(tmp_path):
    """--sglang-clip：截取中间一段并转成 16k 单声道 WAV。"""
    import base64
    import subprocess

    from audio_style_workflow.splitter import resolve_ffmpeg

    ffmpeg = resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip("本机没有 ffmpeg")

    src = tmp_path / "long.mp3"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
         "-c:a", "libmp3lame", str(src)],
        check=True, capture_output=True,
    )

    client = SglangMusicClient(api_mode="generate", audio_base64=True, clip_seconds=10)
    uri = client._audio_payload(src)
    head, _, body = uri.partition(",")
    assert head == "data:audio/wav;base64"
    raw = base64.b64decode(body, validate=True)
    assert len(raw) > 1000  # 有实际音频数据

    # 关掉截取则应发原始文件
    plain = SglangMusicClient(api_mode="generate", audio_base64=True)
    assert plain._audio_payload(src).startswith("data:audio/mpeg;base64,")


def test_sampling_params_passed(audio_file):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"text": "TAGS: Pop\nBPM: 100",
                                         "meta_info": {"prompt_tokens": 900}})

    client = SglangMusicClient(
        audio_base64=True, temperature=1.0, top_p=0.8, top_k=50,
        transport=httpx.MockTransport(handler),
    )
    client.analyze(audio_file)
    sampling = calls[0]["sampling_params"]
    assert sampling["temperature"] == 1.0
    assert sampling["top_p"] == 0.8 and sampling["top_k"] == 50
