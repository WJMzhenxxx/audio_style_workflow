"""LocalMusicFlamingoClient 单元测试。

不依赖真实 transformers / torch / GPU：通过注入 fake 后端验证
调用链、响应契约、风格提取与错误处理。
"""
import argparse

import pytest

from audio_style_workflow.api.base import SongAnalysisError, create_client
from audio_style_workflow.api.local_model_client import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    LocalMusicFlamingoClient,
    extract_styles_from_text,
)
from audio_style_workflow.style_parser import StyleParser


# ----------------------------------------------------------------------
# fake transformers 后端
# ----------------------------------------------------------------------
class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        return self  # outputs[:, n:] 切片后仍可交给 batch_decode

    def to(self, *args, **kwargs):
        return self


class FakeInputs(dict):
    """模拟 processor.apply_chat_template 返回的 BatchFeature：
    dict-like（**inputs 展开）+ 属性访问。"""

    def __init__(self):
        super().__init__(input_ids=FakeTensor((1, 12)), input_features=FakeTensor((1, 100, 80)))

    def to(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeProcessor:
    from_pretrained_calls = 0

    @classmethod
    def from_pretrained(cls, model_id):
        cls.from_pretrained_calls += 1
        return cls()

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt, return_dict):
        self.last_conversation = conversation
        assert tokenize is True and add_generation_prompt is True and return_dict is True
        return FakeInputs()

    def batch_decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [self.decode_result]


class FakeModel:
    from_pretrained_calls = 0

    @classmethod
    def from_pretrained(cls, model_id, device_map=None):
        cls.from_pretrained_calls += 1
        if getattr(cls, "raise_on_load", None):
            raise OSError(cls.raise_on_load)
        inst = cls()
        inst.loaded_device_map = device_map
        return inst

    def __init__(self):
        self.device = "cpu"
        self.dtype = "float32"
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if getattr(self, "raise_on_generate", None):
            raise RuntimeError(self.raise_on_generate)
        return FakeTensor((1, 20))


def make_client(tmp_path, decode_result, monkeypatch, **client_kwargs):
    """构造注入了 fake 后端的客户端。"""
    client = LocalMusicFlamingoClient(**client_kwargs)
    model_holder = {}
    processor = FakeProcessor()
    processor.decode_result = decode_result

    def fake_import(self):
        model = FakeModel()
        model_holder["m"] = model

        class M:
            @classmethod
            def from_pretrained(cls, model_id, device_map=None):
                model.loaded_device_map = device_map
                FakeModel.from_pretrained_calls += 1
                return model

        class P:
            @classmethod
            def from_pretrained(cls, model_id):
                FakeProcessor.from_pretrained_calls += 1
                return processor

        return M, P

    monkeypatch.setattr(client, "_import_backend", fake_import.__get__(client))
    return client, model_holder, processor


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "song.mp3"
    f.write_bytes(b"fake-audio-bytes")
    return f


# ----------------------------------------------------------------------
# 纯函数：extract_styles_from_text
# ----------------------------------------------------------------------
def test_extract_first_line_labels():
    text = "Pop, R&B, Jazz\nA bright up-tempo track with warm production."
    assert extract_styles_from_text(text) == ["Pop", "R&B", "Jazz"]


def test_extract_strips_prefix():
    assert extract_styles_from_text("Styles: Pop, Rock") == ["Pop", "Rock"]
    assert extract_styles_from_text("Genres: Hip-Hop; Lofi") == ["Hip-Hop", "Lofi"]
    assert extract_styles_from_text("风格: 流行, 摇滚") == ["流行", "摇滚"]


def test_extract_handles_trailing_punctuation():
    assert extract_styles_from_text("City Pop, Jazz.") == ["City Pop", "Jazz"]


def test_extract_dedupes():
    assert extract_styles_from_text("Pop, Pop, Rock") == ["Pop", "Rock"]


def test_extract_sentence_fallback_empty():
    # 模型直接给了描述句（没有标签行）-> 不硬拆
    text = "This song is a mid-tempo pop ballad, featuring soft vocals and a gentle groove."
    assert extract_styles_from_text(text) == []


def test_extract_long_fragment_rejected():
    # 拆出的片段超长 -> 判定整行是句子
    text = "A very long explanation about the song structure that goes on and on and on for a while here now"
    assert extract_styles_from_text(text) == []


def test_extract_too_many_labels_rejected():
    assert extract_styles_from_text("a, b, c, d, e, f, g, h, i") == []


def test_extract_empty_text():
    assert extract_styles_from_text("") == []
    assert extract_styles_from_text("\n\n") == []


def test_extract_keeps_multi_word_label():
    assert extract_styles_from_text("Hip Hop, Drum and Bass") == ["Hip Hop", "Drum and Bass"]


# ----------------------------------------------------------------------
# analyze 契约与调用链
# ----------------------------------------------------------------------
def test_analyze_returns_contract(tmp_path, monkeypatch, audio_file):
    decode = "Pop, R&B\nUpbeat funk-flavored pop with punchy brass."
    client, model_holder, processor = make_client(tmp_path, decode, monkeypatch)

    payload = client.analyze(audio_file)

    assert payload["status"] == "ok"
    assert payload["styles"] == ["Pop", "R&B"]
    assert "upbeat funk" in payload["description"].lower()
    assert payload["engine"] == "music-flamingo"
    assert payload["model_id"] == DEFAULT_MODEL_ID


def test_analyze_passes_official_call_chain(tmp_path, monkeypatch, audio_file):
    """验证与 HF 官方示例一致的调用链与参数。"""
    decode = "Rock\nGritty guitar rock."
    client, model_holder, processor = make_client(
        tmp_path, decode, monkeypatch, max_new_tokens=123, model_id="custom/model-x"
    )
    payload = client.analyze(audio_file)

    model = model_holder["m"]
    assert len(model.generate_calls) == 1
    gen_kwargs = model.generate_calls[0]
    assert gen_kwargs["max_new_tokens"] == 123
    assert "input_ids" in gen_kwargs and "input_features" in gen_kwargs
    assert payload["model_id"] == "custom/model-x"

    # prompt 与音频路径进入 conversation
    conversation = processor.last_conversation
    content = conversation[0]["content"]
    assert content[0]["text"] == DEFAULT_PROMPT
    assert content[1] == {"type": "audio", "path": str(audio_file)}


def test_analyze_without_label_line_keeps_description(tmp_path, monkeypatch, audio_file):
    decode = "This is a smooth jazz ballad with brushed drums and warm piano chords throughout."
    client, _, _ = make_client(tmp_path, decode, monkeypatch)

    payload = client.analyze(audio_file)
    assert "styles" not in payload  # 解析不到就不硬造
    assert payload["description"] == decode
    assert "description" in payload["message"]

    # StyleParser（无 LLM）应对该 payload 报风格解析失败，而不是误提取
    with pytest.raises(Exception):
        StyleParser().parse(payload)


def test_analyze_payload_feeds_style_parser(tmp_path, monkeypatch, audio_file):
    decode = "Jazz, Blues\nA smoky late-night combo recording."
    client, _, _ = make_client(tmp_path, decode, monkeypatch)
    payload = client.analyze(audio_file)

    styles, source, _ = StyleParser().parse(payload)
    assert styles == ["Jazz", "Blues"]
    assert source == "json"


def test_lazy_load_only_once(tmp_path, monkeypatch, audio_file):
    decode = "Pop\nNice."
    client, model_holder, _ = make_client(tmp_path, decode, monkeypatch)
    assert not client._loaded  # 构造后未加载

    client.analyze(audio_file)
    client.analyze(audio_file)
    assert FakeModel.from_pretrained_calls >= 1
    assert client._loaded
    # 第二次 analyze 复用同一实例，不重复加载
    assert model_holder["m"].generate_calls.__len__() == 2


def test_preload_then_analyze(tmp_path, monkeypatch, audio_file):
    decode = "Pop\nNice."
    client, _, _ = make_client(tmp_path, decode, monkeypatch)
    client.load()  # 允许提前预热
    assert client._loaded
    client.analyze(audio_file)


def test_missing_file_raises(audio_file):
    client = LocalMusicFlamingoClient()
    with pytest.raises(SongAnalysisError):
        client.analyze(audio_file.parent / "nope.mp3")


def test_generate_failure_wrapped(tmp_path, monkeypatch, audio_file):
    decode = "Pop\nx"
    client, model_holder, _ = make_client(tmp_path, decode, monkeypatch)
    client.load()  # 先触发加载，model_holder 才有实例

    def boom(self, **kw):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(type(model_holder["m"]), "generate", boom)
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "推理失败" in str(err.value)


def test_model_load_failure_wrapped(tmp_path, monkeypatch, audio_file):
    client = LocalMusicFlamingoClient()

    def fake_import(self):
        class M:
            @classmethod
            def from_pretrained(cls, mid, device_map=None):
                raise OSError("HTTP 404: model not found")

        class P:
            @classmethod
            def from_pretrained(cls, mid):
                return None

        return M, P

    import types

    monkeypatch.setattr(client, "_import_backend", types.MethodType(fake_import, client))
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "加载模型失败" in str(err.value)


def test_import_missing_dependency_wrapped(tmp_path, monkeypatch, audio_file):
    """本机未安装 transformers 时，ImportError 应被包装成带安装指引的 SongAnalysisError。"""
    import importlib.util

    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("本机已安装 transformers，无法模拟缺失场景")
    client = LocalMusicFlamingoClient()
    with pytest.raises(SongAnalysisError) as err:
        client.analyze(audio_file)
    assert "requirements-local.txt" in str(err.value)


def test_generate_is_serialized(tmp_path, monkeypatch, audio_file):
    """内部锁保证 generate 串行（fake 计数并发进入次数）。"""
    decode = "Pop\nx"
    client, model_holder, _ = make_client(tmp_path, decode, monkeypatch)
    client.load()

    active = []
    peak = []

    import threading

    def tracked_generate(self, **kw):
        active.append(1)
        peak.append(len(active))
        import time

        time.sleep(0.05)
        active.pop()
        return FakeTensor((1, 20))

    monkeypatch.setattr(type(model_holder["m"]), "generate", tracked_generate)

    threads = [threading.Thread(target=lambda: client.analyze(audio_file)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(peak) == 1, "generate 必须被锁串行化"


# ----------------------------------------------------------------------
# 工厂与 CLI
# ----------------------------------------------------------------------
def test_create_client_local_factory():
    client = create_client("local", model_id="x/y")
    assert isinstance(client, LocalMusicFlamingoClient)
    assert client.model_id == "x/y"
    assert not client._loaded


def test_cli_build_client_local(tmp_path, monkeypatch):
    from audio_style_workflow.main import build_arg_parser, build_client

    parser = build_arg_parser()
    args = parser.parse_args([str(tmp_path), "--local", "--model-id", "nvidia/music-flamingo-2601-hf",
                              "--max-new-tokens", "64"])
    client = build_client(args)
    assert isinstance(client, LocalMusicFlamingoClient)
    assert client.model_id == "nvidia/music-flamingo-2601-hf"
    assert client.max_new_tokens == 64


def test_cli_local_conflicts_with_mock(tmp_path):
    from audio_style_workflow.main import build_arg_parser, build_client

    args = build_arg_parser().parse_args([str(tmp_path), "--local", "--mock"])
    with pytest.raises(ValueError):
        build_client(args)
