"""风格解析器单元测试（不含 LLM 部分）。"""
import pytest

from audio_style_workflow.style_parser import (
    StyleList,
    StyleParseError,
    StyleParser,
    _extract_confidence,
    _extract_styles,
)


def test_standard_styles():
    assert _extract_styles({"status": "ok", "styles": ["Pop", "R&B"]}) == ["Pop", "R&B"]


def test_style_singular_string_splits():
    assert _extract_styles({"style": "Pop / Jazz"}) == ["Pop", "Jazz"]
    assert _extract_styles({"style": "Pop, Jazz; Rock"}) == ["Pop", "Jazz", "Rock"]


def test_genre_field():
    assert _extract_styles({"genre": ["Rock"]}) == ["Rock"]


def test_nested_result():
    payload = {"result": {"analysis": {"music_genre": "City Pop"}}}
    assert _extract_styles(payload) == ["City Pop"]


def test_case_insensitive_and_dashes():
    assert _extract_styles({"Music-Style": ["Blues"]}) == ["Blues"]
    assert _extract_styles({"music_styles": "Metal"}) == ["Metal"]


def test_keep_ampersand_style():
    assert _extract_styles({"style": "R&B"}) == ["R&B"]


def test_tags_field():
    assert _extract_styles({"tags": ["lo-fi", "chill"]}) == ["lo-fi", "chill"]


def test_no_style_found():
    assert _extract_styles({"status": "ok", "message": "hi"}) == []


def test_parse_error_without_llm():
    parser = StyleParser()
    with pytest.raises(StyleParseError):
        parser.parse({"caption": "这是一首欢快的歌"})


def test_parse_json_path():
    parser = StyleParser()
    styles, source, confidence = parser.parse(
        {"status": "ok", "styles": ["Pop", "Rock", "Jazz", "Blues"], "confidence": 0.9}
    )
    assert source == "json"
    assert styles == ["Pop", "Rock", "Jazz", "Blues"]
    assert confidence == 0.9


def test_parse_limits_max_styles():
    parser = StyleParser(max_styles=2)
    styles, _, _ = parser.parse({"styles": ["a", "b", "c", "d"]})
    assert styles == ["a", "b"]


def test_confidence_extraction():
    assert _extract_confidence({"confidence": 0.5}) == 0.5
    assert _extract_confidence({"score": 1}) == 1.0
    assert _extract_confidence({"confidence": "high"}) is None
    assert _extract_confidence({"confidence": 5}) is None  # 超出 0~1 忽略


def test_strip_empty_and_dupes():
    assert _extract_styles({"styles": ["Pop", "", "  ", "Pop", "Rock"]}) == ["Pop", "Rock"]


def test_llm_fallback_with_fake_model():
    class FakeStructured:
        def invoke(self, messages):
            assert any("很摇滚" in str(m.content) for m in messages)
            return StyleList(styles=["Rockabilly", "Country"])

    class FakeLLM:
        def with_structured_output(self, schema):
            assert schema is StyleList
            return FakeStructured()

    parser = StyleParser(llm=FakeLLM())
    styles, source, confidence = parser.parse({"caption": "这首歌很摇滚，还带点乡村味"}, source_name="x.mp3")
    assert styles == ["Rockabilly", "Country"]
    assert source == "llm"
    assert confidence is None


def test_llm_fallback_empty_raises():
    class FakeStructured:
        def invoke(self, messages):
            return StyleList(styles=[])

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructured()

    parser = StyleParser(llm=FakeLLM())
    with pytest.raises(StyleParseError):
        parser.parse({"note": "无法识别"})
