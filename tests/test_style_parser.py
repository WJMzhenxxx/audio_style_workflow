"""风格解析器单元测试（不含 LLM 部分）。"""
import pytest

from audio_style_workflow.style_parser import (
    StyleList,
    StyleParseError,
    StyleParser,
    _extract_confidence,
    _extract_styles,
    extract_genre_rationale,
    extract_musical_meta,
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


# ----------------------------------------------------------------------
# BPM / Tempo / Key 提取
# ----------------------------------------------------------------------
def test_meta_standard_labels():
    text = "Genre: Pop, R&B\nBPM: 120\nTempo: Moderato\nKey: C major"
    assert extract_musical_meta(text) == {"bpm": 120, "tempo": "Moderato", "key": "C major"}


def test_meta_markdown_and_chinese_labels():
    text = "**BPM:** 128\n**Tempo:** 快板\n**Key:** A小调"
    meta = extract_musical_meta(text)
    assert meta["bpm"] == 128
    assert meta["tempo"] == "快板"
    assert meta["key"] == "A小调"


def test_meta_inline_bpm_and_key():
    meta = extract_musical_meta("The track is around 95 BPM in F# minor.")
    assert meta["bpm"] == 95
    assert meta["key"] == "F# minor"


def test_meta_chinese_inline_labels():
    meta = extract_musical_meta("速度：120\n节奏：慢板\n调性：降B大调")
    assert meta == {"bpm": 120, "tempo": "慢板", "key": "降B大调"}


def test_meta_tempo_value_that_is_bpm_goes_to_bpm():
    assert extract_musical_meta("Tempo: 140 BPM") == {"bpm": 140}


def test_meta_ignores_out_of_range_bpm():
    assert "bpm" not in extract_musical_meta("BPM: 9999")
    assert "bpm" not in extract_musical_meta("BPM: 3")


def test_meta_missing_returns_empty():
    assert extract_musical_meta("no metadata here") == {}
    assert extract_musical_meta("") == {}


# ----------------------------------------------------------------------
# 风格判定理由
# ----------------------------------------------------------------------
def test_rationale_labeled_line():
    text = "Pop, R&B\nBPM: 120\nWhy: 中板律动配合温暖和声，贴近 R&B 的质感。"
    assert extract_genre_rationale(text) == "中板律动配合温暖和声，贴近 R&B 的质感。"


def test_rationale_multiline_block():
    text = (
        "Rock\n"
        "Why: Distorted guitars and driving drums explain the genre.\n"
        "Also the vocals are raw.\n"
        "\n"
        "Full description follows."
    )
    assert extract_genre_rationale(text) == (
        "Distorted guitars and driving drums explain the genre. Also the vocals are raw."
    )


def test_rationale_chinese_label_and_causal_fallback():
    assert extract_genre_rationale("Jazz\n理由：使用了 swing 节奏与即兴钢琴。") == (
        "使用了 swing 节奏与即兴钢琴。"
    )
    assert extract_genre_rationale(
        "This sounds like Pop because of the catchy chorus and bright synths."
    ) == "This sounds like Pop because of the catchy chorus and bright synths."


def test_rationale_missing_returns_none():
    assert extract_genre_rationale("") is None
    assert extract_genre_rationale("Tempo: Moderato\nKey: C major") is None
