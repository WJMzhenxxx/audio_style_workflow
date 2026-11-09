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


def test_styles_strip_instruction_echo():
    """模型把 prompt 指令原样接续时，要能从冒号后取到标签。"""
    from audio_style_workflow.style_parser import extract_styles_from_text

    text = "its musical styles/genres as a short comma-separated list: Dream Pop, Ambient Pop, Synth Pop\nBPM: 100"
    assert extract_styles_from_text(text) == ["Dream Pop", "Ambient Pop", "Synth-Pop"]
    # 短指令字段名也要剥掉
    assert extract_styles_from_text("list: Dream Pop, Synth Pop") == ["Dream Pop", "Synth-Pop"]

    # 单字段（冒号前只有一两个词）不应被误切
    assert extract_styles_from_text("Pop, R&B, Jazz") == ["Pop", "R&B", "Jazz"]


# ----------------------------------------------------------------------
# 中文 prompt / 中文输出解析
# ----------------------------------------------------------------------
def test_analysis_prompt_languages():
    from audio_style_workflow.style_parser import get_analysis_prompt

    zh = get_analysis_prompt("zh")
    en = get_analysis_prompt("en")
    assert "中文" in zh and "中文" not in en
    # 风格名 / Tempo / Key 仍要求英文，只有 Why 与正文用中文
    assert "英文" in zh
    assert "中文" in zh.split("Why", 1)[1]
    assert zh != en
    with pytest.raises(ValueError):
        get_analysis_prompt("fr")


def test_chinese_output_parsed():
    from audio_style_workflow.style_parser import (
        extract_genre_rationale,
        extract_musical_meta,
        extract_styles_from_text,
    )

    text = (
        "第1行：流行, 节奏布鲁斯, 爵士\n"
        "BPM: 100\n"
        "Tempo: 中板\n"
        "Key: C大调\n"
        "Why: 温暖的和声进行与中板律动体现了节奏布鲁斯的特征。\n"
        "\n"
        "整首曲子以钢琴为主导，情绪舒缓。"
    )
    assert extract_styles_from_text(text) == ["流行", "节奏布鲁斯", "爵士"]
    assert extract_musical_meta(text) == {
        "bpm": 100, "tempo": "中板", "key": "C大调",
    }
    assert extract_genre_rationale(text).startswith("温暖的和声")


def test_instruction_echo_does_not_leak_into_labels():
    from audio_style_workflow.style_parser import extract_styles_from_text

    echo = "第1行只写风格标签，用逗号分隔，不要加前缀：氛围音乐, 后摇"
    assert extract_styles_from_text(echo) == ["氛围音乐", "后摇"]
    assert extract_styles_from_text("1. 流行, 摇滚") == ["流行", "摇滚"]


def test_extract_mood():
    from audio_style_workflow.style_parser import extract_mood

    assert extract_mood("Mood: Warm, Dreamy\nBPM: 100") == "Warm, Dreamy"
    assert extract_mood("氛围：温暖, 舒缓\nWhy: ...") == "温暖, 舒缓"
    assert extract_mood("情绪: 忧郁") == "忧郁"
    assert extract_mood("BPM: 100\nKey: C major") is None
    assert extract_mood("") is None


def test_extract_score_normalizes_to_ten():
    from audio_style_workflow.style_parser import extract_score

    assert extract_score("Score: 8") == 8
    assert extract_score("评分：8/10") == 8
    assert extract_score("rating: 4/5") == 8      # 按分母换算
    assert extract_score("Score: 0") == 0
    assert extract_score("Score: 15") is None     # 超出分母
    assert extract_score("BPM: 100") is None
    assert extract_score("") is None


def test_extract_review():
    from audio_style_workflow.style_parser import extract_review

    assert extract_review("Review: 旋律抓耳，制作干净。") == "旋律抓耳，制作干净。"
    assert extract_review("评价：一首很棒的单曲") == "一首很棒的单曲"
    assert extract_review("comment: solid track") == "solid track"
    assert extract_review("Mood: 温暖") is None


def test_genre_alias_normalization():
    from audio_style_workflow.style_parser import extract_styles_from_text, normalize_genre

    assert normalize_genre("kpop") == "K-Pop"
    assert normalize_genre("Korean Pop") == "K-Pop"
    assert normalize_genre("citypop") == "City Pop"
    assert normalize_genre("Rnb") == "R&B"
    assert normalize_genre("Dream Pop") == "Dream Pop"
    assert extract_styles_from_text("kpop, Citypop, rnb") == ["K-Pop", "City Pop", "R&B"]


def test_prompts_do_not_anchor_on_specific_scenes():
    """回归：曾在示例里点名 K-Pop，导致模型对每首都输出 K-Pop。"""
    from audio_style_workflow.style_parser import get_analysis_prompt

    for lang in ("zh", "en"):
        prompt = get_analysis_prompt(lang)
        lowered = prompt.lower()
        for scene in ("k-pop", "kpop", "city pop", "j-pop", "mandopop"):
            assert scene not in lowered, (lang, scene)
        assert "1-3" in prompt          # 限制标签数量，避免凑数


def test_styles_from_tags_line_after_description():
    """描述在前、TAGS 摘要在后：必须能取到标签行。"""
    from audio_style_workflow.style_parser import extract_styles_from_text

    text = (
        "这是一首明亮的中速电子流行曲，合成器铺底与女声交织。\n"
        "TAGS: Dance-Pop, Synth-Pop\n"
        "BPM: 124\n"
    )
    assert extract_styles_from_text(text) == ["Dance-Pop", "Synth-Pop"]
    # 中文标签行同样支持
    assert extract_styles_from_text("描述……\n风格标签：Dream Pop，Synth-Pop") == ["Dream Pop", "Synth-Pop"]


def test_prompt_is_description_first():
    """对齐官方用法：先描述再摘要，而不是第一行就压标签。"""
    from audio_style_workflow.style_parser import get_analysis_prompt

    zh = get_analysis_prompt("zh")
    assert zh.index("描述") < zh.index("TAGS:")
    assert "风格与速度" in zh          # 沿用官方 demo 的提问维度


def test_genre_case_and_concat_normalized():
    """模型常输出小写/连写：统一成 Title Case 并把 poprock 合并成 Pop-Rock。"""
    from audio_style_workflow.style_parser import extract_styles_from_text, normalize_genre

    assert normalize_genre("poprock") == "Pop-Rock"
    assert normalize_genre("rock") == "Rock"
    assert normalize_genre("progressive house") == "Progressive House"
    assert normalize_genre("EDM") == "EDM"          # 缩写保持大写
    assert extract_styles_from_text("TAGS: poprock, rock, pop") == ["Pop-Rock", "Rock", "Pop"]


def test_concat_genre_split():
    from audio_style_workflow.style_parser import normalize_genre

    assert normalize_genre("progressivehouse") == "Progressive House"
    assert normalize_genre("ambientpop") == "Ambient Pop"
    assert normalize_genre("vocaltrance") == "Vocal Trance"


def test_prose_falls_back_to_description_scan():
    """模型只给描述、没给 TAGS 行时，从描述里兜底提取（不能把整句当标签）。"""
    from audio_style_workflow.style_parser import extract_styles_from_text

    prose = (
        "这是一首典型的渐进式Trance作品，整体风格融合了旋律化的合成器编排"
        "与强烈的节奏推进感，展现出2010年代中期现代Trance音乐的标志性特征。"
    )
    assert extract_styles_from_text(prose) == ["Trance"]


def test_tempo_label_from_bpm():
    from audio_style_workflow.style_parser import tempo_label_from_bpm

    assert tempo_label_from_bpm(100) == "Andante"
    assert tempo_label_from_bpm(125) == "Allegro"
    assert tempo_label_from_bpm(136) == "Allegro"
    assert tempo_label_from_bpm(50) == "Largo"
    assert tempo_label_from_bpm(180) == "Presto"


#: 官方 sections caption 格式的模型输出（MOSS 训练分布内）
SECTIONS_OUTPUT = """这是一首明亮而富有推进力的流行摇滚作品，副歌极具爆发力。

**Genre**: Pop-Rock, Electropop
**Tempo**: 125 BPM / 中速
**Harmony & Key**: G major，主歌 I–V–vi–IV 循环
**Instrument**: 合成器铺底、失真吉他、四拍底鼓、清亮女声
**Song Structure & Dynamics**: 主歌克制、预副歌递进、副歌全开
**Vocals & Lyrics**: 女声主唱，层次和声
**Mood & Theme**: 激昂, 希望, 自我觉醒
**Score**: 8
**Review**: 制作精良、副歌抓耳的一首流行摇滚圣歌。
**Why**: 失真吉他 + 四拍底鼓 + 强副歌属于流行摇滚。
"""


def test_sections_format_styles():
    from audio_style_workflow.style_parser import extract_styles_from_text

    assert extract_styles_from_text(SECTIONS_OUTPUT) == ["Pop-Rock", "Electropop"]


def test_sections_format_meta():
    """**Tempo**: 125 BPM / 中速 与 **Harmony & Key**: 要能拆出 125 / 中速 / G major。"""
    from audio_style_workflow.style_parser import extract_musical_meta

    meta = extract_musical_meta(SECTIONS_OUTPUT)
    assert meta["bpm"] == 125
    assert meta["tempo"] == "中速"
    assert meta["key"] == "G major"


def test_sections_format_mood_score_review_why():
    from audio_style_workflow.style_parser import (
        extract_genre_rationale,
        extract_mood,
        extract_review,
        extract_score,
    )

    assert extract_mood(SECTIONS_OUTPUT) == "激昂, 希望, 自我觉醒"
    assert extract_score(SECTIONS_OUTPUT) == 8
    assert extract_review(SECTIONS_OUTPUT).startswith("制作精良")
    assert extract_genre_rationale(SECTIONS_OUTPUT).startswith("失真吉他")


def test_sections_en_output():
    from audio_style_workflow.style_parser import (
        extract_mood,
        extract_musical_meta,
        extract_styles_from_text,
    )

    text = (
        "**Genre**: K-Pop, Dance-Pop\n"
        "**Tempo**: 124 BPM / fast\n"
        "**Harmony & Key**: G major\n"
        "**Mood & Theme**: Bright, Uplifting\n"
    )
    assert extract_styles_from_text(text) == ["K-Pop", "Dance-Pop"]
    meta = extract_musical_meta(text)
    assert meta["bpm"] == 124 and meta["tempo"].lower() == "fast"
    assert meta["key"] == "G major"
    assert extract_mood(text) == "Bright, Uplifting"


def test_prompt_uses_official_section_headers():
    """对齐官方 caption 的 sections 表头。"""
    from audio_style_workflow.style_parser import get_analysis_prompt

    for lang in ("zh", "en"):
        prompt = get_analysis_prompt(lang)
        for header in ("Genre", "Tempo", "Harmony & Key", "Instrument",
                       "Song Structure & Dynamics", "Vocals & Lyrics",
                       "Mood & Theme"):
            assert header in prompt, (lang, header)


def test_mood_long_phrase_is_trimmed():
    """Mood & Theme 常写成整句解释，写标签前要裁成短词。"""
    from audio_style_workflow.style_parser import extract_mood

    assert extract_mood("**Mood & Theme**: 明亮、现代、充满能量的电子流行氛围") == "明亮, 现代"
    assert extract_mood("**Mood & Theme**: 梦幻、忧郁、浪漫，带有 80 年代合成器流行乐的怀旧感。") == "梦幻, 忧郁, 浪漫"
    assert extract_mood("**Mood & Theme**: 梦幻、空灵、略带忧郁的电子氛围。") == "梦幻, 空灵"
    assert extract_mood("**Mood & Theme**: uplifting, hopeful, energetic") == "uplifting, hopeful, energetic"


def test_prompt_puts_why_before_tempo():
    """Why 放在最后时模型有 6/9 次整段漏写，挪到 Genre 之后。"""
    from audio_style_workflow.style_parser import get_analysis_prompt

    for lang in ("zh", "en"):
        prompt = get_analysis_prompt(lang)
        assert prompt.index("**Why**") < prompt.index("**Tempo**") < prompt.index("**Review**")
    assert "用中文写" in get_analysis_prompt("zh")
