"""标签写入器单元测试（需要 ffmpeg 生成测试音频，缺 ffmpeg 时跳过）。"""
import shutil
import subprocess

import pytest

from audio_style_workflow.tagger import (
    TaggerError,
    get_styles,
    read_basic_info,
    write_basic_tags,
    write_styles,
)
from audio_style_workflow.splitter import resolve_ffmpeg

pytestmark = pytest.mark.skipif(resolve_ffmpeg() is None, reason="本机没有 ffmpeg")


@pytest.fixture(scope="module")
def sample_flac(tmp_path_factory) -> object:
    out = tmp_path_factory.mktemp("audio") / "sample.flac"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "flac", str(out)],
        capture_output=True, check=True,
    )
    return out


@pytest.fixture(scope="module")
def sample_mp3(tmp_path_factory) -> object:
    out = tmp_path_factory.mktemp("audio") / "sample.mp3"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "libmp3lame", str(out)],
        capture_output=True, check=True,
    )
    return out


def test_write_and_read_flac_custom_tag(sample_flac):
    value = write_styles(sample_flac, ["Pop", "R&B"], key="MY_STYLE")
    assert value == "Pop; R&B"
    assert get_styles(sample_flac, key="MY_STYLE") == "Pop; R&B"


def test_write_and_read_mp3_custom_tag(sample_mp3):
    write_styles(sample_mp3, ["Jazz"], key="STYLE")
    assert get_styles(sample_mp3, key="STYLE") == "Jazz"


def test_custom_joiner(sample_flac):
    write_styles(sample_flac, ["A", "B"], key="X", joiner=" | ")
    assert get_styles(sample_flac, key="X") == "A | B"


def test_overwrite_existing(sample_flac):
    write_styles(sample_flac, ["Old"], key="STYLE")
    write_styles(sample_flac, ["New"], key="STYLE")
    assert get_styles(sample_flac, key="STYLE") == "New"


def test_missing_tag_returns_none(sample_flac):
    assert get_styles(sample_flac, key="NOT_EXIST") is None


def test_empty_styles_raises(sample_flac):
    with pytest.raises(TaggerError):
        write_styles(sample_flac, ["", "  "])


def test_invalid_key_raises(sample_flac):
    with pytest.raises(TaggerError):
        write_styles(sample_flac, ["Pop"], key="")
    with pytest.raises(TaggerError):
        write_styles(sample_flac, ["Pop"], key="x" * 100)


def test_write_basic_tags_flac(sample_flac):
    write_basic_tags(sample_flac, title="歌名", artist="艺人", album="专辑", track_no=7)
    info = read_basic_info(sample_flac)
    assert info["title"] == "歌名"
    assert info["artist"] == "艺人"
    assert info["album"] == "专辑"


def test_read_basic_info_mp3(sample_mp3):
    write_basic_tags(sample_mp3, title="MP3 Title", artist="A", track_no=2)
    info = read_basic_info(sample_mp3)
    assert info["title"] == "MP3 Title"
    assert info["artist"] == "A"


def test_get_styles_on_invalid_file(tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"not a flac")
    assert get_styles(bad, key="STYLE") is None  # 不应抛异常
    with pytest.raises(TaggerError):
        write_styles(bad, ["Pop"])  # 写入要如实报错


# ----------------------------------------------------------------------
# BPM / Tempo / Key 标签
# ----------------------------------------------------------------------
@pytest.fixture()
def fresh_flac(tmp_path) -> object:
    """每个用例一份干净 FLAC，避免模块级 fixture 互相污染。"""
    out = tmp_path / "meta.flac"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "flac", str(out)],
        capture_output=True, check=True,
    )
    return out


def test_write_and_read_extra_tags_flac(fresh_flac):
    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags

    written = write_extra_tags(fresh_flac, bpm=118, tempo="Moderato", key="C major")
    assert written == {"bpm": "118", "tempo": "Moderato", "key": "C major"}
    assert read_extra_tags(fresh_flac) == {
        "bpm": "118", "tempo": "Moderato", "key": "C major", "mood": None,
        "score": None, "review": None,
    }


def test_extra_tags_partial_and_empty(fresh_flac):
    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags

    assert write_extra_tags(fresh_flac, bpm=None, tempo="", key=None) == {}
    assert write_extra_tags(fresh_flac, bpm=90) == {"bpm": "90"}
    assert read_extra_tags(fresh_flac) == {
        "bpm": "90", "tempo": None, "key": None, "mood": None,
        "score": None, "review": None,
    }


def test_extra_tags_reject_invalid_bpm(fresh_flac):
    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags

    assert write_extra_tags(fresh_flac, bpm=0, tempo=None, key=None) == {}
    assert read_extra_tags(fresh_flac)["bpm"] is None


# ----------------------------------------------------------------------
# Windows 可识别的标准帧 + 氛围
# ----------------------------------------------------------------------
def test_mp3_uses_standard_id3_frames(fresh_flac, tmp_path):
    """MP3 必须写 TCON/TMOO/TBPM/TKEY，Windows 资源管理器才显示。"""
    from mutagen.id3 import ID3

    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags, write_styles

    mp3 = tmp_path / "win.mp3"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libmp3lame", str(mp3)],
        capture_output=True, check=True,
    )
    write_styles(mp3, ["Pop", "Dream Pop"])          # 默认 GENRE -> TCON
    write_extra_tags(mp3, bpm=100, tempo="Moderato",
                     key="F# major", mood="温暖, 舒缓")

    tags = ID3(str(mp3))
    assert tags.getall("TCON")[0].text == ["Pop; Dream Pop"]
    assert tags.getall("TMOO")[0].text == ["温暖, 舒缓"]
    assert tags.getall("TBPM")[0].text == ["100"]
    assert tags.getall("TKEY")[0].text == ["F# major"]
    assert get_styles(mp3) == "Pop; Dream Pop"
    assert read_extra_tags(mp3)["mood"] == "温暖, 舒缓"


def test_flac_writes_genre_and_mood(fresh_flac):
    from mutagen.flac import FLAC

    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags, write_styles

    write_styles(fresh_flac, ["Pop", "Dream Pop"])
    write_extra_tags(fresh_flac, bpm=100, mood="温暖, 舒缓")
    audio = FLAC(str(fresh_flac))
    assert audio.get("GENRE") == ["Pop; Dream Pop"]
    assert audio.get("MOOD") == ["温暖, 舒缓"]
    assert read_extra_tags(fresh_flac)["mood"] == "温暖, 舒缓"


def test_custom_key_still_uses_txxx(tmp_path):
    from mutagen.id3 import ID3

    from audio_style_workflow.tagger import get_styles, write_styles

    mp3 = tmp_path / "custom.mp3"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libmp3lame", str(mp3)],
        capture_output=True, check=True,
    )
    write_styles(mp3, ["Jazz"], key="MY_STYLE")
    assert ID3(str(mp3)).getall("TXXX:MY_STYLE")[0].text == ["Jazz"]
    assert get_styles(mp3, key="MY_STYLE") == "Jazz"


def test_score_and_review_frames(tmp_path):
    """评分 -> POPM/RATING，评价 -> COMM/COMMENT。"""
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3

    from audio_style_workflow.tagger import read_extra_tags, write_extra_tags

    mp3 = tmp_path / "s.mp3"
    flac = tmp_path / "s.flac"
    for out, codec in ((mp3, "libmp3lame"), (flac, "flac")):
        subprocess.run(
            [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:a", codec, str(out)],
            capture_output=True, check=True,
        )
        write_extra_tags(out, score=8, review="旋律抓耳，制作干净。")

    popm = ID3(str(mp3)).getall("POPM")[0]
    assert popm.rating == 196                    # 8/10 约 4 星
    assert ID3(str(mp3)).getall("COMM")[0].text == ["旋律抓耳，制作干净。"]
    assert FLAC(str(flac)).get("RATING") == ["80"]
    assert FLAC(str(flac)).get("COMMENT") == ["旋律抓耳，制作干净。"]

    info = read_extra_tags(mp3)
    assert info["score"] == "8" and info["review"] == "旋律抓耳，制作干净。"
