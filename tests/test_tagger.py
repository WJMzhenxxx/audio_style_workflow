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
