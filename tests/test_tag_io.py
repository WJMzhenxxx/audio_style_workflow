"""标签导出 / 导入测试。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from mutagen.flac import FLAC
from mutagen.id3 import ID3

from audio_style_workflow.scanner import scan_directory
from audio_style_workflow.splitter import TtaSplitter, resolve_ffmpeg
from audio_style_workflow.tag_io import (
    EXPORT_VERSION,
    audio_md5,
    export_tags,
    file_md5,
    import_tags,
    load_export,
)
from audio_style_workflow.tagger import get_styles, read_extra_tags, write_extra_tags, write_styles

pytestmark = pytest.mark.skipif(resolve_ffmpeg() is None, reason="需要 ffmpeg")


def _make_audio(path: Path, *, codec: str, seconds: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", codec, str(path)],
        check=True, capture_output=True,
    )
    return path


def _tag(path: Path, *, genre: str = "Jazz; Bossa Nova", score: int = 9) -> None:
    write_styles(path, [g.strip() for g in genre.split(";")])
    write_extra_tags(
        path, bpm=88, tempo="Andante", key="D minor",
        mood="慵懒, 温暖", score=score, review="午后很舒服。",
    )


def _strip(path: Path) -> None:
    """抹掉本项目写入的标签（模拟"干净的另一份拷贝"）。"""
    if path.suffix.lower() == ".flac":
        audio = FLAC(str(path))
        for key in ("GENRE", "MOOD", "BPM", "TEMPO", "KEY", "RATING", "COMMENT"):
            audio.pop(key, None)
        audio.save()
    else:
        tags = ID3(str(path))
        for frame in ("TCON", "TMOO", "TBPM", "TKEY", "POPM", "COMM", "TXXX:RATING"):
            tags.delall(frame)
        tags.save()


def test_export_import_round_trip(tmp_path):
    lib = tmp_path / "lib"
    _make_audio(lib / "a.mp3", codec="libmp3lame")
    _make_audio(lib / "b.flac", codec="flac")
    for path in (lib / "a.mp3", lib / "b.flac"):
        _tag(path)

    out = tmp_path / "tags.json"
    report = export_tags(lib, out)
    assert report.tracks == 2 and report.with_tags == 2

    data = load_export(out)
    assert data["version"] == EXPORT_VERSION
    entry = data["tracks"][0]
    assert entry["tags"]["GENRE"] == "Jazz; Bossa Nova"
    assert entry["tags"]["RATING"] == "9"
    assert entry["md5"] and entry["audio_md5"]

    # 另一份拷贝：改名 + 清空标签（文件字节已变，只能靠音频 md5 命中）
    other = tmp_path / "other"
    other.mkdir()
    for item in data["tracks"]:
        dst = other / f"renamed_{Path(item['rel_path']).name}"
        shutil.copy2(lib / item["rel_path"], dst)
        _strip(dst)

    result = import_tags(data, other)
    assert result.applied == 2
    assert not result.missing and not result.failed

    for path in other.iterdir():
        assert get_styles(path) == "Jazz; Bossa Nova"
        info = read_extra_tags(path)
        assert info["bpm"] == "88"
        assert info["key"] == "D minor"
        assert info["mood"] == "慵懒, 温暖"
        assert info["score"] == "9"       # POPM 只有 5 档，靠 TXXX:RATING 保住精度
        assert info["review"] == "午后很舒服。"


def test_file_md5_changes_but_audio_md5_is_stable(tmp_path):
    path = _make_audio(tmp_path / "x.mp3", codec="libmp3lame")
    _tag(path)
    file_before, audio_before = file_md5(path), audio_md5(path)

    _strip(path)
    assert file_md5(path) != file_before, "改写标签后文件 md5 应当变化"
    assert audio_md5(path) == audio_before, "音频 md5 不应受标签影响"


def test_export_skips_untagged_by_default(tmp_path):
    lib = tmp_path / "lib"
    _make_audio(lib / "tagged.mp3", codec="libmp3lame")
    _make_audio(lib / "clean.mp3", codec="libmp3lame")
    _tag(lib / "tagged.mp3")

    report = export_tags(lib, tmp_path / "tags.json")
    assert report.tracks == 1 and report.skipped_untagged == 1

    report_all = export_tags(lib, tmp_path / "all.json", include_untagged=True)
    assert report_all.tracks == 2
    data = json.loads((tmp_path / "all.json").read_text(encoding="utf-8"))
    assert any(not item["tags"] for item in data["tracks"])


def test_import_dry_run_writes_nothing(tmp_path):
    lib = tmp_path / "lib"
    _make_audio(lib / "a.mp3", codec="libmp3lame")
    _tag(lib / "a.mp3")
    data = load_export(export_tags(lib, tmp_path / "tags.json").output)

    other = tmp_path / "other"
    other.mkdir()
    shutil.copy2(lib / "a.mp3", other / "a.mp3")
    _strip(other / "a.mp3")

    result = import_tags(data, other, dry_run=True)
    assert result.applied == 1 and result.dry_run
    assert get_styles(other / "a.mp3") is None


def test_import_splits_missing_tta_album(tmp_path):
    """TTA 整轨还没切分时，导入应先切分再写标签。"""
    lib = tmp_path / "album"
    lib.mkdir()
    tta = lib / "Album.tta"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=9",
         "-c:a", "tta", "-strict", "-2", str(tta)],
        check=True, capture_output=True,
    )
    (lib / "Album.cue").write_text(
        'PERFORMER "Test Artist"\nTITLE "Test Album"\nFILE "Album.tta" WAVE\n'
        '  TRACK 01 AUDIO\n    TITLE "First"\n    INDEX 01 00:00:00\n'
        '  TRACK 02 AUDIO\n    TITLE "Second"\n    INDEX 01 00:03:00\n'
        '  TRACK 03 AUDIO\n    TITLE "Third"\n    INDEX 01 00:06:00\n',
        encoding="utf-8",
    )

    album = scan_directory(lib).albums[0]
    tasks = TtaSplitter().split(album, reuse_existing=True)
    assert len(tasks) == 3
    for index, task in enumerate(tasks, 1):
        write_styles(task.path, [f"Style{index}"])
        write_extra_tags(task.path, bpm=100 + index, score=7 + index)

    data = load_export(export_tags(lib, tmp_path / "tags.json").output)
    assert data["album_count"] == 1
    assert data["tracks"][0]["album"]["tta_md5"]

    for task in tasks:
        task.path.unlink()          # 只留整轨 + CUE

    result = import_tags(data, lib)
    assert result.split_albums == 1
    assert result.split_tracks == 3
    assert result.applied == 3 and not result.missing

    for index, name in enumerate(("First", "Second", "Third"), 1):
        path = lib / f"0{index} - {name}.flac"
        assert path.is_file()
        assert get_styles(path) == f"Style{index}"
        assert read_extra_tags(path)["bpm"] == str(100 + index)


def test_import_reports_missing_entries(tmp_path):
    lib = tmp_path / "lib"
    _make_audio(lib / "a.mp3", codec="libmp3lame")
    _tag(lib / "a.mp3")
    data = load_export(export_tags(lib, tmp_path / "tags.json").output)

    empty = tmp_path / "empty"
    empty.mkdir()
    result = import_tags(data, empty)
    assert result.applied == 0 and len(result.missing) == 1
