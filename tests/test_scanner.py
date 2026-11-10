"""目录扫描（TTA/CUE 配对）单元测试。"""
import pytest

from audio_style_workflow.scanner import scan_directory


def _make(path, content: bytes | str = b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


CUE = (
    'PERFORMER "artist"\nTITLE "album"\nFILE "album.tta" WAVE\n'
    "  TRACK 01 AUDIO\n    TITLE \"one\"\n    INDEX 01 00:00:00\n"
    "  TRACK 02 AUDIO\n    TITLE \"two\"\n    INDEX 01 00:00:30\n"
)


def test_scan_basic_files(tmp_path):
    _make(tmp_path / "a.mp3")
    _make(tmp_path / "sub" / "b.flac")
    _make(tmp_path / "sub" / "deep" / "c.mp3")
    _make(tmp_path / "readme.txt")

    state = scan_directory(tmp_path)
    assert len(state.tasks) == 3
    assert state.tasks[0].fmt.value == "mp3"
    assert {t.path.name for t in state.tasks} == {"a.mp3", "b.flac", "c.mp3"}
    assert state.albums == []


def test_scan_pairs_same_name_cue(tmp_path):
    _make(tmp_path / "album.tta", b"tta-bytes")
    _make(tmp_path / "album.cue", CUE)

    state = scan_directory(tmp_path)
    assert len(state.albums) == 1
    assert state.albums[0].cue is not None and state.albums[0].cue.name == "album.cue"
    assert state.albums[0].error is None
    assert state.warnings == []


def test_scan_pairs_case_insensitive(tmp_path):
    _make(tmp_path / "Album.tta", b"tta")
    _make(tmp_path / "album.CUE", CUE)
    state = scan_directory(tmp_path)
    assert state.albums[0].cue is not None


def test_scan_pairs_by_file_directive(tmp_path):
    # 不同名：整轨叫 disc.tta，cue 叫 meta.cue 但 FILE 指向 disc.tta
    _make(tmp_path / "disc.tta", b"tta")
    _make(tmp_path / "meta.cue", CUE.replace("album.tta", "disc.tta"))
    state = scan_directory(tmp_path)
    assert state.albums[0].cue is not None
    assert state.albums[0].error is None


def test_scan_unpaired_tta_warns(tmp_path):
    _make(tmp_path / "lone.tta", b"tta")
    state = scan_directory(tmp_path)
    assert state.albums[0].cue is None
    assert state.albums[0].error is not None
    assert any("lone.tta" in w for w in state.warnings)


def test_scan_skips_invalid_cue(tmp_path):
    _make(tmp_path / "broken.cue", "TITLE \"no tracks here\"\n")
    _make(tmp_path / "x.tta", b"tta")
    state = scan_directory(tmp_path)
    assert state.albums[0].cue is None
    assert state.warnings  # 解析失败的 cue 有告警


def test_scan_skip_tagged_flags_tasks(tmp_path):
    # 手工构造一个带 STYLE 标签的"假 flac"：用最小 FLAC 流太麻烦，直接借 ffmpeg 生成
    from audio_style_workflow.splitter import resolve_ffmpeg
    import subprocess

    ffmpeg = resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip("无 ffmpeg")
    flac = tmp_path / "tagged.flac"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "flac", str(flac)],
        capture_output=True, check=True,
    )
    from audio_style_workflow.tagger import write_styles
    write_styles(flac, ["Old Style"], key="STYLE")

    state = scan_directory(tmp_path, skip_tag_key="STYLE")
    assert state.tasks[0].skip_reason is not None
    state2 = scan_directory(tmp_path)  # 不开启 skip 时不标记
    assert state2.tasks[0].skip_reason is None


def test_scanner_includes_wav(tmp_path):
    """WAV（以及 .wave）也要被扫描进来。"""
    from audio_style_workflow.scanner import scan_directory

    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "b.wave").write_bytes(b"RIFF")
    (tmp_path / "c.mp3").write_bytes(b"x")
    (tmp_path / "d.flac").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    state = scan_directory(tmp_path)
    names = sorted(task.path.name for task in state.tasks)
    assert names == ["a.wav", "b.wave", "c.mp3", "d.flac"]
    assert len(state.warnings) == 0
