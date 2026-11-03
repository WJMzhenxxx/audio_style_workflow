"""ScanStateStore（扫描记录）单元测试。"""
import json
import os
import time

import pytest

from audio_style_workflow.state_store import DEFAULT_STATE_FILENAME, ScanStateStore


def _touch_change(path, size=10):
    """修改文件内容与 mtime，确保指纹变化。"""
    path.write_bytes(b"x" * size)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(1e9)))  # mtime +1s


def test_mark_and_query_track(tmp_path):
    song = tmp_path / "a.mp3"
    song.write_bytes(b"12345")
    store = ScanStateStore(tmp_path)

    assert not store.is_scanned(song)
    store.mark_scanned(song, styles=["Pop"])
    assert store.is_scanned(song)


def test_fingerprint_change_invalidates(tmp_path):
    song = tmp_path / "a.mp3"
    song.write_bytes(b"12345")
    store = ScanStateStore(tmp_path)
    store.mark_scanned(song)

    _touch_change(song, size=9)
    assert not store.is_scanned(song), "文件内容变化后应视为未扫描"


def test_save_load_roundtrip(tmp_path):
    song = tmp_path / "a.flac"
    song.write_bytes(b"12345")
    store = ScanStateStore(tmp_path)
    store.mark_scanned(song, styles=["Jazz"])
    store.save()

    other = ScanStateStore(tmp_path)
    other.load()
    assert other.is_scanned(song)
    assert other.size == 1


def test_default_path_and_custom_path(tmp_path):
    store = ScanStateStore(tmp_path)
    assert store.path == tmp_path / DEFAULT_STATE_FILENAME

    custom = tmp_path / "sub" / "my_state.json"
    store2 = ScanStateStore(tmp_path, custom)
    song = tmp_path / "a.mp3"
    song.write_bytes(b"1")
    store2.mark_scanned(song)
    store2.save()
    assert custom.is_file()


def test_corrupted_file_treated_as_empty(tmp_path):
    bad = tmp_path / DEFAULT_STATE_FILENAME
    bad.write_text("{ not json !!!", encoding="utf-8")
    store = ScanStateStore(tmp_path)
    store.load()  # 只警告不抛
    assert store.size == 0


def test_reset(tmp_path):
    song = tmp_path / "a.mp3"
    song.write_bytes(b"1")
    store = ScanStateStore(tmp_path)
    store.mark_scanned(song)
    store.save()
    assert store.path.is_file()

    store.reset()
    assert store.size == 0
    assert not store.path.is_file()
    assert not store.is_scanned(song)


def test_album_fingerprint(tmp_path):
    tta = tmp_path / "album.tta"
    cue = tmp_path / "album.cue"
    tta.write_bytes(b"tta")
    cue.write_text("FILE album.tta WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n", encoding="utf-8")

    store = ScanStateStore(tmp_path)
    store.mark_album(tta, cue)
    store.save()
    reloaded = ScanStateStore(tmp_path)
    reloaded.load()
    assert reloaded.is_album_scanned(tta, cue)

    # CUE 变化 -> 专辑视为未扫描
    _touch_change(cue, size=20)
    assert not reloaded.is_album_scanned(tta, cue)

    # TTA 变化 -> 同样
    reloaded.mark_album(tta, cue)
    _touch_change(tta, size=7)
    assert not reloaded.is_album_scanned(tta, cue)


def test_album_cue_removed_invalidates(tmp_path):
    tta = tmp_path / "a.tta"
    cue = tmp_path / "a.cue"
    tta.write_bytes(b"t")
    cue.write_text("x", encoding="utf-8")
    store = ScanStateStore(tmp_path)
    store.mark_album(tta, cue)
    assert store.is_album_scanned(tta, cue)

    cue.unlink()
    assert not store.is_album_scanned(tta, cue), "CUE 丢失后应重新处理（记录时有 CUE）"


def test_outside_root_uses_absolute_key(tmp_path):
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    song = outside_dir / "b.mp3"
    song.write_bytes(b"1")
    store = ScanStateStore(tmp_path)
    store.mark_scanned(song)
    store.save()

    reloaded = ScanStateStore(tmp_path)
    reloaded.load()
    assert reloaded.is_scanned(song)


def test_state_file_excluded_from_scan(tmp_path):
    """状态文件本身不能被当成音乐文件重复处理（扫描器只认音频后缀，这里验证落盘内容）。"""
    song = tmp_path / "a.mp3"
    song.write_bytes(b"1")
    store = ScanStateStore(tmp_path)
    store.mark_scanned(song)
    store.save()
    payload = json.loads((tmp_path / DEFAULT_STATE_FILENAME).read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert set(payload["files"].keys()) == {"rel:a.mp3"}
