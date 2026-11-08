"""日志模块测试：latest.log 全量日志 + details 逐曲明细。"""
import json
import logging

import pytest

from audio_style_workflow.api.mock_client import MockSongAnalysisClient
from audio_style_workflow.logging_setup import (
    AnalysisDetailRecorder,
    DEFAULT_LOG_DIR,
    DETAILS_DIR_NAME,
    LATEST_LOG_NAME,
    describe_request,
    prepare_log_dirs,
    setup_logging,
)
from audio_style_workflow.style_parser import StyleParser
from audio_style_workflow.workflow import WorkflowConfig, run_workflow


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    # 还原根日志，避免测试间互相污染
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_asw_managed", False):
            root.removeHandler(h)
            h.close()
    root.setLevel(logging.WARNING)


def test_prepare_log_dirs_creates_structure(tmp_path):
    paths = prepare_log_dirs(tmp_path / "mylog")
    assert paths.root.is_dir()
    assert paths.details.is_dir()
    assert paths.latest.name == LATEST_LOG_NAME
    assert paths.details.name == DETAILS_DIR_NAME


def test_prepare_log_dirs_default_is_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 在临时目录下验证默认值，避免污染工作区
    paths = prepare_log_dirs()
    assert paths.root == DEFAULT_LOG_DIR
    assert paths.details.is_dir()


def test_setup_logging_writes_latest_log(tmp_path):
    paths = setup_logging(tmp_path / "log", console=False)
    logging.getLogger("audio_style_workflow.test").info("hello-from-test")
    logging.getLogger("audio_style_workflow.test").debug("debug-should-not-appear")
    for h in logging.getLogger().handlers:
        h.flush()

    text = paths.latest.read_text(encoding="utf-8")
    assert "hello-from-test" in text
    # 默认 INFO 级别，调试日志不落盘
    assert "debug-should-not-appear" not in text
    assert "INFO" in text


def test_setup_logging_verbose_enables_debug(tmp_path):
    paths = setup_logging(tmp_path / "log", verbose=True, console=False)
    logging.getLogger("audio_style_workflow.test").debug("debug-line")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "debug-line" in paths.latest.read_text(encoding="utf-8")


def test_setup_logging_overwrites_each_run(tmp_path):
    log_dir = tmp_path / "log"
    setup_logging(log_dir, console=False)
    logging.getLogger("audio_style_workflow.test").info("run-one")
    for h in logging.getLogger().handlers:
        h.flush()

    setup_logging(log_dir, console=False)  # 第二次运行应覆盖
    logging.getLogger("audio_style_workflow.test").info("run-two")
    for h in logging.getLogger().handlers:
        h.flush()

    text = (log_dir / LATEST_LOG_NAME).read_text(encoding="utf-8")
    assert "run-two" in text
    assert "run-one" not in text


def test_recorder_path_is_stable_and_sanitized(tmp_path):
    rec = AnalysisDetailRecorder(tmp_path / "log")
    a = tmp_path / "a b:c?.mp3"
    p1 = rec.path_for(a)
    p2 = rec.path_for(a)
    assert p1 == p2
    assert p1.parent == rec.details_dir
    assert p1.suffix == ".json"
    # 非法文件名字符被替换
    assert ":" not in p1.name and "?" not in p1.name


def test_recorder_writes_request_and_response(tmp_path):
    rec = AnalysisDetailRecorder(tmp_path / "log")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake-audio-bytes")

    target = rec.record(
        audio,
        request={"audio": str(audio), "client": "mock"},
        response={"status": "ok", "styles": ["Pop"]},
        error=None,
        elapsed=0.123,
    )
    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["request"]["client"] == "mock"
    assert data["response"]["styles"] == ["Pop"]
    assert data["error"] is None
    assert data["elapsed_sec"] == 0.123
    assert "recorded_at" in data


def test_recorder_records_error(tmp_path):
    rec = AnalysisDetailRecorder(tmp_path / "log")
    audio = tmp_path / "bad.mp3"
    audio.write_bytes(b"x")
    target = rec.record(audio, request={}, response=None, error="服务不可用")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["error"] == "服务不可用"
    assert data["response"] is None


def test_describe_request_collects_client_metadata(tmp_path):
    from audio_style_workflow.api.sglang_client import SglangMusicClient

    audio = tmp_path / "x.flac"
    audio.write_bytes(b"abc")
    client = SglangMusicClient(endpoint="http://h:30000", api_mode="chat",
                              thinking_budget=50, max_new_tokens=1024)
    info = describe_request(audio, client)
    assert info["client"] == "sglang-music"
    assert info["endpoint"] == "http://h:30000"
    assert info["api_mode"] == "chat"
    assert info["thinking_budget"] == 50
    assert info["max_new_tokens"] == 1024
    assert info["audio_size"] == 3
    assert "prompt" in info


def test_workflow_writes_detail_per_analyzed_song(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a.mp3", "b.mp3"):
        (music / name).write_bytes(b"ID3-fake-" + name.encode())

    config = WorkflowConfig(root=music, log_dir=tmp_path / "log")
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)

    details = tmp_path / "log" / DETAILS_DIR_NAME
    files = sorted(details.glob("*.json"))
    assert len(files) == 2, f"应有 2 份明细，实际 {files}"
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["response"]["status"] == "ok"
        assert data["request"]["client"] == "mock"
        assert data["audio"].endswith(".mp3")


def test_workflow_without_log_dir_writes_nothing(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "a.mp3").write_bytes(b"ID3-fake")

    config = WorkflowConfig(root=music)  # log_dir 默认 None
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)

    assert not (tmp_path / "log").exists()


def test_workflow_records_failed_analysis(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "a.mp3").write_bytes(b"ID3-fake")

    from audio_style_workflow.api.base import SongAnalysisClient, SongAnalysisError

    class Fail(SongAnalysisClient):
        name = "fail"

        def analyze(self, audio_path):
            raise SongAnalysisError("boom")

    config = WorkflowConfig(root=music, log_dir=tmp_path / "log")
    report = run_workflow(Fail(), StyleParser(), config)
    assert len(report.failed) == 1

    files = list((tmp_path / "log" / DETAILS_DIR_NAME).glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "boom" in data["error"]


def test_workflow_concurrent_details_all_written(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for i in range(6):
        (music / f"s{i}.mp3").write_bytes(f"ID3-fake-{i}".encode())

    config = WorkflowConfig(root=music, log_dir=tmp_path / "log", concurrency=4)
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert report.analyzed == 6

    files = list((tmp_path / "log" / DETAILS_DIR_NAME).glob("*.json"))
    assert len(files) == 6
    # 每个文件都能正常解析（无写入撕裂）
    for f in files:
        json.loads(f.read_text(encoding="utf-8"))
