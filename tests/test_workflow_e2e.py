"""端到端集成测试：ffmpeg 生成真实 TTA 整轨，走完整个 LangChain 工作流。

覆盖：TTA 按 CUE 切分 -> mock 分析 -> 风格提取 -> 写回自定义标签，
以及 skip-scanned 扫描记录断点续跑机制。
"""
import subprocess

import pytest

from audio_style_workflow.api.base import SongAnalysisClient, SongAnalysisError
from audio_style_workflow.api.mock_client import MockSongAnalysisClient
from audio_style_workflow.models import AudioFormat
from audio_style_workflow.state_store import DEFAULT_STATE_FILENAME, ScanStateStore
from audio_style_workflow.style_parser import StyleParser
from audio_style_workflow.splitter import resolve_ffmpeg
from audio_style_workflow.tagger import get_styles
from audio_style_workflow.workflow import WorkflowConfig, run_workflow

pytestmark = pytest.mark.skipif(resolve_ffmpeg() is None, reason="本机没有 ffmpeg")

FFMPEG = resolve_ffmpeg()

# 95 秒整轨，3 轨：0-30 / 30-65 / 65-95
# 注意 CUE 时间格式是 分:秒:帧（帧率 75fps），30 秒 = 00:30:00
CUE = """\
PERFORMER "测试艺人"
TITLE "整轨专辑"
FILE "album.tta" WAVE
  TRACK 01 AUDIO
    TITLE "第一首"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "第二首"
    INDEX 01 00:30:00
  TRACK 03 AUDIO
    TITLE "第三首"
    PERFORMER "特别来宾"
    INDEX 01 01:05:00
"""


@pytest.fixture()
def library(tmp_path):
    music = tmp_path / "music"
    deep = music / "flac 部分"  # 中文目录名 + 空格，测试路径健壮性
    deep.mkdir(parents=True)

    ffmpeg = FFMPEG

    # TTA 整轨（95 秒）
    tta = music / "album.tta"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=95", "-c:a", "tta", str(tta)],
        capture_output=True, check=True,
    )
    (music / "album.cue").write_text(CUE, encoding="utf-8")

    # 直连 mp3 / flac
    mp3 = deep / "song-a.mp3"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "libmp3lame", str(mp3)],
        capture_output=True, check=True,
    )
    flac = deep / "song-b.flac"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=550:duration=3", "-c:a", "flac", str(flac)],
        capture_output=True, check=True,
    )
    return music


def _durations(paths):
    from mutagen.flac import FLAC

    return [round(FLAC(str(p)).info.length, 1) for p in paths]


def test_full_workflow_writes_tags(library):
    config = WorkflowConfig(root=library, custom_tag_key="MY_STYLE")
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)

    assert len(report.tasks) == 5  # 3 分轨 + mp3 + flac
    assert len(report.warnings) == 0
    assert report.tagged == 5
    assert report.analyzed == 5

    for task in report.tasks:
        assert task.error is None, task.error
        value = get_styles(task.path, key="MY_STYLE")
        assert value, f"{task.path} 未写入标签"
        assert task.style_parser == "json"
        assert len(task.styles) >= 2


def test_split_tracks_and_basic_tags(library):
    run_workflow(MockSongAnalysisClient(), StyleParser(), WorkflowConfig(root=library))

    names = sorted(p.name for p in library.glob("0*.flac"))
    assert names == ["01 - 第一首.flac", "02 - 第二首.flac", "03 - 第三首.flac"]

    # 切分时长：30 / 35 / 30 秒（±1s 容差）
    durations = _durations([library / "01 - 第一首.flac", library / "02 - 第二首.flac", library / "03 - 第三首.flac"])
    assert abs(durations[0] - 30) <= 1.0
    assert abs(durations[1] - 35) <= 1.0
    assert abs(durations[2] - 30) <= 1.0
    total = round(durations[0] + durations[1] + durations[2], 1)
    assert abs(total - 95) <= 1.5

    # 基础标签：专辑/标题/音轨号；第 3 轨艺人覆盖为特别来宾
    from mutagen.flac import FLAC

    tags = FLAC(str(library / "01 - 第一首.flac"))
    assert tags["TITLE"] == ["第一首"]
    assert tags["ALBUM"] == ["整轨专辑"]
    assert tags["ARTIST"] == ["测试艺人"]
    assert tags["ALBUMARTIST"] == ["测试艺人"]
    assert tags["TRACKNUMBER"] == ["1"]

    third = FLAC(str(library / "03 - 第三首.flac"))
    assert third["ARTIST"] == ["特别来宾"]


def test_skip_tagged_second_run(library):
    config = WorkflowConfig(root=library, custom_tag_key="STYLE", skip_tagged=True)
    first = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert first.tagged == 5

    second = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    # 分轨沿用已存在文件并在切分后标记 skip，直连文件在扫描时标记 skip
    assert second.analyzed == 0
    assert second.skipped == 5


def test_dry_run_does_not_write(library):
    config = WorkflowConfig(root=library, custom_tag_key="STYLE", dry_run=True)
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert report.analyzed == 5
    assert report.tagged == 0
    for task in report.tasks:
        assert task.would_write  # 只有预览值
        assert get_styles(task.path, key="STYLE") is None


def test_split_dir_option(library, tmp_path):
    out_dir = tmp_path / "分轨输出"
    config = WorkflowConfig(root=library, split_dir=out_dir, custom_tag_key="STYLE")
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert report.tagged == 5
    assert len(list(out_dir.glob("*.flac"))) == 3
    # 原目录保持整洁，没有分轨
    assert len(list(library.glob("0*.flac"))) == 0


def test_summary_report(library):
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), WorkflowConfig(root=library))
    text = "\n".join(report.summary_lines())
    assert "处理摘要" in text
    assert "歌曲总数" in text and ": 5" in text


# ------------------------------------------------------------------
# skip-scanned（扫描记录断点续跑）
# ------------------------------------------------------------------
class FailingClient(SongAnalysisClient):
    """模拟分析服务不可用。"""

    name = "always-fail"

    def analyze(self, audio_path):
        raise SongAnalysisError("服务不可用（模拟故障）")


def _skip_scanned_config(root, **kwargs) -> WorkflowConfig:
    return WorkflowConfig(root=root, skip_scanned=True, **kwargs)


def test_skip_scanned_second_run_skips_everything(library):
    config = _skip_scanned_config(library)
    first = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert first.tagged == 5
    assert (library / DEFAULT_STATE_FILENAME).is_file()
    assert first.state_path == library / DEFAULT_STATE_FILENAME

    second = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert second.analyzed == 0
    assert second.skipped == 5
    assert second.skipped_scanned == 5
    # 专辑级命中：连 TTA 切分都跳过
    assert second.albums_skipped_scanned == 1
    assert second.failed == []


def test_skip_scanned_reprocesses_changed_file(library):
    config = _skip_scanned_config(library)
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)

    # 修改一首直连歌曲（重写内容+改 mtime）——应仅它被重新分析
    import os

    changed = library / "flac 部分" / "song-a.mp3"
    changed.write_bytes(b"changed-content-must-differ-in-size-" + b"0" * 64)
    st = os.stat(changed)
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns + int(1e9)))

    third = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert third.analyzed == 1
    assert third.skipped_scanned == 4
    assert [t.path for t in third.tasks if t.analysis] == [changed]


def test_skip_scanned_failing_run_not_marked(library):
    config = _skip_scanned_config(library)
    bad = run_workflow(FailingClient(), StyleParser(), config)
    assert len(bad.failed) == 5
    assert bad.tagged == 0
    # 失败不记录：二次运行（服务恢复）仍应全量处理
    good = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert good.analyzed == 5
    assert good.tagged == 5


def test_skip_scanned_dry_run_no_state_file(library):
    config = WorkflowConfig(root=library, skip_scanned=True, dry_run=True)
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert not (library / DEFAULT_STATE_FILENAME).is_file(), "dry-run 不应产生状态文件"


def test_skip_scanned_reset_state(library):
    config = _skip_scanned_config(library)
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert (library / DEFAULT_STATE_FILENAME).is_file()

    reset_config = WorkflowConfig(root=library, skip_scanned=True, reset_state=True)
    report = run_workflow(MockSongAnalysisClient(), StyleParser(), reset_config)
    assert report.analyzed == 5, "重置记录后应全量重新处理"


def test_skip_scanned_custom_state_file(library, tmp_path):
    state_file = tmp_path / "custom" / "state.json"
    config = WorkflowConfig(root=library, skip_scanned=True, state_file=state_file)
    first = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert state_file.is_file()
    assert first.state_path == state_file

    second = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    assert second.analyzed == 0
    assert second.skipped_scanned == 5


def test_skip_scanned_album_resplit_when_tta_changes(library):
    config = _skip_scanned_config(library)
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)

    # TTA 源变化 -> 专辑指纹失效 -> 重新切分并重新分析全部分轨
    import os

    tta = library / "album.tta"
    subprocess.run(
        [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=660:duration=95", "-c:a", "tta", str(tta)],
        capture_output=True, check=True,
    )
    st = os.stat(tta)
    os.utime(tta, ns=(st.st_atime_ns, st.st_mtime_ns + int(2e9)))

    report = run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    flac_tasks = [t for t in report.tasks if t.source == tta and t.path.suffix == ".flac"]
    assert len(flac_tasks) == 3
    assert all(t.analysis for t in flac_tasks), "TTA 变化后 3 首分轨都应重新分析"
    assert report.skipped_scanned == 2  # 两首直连歌曲仍跳过


def test_summary_shows_scanned_stats(library):
    config = _skip_scanned_config(library)
    run_workflow(MockSongAnalysisClient(), StyleParser(), config)
    text = "\n".join(run_workflow(MockSongAnalysisClient(), StyleParser(), config).summary_lines())
    assert "扫描记录命中 5" in text
    assert "1 张" in text  # 整张跳过切分的专辑数


# ------------------------------------------------------------------
# BPM / Tempo / Key 打标签
# ------------------------------------------------------------------
def test_musical_tags_written_for_all_tracks(library):
    from audio_style_workflow.tagger import read_extra_tags

    class MetaClient(SongAnalysisClient):
        name = "meta"

        def analyze(self, audio_path):
            return {
                "status": "ok",
                "styles": ["Pop", "R&B"],
                "description": "BPM: 120\nTempo: Moderato\nKey: C major",
                "bpm": 120,
                "tempo": "Moderato",
                "key": "C major",
            }

    report = run_workflow(MetaClient(), StyleParser(), WorkflowConfig(root=library))
    assert report.tagged == 5
    for task in report.tasks:
        assert task.bpm == 120
        assert task.tempo == "Moderato"
        assert task.key == "C major"
        assert read_extra_tags(task.path) == {
            "bpm": "120", "tempo": "Moderato", "key": "C major", "mood": None,
            "score": None, "review": None,
        }


def test_musical_tags_can_be_disabled(library):
    from audio_style_workflow.tagger import read_extra_tags

    class MetaClient(SongAnalysisClient):
        name = "meta"

        def analyze(self, audio_path):
            return {
                "status": "ok", "styles": ["Pop"],
                "description": "BPM: 120", "bpm": 120,
            }

    config = WorkflowConfig(root=library, write_musical_tags=False)
    report = run_workflow(MetaClient(), StyleParser(), config)
    assert report.tagged == 5
    for task in report.tasks:
        assert read_extra_tags(task.path)["bpm"] is None


# ------------------------------------------------------------------
# 逐曲闭环模式
# ------------------------------------------------------------------
def test_build_workflow_per_song_shape(library):
    from audio_style_workflow.workflow import build_workflow

    wf = build_workflow(
        MockSongAnalysisClient(), StyleParser(),
        WorkflowConfig(root=library, per_song=True),
    )
    assert [step.name for step in wf.steps] == ["scan", "split_tta", "per_song"]


def test_per_song_tags_incrementally(library):
    """逐曲模式：分析下一首之前，上一首的标签已写好。"""
    seen_counts = []

    class Recorder(MockSongAnalysisClient):
        def analyze(self, audio_path):
            seen_counts.append(
                sum(1 for p in library.rglob("*")
                    if p.is_file() and get_styles(p, key="GENRE"))
            )
            return super().analyze(audio_path)

    config = WorkflowConfig(root=library, per_song=True, skip_scanned=True)
    report = run_workflow(Recorder(), StyleParser(), config)

    assert report.tagged == 5
    assert report.failed == []
    # 每进入下一首时，已完成数量单调递增
    assert seen_counts == sorted(seen_counts)
    assert seen_counts[-1] >= 4


def test_per_song_resume_after_interrupt(library):
    """逐曲模式中途中断：扫描记录已落盘，重跑 --skip-scanned 可续跑。"""
    import json

    from audio_style_workflow.state_store import DEFAULT_STATE_FILENAME

    state_file = library / DEFAULT_STATE_FILENAME

    class Boom(MockSongAnalysisClient):
        def __init__(self, limit):
            super().__init__()
            self.limit = limit
            self.calls = 0

        def analyze(self, audio_path):
            self.calls += 1
            if self.calls > self.limit:
                raise KeyboardInterrupt("模拟中断")
            return super().analyze(audio_path)

    cfg = WorkflowConfig(
        root=library, per_song=True, skip_scanned=True, state_flush_every=1,
    )
    with pytest.raises(KeyboardInterrupt):
        run_workflow(Boom(3), StyleParser(), cfg)

    assert state_file.is_file(), "中断后扫描记录应已落盘"
    saved = json.loads(state_file.read_text(encoding="utf-8"))["files"]
    assert len(saved) >= 3, saved

    report = run_workflow(
        MockSongAnalysisClient(), StyleParser(),
        WorkflowConfig(root=library, per_song=True, skip_scanned=True),
    )
    assert report.skipped_scanned >= 3
    assert report.failed == []
