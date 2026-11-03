"""核心数据模型：工作流各阶段流转的任务对象。

管道各步（scan -> split -> analyze -> parse -> tag）都会持续富化
:class:`TrackTask`，最终报告基于其完整状态生成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AudioFormat(str, Enum):
    """分析管道直接支持的音频格式（TTA 整轨会被切分为 FLAC 后进入管道）。"""

    MP3 = "mp3"
    FLAC = "flac"

    @classmethod
    def from_path(cls, path: Path) -> Optional["AudioFormat"]:
        suffix = Path(path).suffix.lower().lstrip(".")
        try:
            return cls(suffix)
        except ValueError:
            return None


@dataclass
class CueTrack:
    """CUE 中的一条轨道。"""

    number: int
    title: str = ""
    performer: str = ""
    start_sec: Optional[float] = None  # INDEX 时间（秒），相对整轨起点


@dataclass
class CueSheet:
    """一张整轨专辑的 CUE 信息。"""

    file_name: str = ""      # FILE 指令引用的音频文件名
    title: str = ""          # 专辑级 TITLE
    performer: str = ""      # 专辑级 PERFORMER
    tracks: list[CueTrack] = field(default_factory=list)


@dataclass
class TrackTask:
    """一首进入分析管道的歌曲及其全部中间状态。"""

    path: Path                                   # 待分析文件
    source: Path                                 # 来源文件（TTA 分轨时指向原始整轨）
    fmt: AudioFormat
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    track_no: Optional[int] = None
    analysis: Optional[dict[str, Any]] = None    # 分析端点原始 JSON 响应
    styles: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    style_parser: Optional[str] = None           # "json" / "llm"
    tagged: bool = False
    would_write: Optional[str] = None            # dry-run 模式下记录将要写入的值
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def has_style(self) -> bool:
        return bool(self.styles)


@dataclass
class TtaAlbum:
    """一张待切分的整轨专辑（TTA + CUE）。"""

    tta: Path
    cue: Optional[Path] = None       # 缺少 CUE 时无法切分
    sheet: Optional[CueSheet] = None
    error: Optional[str] = None      # 无法切分的原因（配对失败等）


@dataclass
class PipelineState:
    """工作流各步骤之间传递的状态容器。"""

    tasks: list[TrackTask] = field(default_factory=list)
    albums: list[TtaAlbum] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 阶段间传递的统计信息（如 albums_skipped_scanned）
    stats: dict[str, int] = field(default_factory=dict)
