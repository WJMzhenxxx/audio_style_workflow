"""用 ffmpeg 将 TTA 整轨按 CUE 切分为 FLAC 分轨。

TTA 与 FLAC 均为无损编码，因此 ``tta -> flac`` 逐轨转码无质量损失。
切分使用输出级精确 seek（``-ss``/``-t`` 放在 ``-i`` 之后），
并用 ``.part`` 临时文件防止半成品。
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .cue import parse_cue_file
from .models import AudioFormat, TtaAlbum, TrackTask
from .tagger import write_basic_tags

logger = logging.getLogger(__name__)

_DEFAULT_FFMPEG = "ffmpeg"  # 环境变量 FFMPEG_PATH 可覆盖


def resolve_ffmpeg(ffmpeg: Optional[str] = None) -> Optional[str]:
    """解析 ffmpeg 可执行文件路径，找不到返回 None。"""
    candidate = str(ffmpeg or _DEFAULT_FFMPEG or "ffmpeg")
    if not candidate:
        return None
    if Path(candidate).is_file():
        return candidate
    return shutil.which(candidate)


def sanitize_filename(name: str, maxlen: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(". ")
    return cleaned[:maxlen].rstrip()


class SplitError(RuntimeError):
    """TTA 切分失败。"""


class TtaSplitter:
    """按 CUE 把 TTA 整轨切成带基础标签的 FLAC 分轨。"""

    def __init__(self, ffmpeg: Optional[str] = None) -> None:
        import os

        candidate = ffmpeg or os.environ.get("FFMPEG_PATH") or _DEFAULT_FFMPEG
        resolved = resolve_ffmpeg(candidate)
        if resolved is None:
            raise SplitError(
                f"找不到 ffmpeg（尝试: {candidate!r}）。请安装 ffmpeg 或用 --ffmpeg / 环境变量 FFMPEG_PATH 指定路径"
            )
        self._ffmpeg = resolved
        self._log = logger

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def split(
        self,
        album: TtaAlbum,
        *,
        out_dir: Optional[Path] = None,
        reuse_existing: bool = True,
    ) -> list[TrackTask]:
        """切分一张专辑，返回分轨 Task 列表。

        :param reuse_existing: 输出文件已存在时直接复用（断点续跑），不重复切分。
        :raises SplitError: 缺 CUE、CUE 无效或 ffmpeg 执行失败。
        """
        if album.cue is None:
            raise SplitError("缺少 CUE 文件，无法切分")
        sheet = album.sheet or parse_cue_file(album.cue)
        album.sheet = sheet

        out_dir = Path(out_dir) if out_dir else album.tta.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        starts = [t.start_sec or 0.0 for t in sheet.tracks]
        tasks: list[TrackTask] = []
        for idx, track in enumerate(sheet.tracks):
            start = starts[idx]
            end = starts[idx + 1] if idx + 1 < len(starts) else None
            if end is not None and end <= start:
                self._log.warning(
                    "[tta] %s 第 %d 轨 INDEX 时间未递增（%s -> %s），该轨时间可能异常",
                    album.tta.name, track.number, start, end,
                )

            target = out_dir / self._output_name(track)
            if reuse_existing and target.is_file() and target.stat().st_size > 0:
                self._log.info("[tta] 分轨已存在，复用: %s", target.name)
            else:
                self._extract(album.tta, start, end, target)

            artist = track.performer or sheet.performer or None
            write_basic_tags(
                target,
                title=track.title or target.stem,
                artist=artist,
                album_artist=sheet.performer or None,
                album=sheet.title or None,
                track_no=track.number,
            )
            tasks.append(
                TrackTask(
                    path=target,
                    source=album.tta,
                    fmt=AudioFormat.FLAC,
                    title=track.title or None,
                    artist=artist,
                    album=sheet.title or None,
                    track_no=track.number,
                )
            )
        self._log.info("[tta] %s 切分完成，共 %d 轨", album.tta.name, len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    @staticmethod
    def _output_name(track) -> str:
        title = sanitize_filename(track.title)
        stem = f"{track.number:02d} - {title}" if title else f"Track{track.number:02d}"
        return f"{stem}.flac"

    def _extract(self, tta: Path, start: float, end: Optional[float], target: Path) -> None:
        args = ["-hide_banner", "-loglevel", "error", "-y", "-i", str(tta), "-vn"]
        if start > 0:
            args += ["-ss", f"{start:.6f}"]
        if end is not None:
            args += ["-t", f"{max(end - start, 0.0):.6f}"]
        args += ["-map", "0:a:0", "-c:a", "flac", "-f", "flac", "out"]

        tmp = target.with_name(target.name + ".part")
        cmd = [self._ffmpeg, *args[:-1], str(tmp)]
        self._log.debug("ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not tmp.is_file():
            detail = proc.stderr.decode("utf-8", "replace").strip()[-600:]
            raise SplitError(f"ffmpeg 切分失败（退出码 {proc.returncode}）: {detail or '无错误输出'}")
        tmp.replace(target)
