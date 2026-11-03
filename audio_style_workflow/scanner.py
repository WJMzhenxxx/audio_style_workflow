"""递归扫描音乐目录，建立分析任务清单。

- mp3 / flac 直接进入分析管道；
- tta 整轨必须匹配到 CUE（优先同名 ``foo.tta`` + ``foo.cue``，
  其次解析目录内未匹配的 CUE，根据其 ``FILE`` 指令回退配对），
  匹配失败则记录警告并跳过。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .cue import CueParseError, parse_cue_file
from .models import AudioFormat, PipelineState, TtaAlbum, TrackTask
from .tagger import get_styles

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".flac"}
TTA_EXT = ".tta"
CUE_EXT = ".cue"


def scan_directory(root: Path, *, skip_tag_key: Optional[str] = None) -> PipelineState:
    """扫描目录（含子目录），返回直接可分析任务 + 待切分专辑清单。

    :param root: 音乐库根目录
    :param skip_tag_key: 若给定，已经带有该自定义标签的文件将被标记
        ``skip_reason``（配合 skip-tagged 语义避免重复调用分析端点）。
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {root}")

    audio_files: list[Path] = []
    tta_files: list[Path] = []
    cue_files: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTS:
            audio_files.append(path)
        elif suffix == TTA_EXT:
            tta_files.append(path)
        elif suffix == CUE_EXT:
            cue_files.append(path)

    state = PipelineState()

    # 1) 同名 tta/cue 配对（忽略大小写、同目录）
    cue_index: dict[tuple[str, str], Path] = {}
    for cue in cue_files:
        cue_index.setdefault((str(cue.parent).lower(), cue.stem.lower()), cue)

    matched: dict[Path, Path] = {}
    unmatched_tta: list[Path] = []
    for tta in tta_files:
        cue = cue_index.get((str(tta.parent).lower(), tta.stem.lower()))
        if cue is not None:
            matched[tta] = cue
        else:
            unmatched_tta.append(tta)

    # 2) FILE 指令回退配对：用未匹配的 CUE 内容指向来判断
    if unmatched_tta:
        unused = [c for c in cue_files if c not in set(matched.values())]
        parsed_sheets: dict[Path, Optional[object]] = {}
        for cue in unused:
            try:
                parsed_sheets[cue] = parse_cue_file(cue)
            except (CueParseError, OSError) as exc:
                parsed_sheets[cue] = None
                state.warnings.append(f"[cue] {cue.name}: 解析失败（{exc}），已忽略")

        for tta in unmatched_tta:
            for cue, sheet in parsed_sheets.items():
                if sheet is None or cue in set(matched.values()):
                    continue
                if sheet.file_name and Path(sheet.file_name).name.lower() == tta.name.lower():
                    matched[tta] = cue
                    break

    for tta in tta_files:
        cue = matched.get(tta)
        album = TtaAlbum(tta=tta, cue=cue)
        if cue is None:
            album.error = "未找到对应 CUE（需要同名 .cue，或 CUE 的 FILE 指令指向该文件）"
            state.warnings.append(f"[tta] {tta.name}: {album.error}")
        state.albums.append(album)

    # 3) 直接可分析的文件
    for path in audio_files:
        fmt = AudioFormat.from_path(path)
        if fmt is None:  # pragma: no cover - 扩展名过滤已保证
            continue
        task = TrackTask(path=path, source=path, fmt=fmt)
        if skip_tag_key:
            existing = get_styles(path, key=skip_tag_key)
            if existing:
                task.skip_reason = f"已有自定义标签 {skip_tag_key!r}={existing!r}"
        state.tasks.append(task)

    logger.info(
        "扫描完成: %d 个直连音频文件, %d 张 TTA 整轨（%d 张配对 CUE）, %d 条警告",
        len(audio_files), len(tta_files), len(matched), len(state.warnings),
    )
    return state
