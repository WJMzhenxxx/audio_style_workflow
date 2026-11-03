"""CUE sheet 解析器。

支持 UTF-8(含 BOM)/GBK/Latin-1 编码回退，处理 ``INDEX`` 的
``mm:ss:ff`` 时间格式（CD 帧率 75fps）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import CueSheet, CueTrack

#: CD-DA 帧率（CUE 时间里的帧单位）
CDDA_FRAMES_PER_SECOND = 75

_TIME_RE = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})")
_TRACK_RE = re.compile(r"^TRACK\s+(\d+)\s+(\S+)", re.IGNORECASE)
_FILE_RE = re.compile(r"^FILE\s+(.*)$", re.IGNORECASE)
_INDEX_RE = re.compile(r"^INDEX\s+(\d+)\s+(.+)$", re.IGNORECASE)
_PERFORMER_RE = re.compile(r"^PERFORMER\s+(.*)$", re.IGNORECASE)
_TITLE_RE = re.compile(r"^TITLE\s+(.*)$", re.IGNORECASE)


class CueParseError(ValueError):
    """CUE 内容无法解析。"""


def frames_to_seconds(minutes: int, seconds: int, frames: int) -> float:
    """将 CUE 时间（分:秒:帧）换算为秒。"""
    return minutes * 60 + seconds + frames / CDDA_FRAMES_PER_SECOND


def parse_time(value: str) -> float:
    m = _TIME_RE.fullmatch(value.strip())
    if not m:
        raise CueParseError(f"非法 CUE 时间格式: {value!r}")
    return frames_to_seconds(*(int(g) for g in m.groups()))


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].strip()
    return value


def _decode(raw: bytes) -> str:
    """按 UTF-8(含BOM) -> GBK -> Latin-1 顺序尝试解码。"""
    for encoding in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_cue_file(cue_path: Path) -> CueSheet:
    return parse_cue_text(_decode(Path(cue_path).read_bytes()), source=str(cue_path))


def parse_cue_text(text: str, source: str = "<memory>") -> CueSheet:
    sheet = CueSheet()
    current: Optional[CueTrack] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("REM"):
            continue

        m = _TRACK_RE.match(line)
        if m:
            current = CueTrack(number=int(m.group(1)))
            sheet.tracks.append(current)
            continue

        m = _FILE_RE.match(line)
        if m:
            rest = m.group(1).strip()
            quoted = re.match(r'"([^"]*)"', rest)
            if quoted:
                sheet.file_name = quoted.group(1)
            else:
                # FILE album.tta WAVE —— 取第一个词，丢弃文件类型描述
                parts = rest.split()
                sheet.file_name = parts[0] if parts else ""
            continue

        m = _INDEX_RE.match(line)
        if m and current is not None:
            index_no = int(m.group(1))
            try:
                sec = parse_time(m.group(2))
            except CueParseError:
                continue
            # 优先 INDEX 01（轨道真正起点）；若只有其它 INDEX 也接受第一个
            if current.start_sec is None:
                current.start_sec = sec
            elif index_no == 1:
                current.start_sec = sec
            continue

        m = _TITLE_RE.match(line)
        if m:
            value = _strip_quotes(m.group(1))
            if current is not None:
                current.title = value
            else:
                sheet.title = value
            continue

        m = _PERFORMER_RE.match(line)
        if m:
            value = _strip_quotes(m.group(1))
            if current is not None:
                current.performer = value
            else:
                sheet.performer = value
            continue

        # 其余指令（FLAGS、PREGAP、ISRC 等）忽略

    if not sheet.tracks:
        raise CueParseError(f"CUE 中没有任何 TRACK 定义: {source}")

    for track in sheet.tracks:
        if track.start_sec is None:
            track.start_sec = 0.0
    sheet.tracks.sort(key=lambda t: t.number)
    return sheet
