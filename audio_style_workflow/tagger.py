"""把风格解析结果写回音频文件的自定义标签（基于 mutagen）。

- FLAC : VorbisComment 自由字段（默认键 ``STYLE``）
- MP3  : ID3 ``TXXX:<key>`` 用户自定义文本帧（默认 ``TXXX:STYLE``）

键名/分隔符均可自定义。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError, TALB, TIT2, TPE1, TPE2, TRCK, TXXX

from .models import AudioFormat

#: 默认自定义风格标签键名
DEFAULT_CUSTOM_KEY = "STYLE"
#: 多风格默认连接符（写成单值，避免个别播放器把多值字段显示成两行）
DEFAULT_JOINER = "; "

_MAX_KEY_LEN = 60


class TaggerError(RuntimeError):
    """写标签失败。"""


def _check_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        raise TaggerError("自定义标签键不能为空")
    if len(key) > _MAX_KEY_LEN:
        raise TaggerError(f"自定义标签键过长（>{_MAX_KEY_LEN} 字符）")
    if "\x00" in key:
        raise TaggerError("自定义标签键不能包含 NUL 字符")
    return key


def write_styles(
    path: Path,
    styles: Sequence[str],
    *,
    key: str = DEFAULT_CUSTOM_KEY,
    joiner: str = DEFAULT_JOINER,
) -> str:
    """将风格列表写入自定义标签，返回实际写入的值。

    :raises TaggerError: 格式不受支持 / 无有效风格 / 底层写入失败
    """
    path = Path(path)
    key = _check_key(key)
    unique: list[str] = []
    for s in styles:
        text = str(s).strip()
        if text and text not in unique:
            unique.append(text)
    if not unique:
        raise TaggerError(f"{path.name}: 没有可写入的风格值")
    value = joiner.join(unique)

    fmt = AudioFormat.from_path(path)
    try:
        if fmt is AudioFormat.FLAC:
            audio = FLAC(str(path))
            audio[key] = [value]
            audio.save()
        elif fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                tags = ID3()
            tags.setall(f"TXXX:{key}", [TXXX(encoding=3, desc=key, text=value)])
            tags.save(str(path), v2_version=4)
        else:
            raise TaggerError(f"暂不支持向 {path.suffix or '<无后缀>'} 格式写标签: {path}")
    except TaggerError:
        raise
    except Exception as exc:  # mutagen 各种底层异常
        raise TaggerError(f"写 {path.name} 的标签失败: {exc}") from exc
    return value


def get_styles(path: Path, *, key: str = DEFAULT_CUSTOM_KEY) -> Optional[str]:
    """读取自定义风格标签；不存在或读取失败（如文件损坏）返回 None。"""
    path = Path(path)
    key = _check_key(key)
    fmt = AudioFormat.from_path(path)
    try:
        if fmt is AudioFormat.FLAC:
            values = FLAC(str(path)).get(key)
            if isinstance(values, list) and values:
                return str(values[0])
            return None
        if fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                return None
            frames = tags.getall(f"TXXX:{key}")
            if frames and frames[0].text:
                return str(frames[0].text[0])
            return None
    except Exception:
        return None
    return None


def write_basic_tags(
    path: Path,
    *,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    album_artist: Optional[str] = None,
    track_no: Optional[int] = None,
) -> None:
    """写入常规基础标签（用于 TTA 切分出的分轨）。"""
    path = Path(path)
    fmt = AudioFormat.from_path(path)
    try:
        if fmt is AudioFormat.FLAC:
            audio = FLAC(str(path))
            if title is not None:
                audio["TITLE"] = [title]
            if artist is not None:
                audio["ARTIST"] = [artist]
            if album_artist is not None:
                audio["ALBUMARTIST"] = [album_artist]
            if album is not None:
                audio["ALBUM"] = [album]
            if track_no is not None:
                audio["TRACKNUMBER"] = [str(track_no)]
            audio.save()
        elif fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                tags = ID3()
            if title is not None:
                tags.setall("TIT2", [TIT2(encoding=3, text=title)])
            if artist is not None:
                tags.setall("TPE1", [TPE1(encoding=3, text=artist)])
            if album_artist is not None:
                tags.setall("TPE2", [TPE2(encoding=3, text=album_artist)])
            if album is not None:
                tags.setall("TALB", [TALB(encoding=3, text=album)])
            if track_no is not None:
                tags.setall("TRCK", [TRCK(encoding=3, text=str(track_no))])
            tags.save(str(path), v2_version=4)
        else:
            raise TaggerError(f"暂不支持向 {path.suffix} 写基础标签")
    except TaggerError:
        raise
    except Exception as exc:
        raise TaggerError(f"写 {path.name} 的基础标签失败: {exc}") from exc


def read_basic_info(path: Path) -> dict[str, Optional[str]]:
    """读取基础标签（title/artist/album/track），用于日志与报告。"""
    path = Path(path)
    fmt = AudioFormat.from_path(path)
    info: dict[str, Optional[str]] = {"title": None, "artist": None, "album": None, "track": None}
    try:
        if fmt is AudioFormat.FLAC:
            audio = FLAC(str(path))
            for out_key, tag_key in (
                ("title", "TITLE"),
                ("artist", "ARTIST"),
                ("album", "ALBUM"),
                ("track", "TRACKNUMBER"),
            ):
                values = audio.get(tag_key)
                if isinstance(values, list) and values:
                    info[out_key] = str(values[0])
        elif fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                return info

            def _first(frame_name: str) -> Optional[str]:
                frames = tags.getall(frame_name)
                if frames and frames[0].text:
                    return str(frames[0].text[0])
                return None

            info["title"] = _first("TIT2")
            info["artist"] = _first("TPE1")
            info["album"] = _first("TALB")
            info["track"] = _first("TRCK")
        else:
            return info
    except Exception:
        return info
    return info


# 防止误用未导入告警（TIT2 等已在上方使用）
_ = (TIT2, TPE1, TPE2, TALB, TRCK)
