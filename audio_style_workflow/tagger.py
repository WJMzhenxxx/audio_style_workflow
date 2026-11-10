"""把风格解析结果写回音频文件的标签（基于 mutagen）。

- FLAC：VorbisComment 自由字段（默认 ``GENRE`` / ``MOOD`` / ``BPM`` / ``TEMPO`` / ``KEY``）
- MP3 ：**标准 ID3 帧**——``TCON``(流派) / ``TMOO``(氛围) / ``TBPM``(BPM) / ``TKEY``(调性)，
  ``TEMPO`` 无标准帧，落 ``TXXX:TEMPO``

用标准帧是为了让 Windows 资源管理器「属性 → 详细信息」能直接显示流派/氛围。
键名/分隔符均可自定义。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from mutagen.flac import FLAC
from mutagen.id3 import (
    ID3,
    ID3NoHeaderError,
    COMM,
    TALB,
    TBPM,
    TCON,
    TIT2,
    TKEY,
    TMOO,
    POPM,
    TPE1,
    TPE2,
    TRCK,
    TXXX,
)

from .models import AudioFormat

#: 默认自定义风格标签键名
DEFAULT_CUSTOM_KEY = "GENRE"
#: 多风格默认连接符（写成单值，避免个别播放器把多值字段显示成两行）
DEFAULT_JOINER = "; "
#: 音乐属性标签键名（FLAC VorbisComment；MP3 见 _write 内映射）
DEFAULT_BPM_KEY = "BPM"
DEFAULT_TEMPO_KEY = "TEMPO"
DEFAULT_KEY_KEY = "KEY"
DEFAULT_MOOD_KEY = "MOOD"
DEFAULT_RATING_KEY = "RATING"
DEFAULT_REVIEW_KEY = "COMMENT"

_MAX_KEY_LEN = 60
#: 评分上限（0-10）
_SCORE_MAX = 10
#: POPM 的 email 字段仅作标识，Windows 不显示
_POPM_EMAIL = "audio_style_workflow"
#: 星级 -> POPM rating 字节（Windows 识别的固定阈值）
_POPM_BY_STARS = {1: 1, 2: 64, 3: 128, 4: 196, 5: 255}

#: 键名 -> MP3 标准 ID3 帧（Windows 资源管理器只显示标准帧）
_MP3_STANDARD_FRAMES: dict[str, type] = {
    "GENRE": TCON,   # 流派
    "MOOD": TMOO,    # 氛围
    "BPM": TBPM,
    "KEY": TKEY,
}


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


def _mp3_frame_id(key: str) -> Optional[str]:
    """键名对应的标准 ID3 帧 id（GENRE/MOOD/BPM/KEY），否则 None → 走 TXXX。"""
    frame_cls = _MP3_STANDARD_FRAMES.get(key.strip().upper())
    return frame_cls.__name__ if frame_cls is not None else None


def _set_mp3_text(tags: ID3, key: str, value: str) -> None:
    """按标准帧写；无标准帧的键回落到 ``TXXX:<key>``。"""
    frame_cls = _MP3_STANDARD_FRAMES.get(key.strip().upper())
    if frame_cls is not None:
        tags.setall(frame_cls.__name__, [frame_cls(encoding=3, text=value)])
    else:
        tags.setall(f"TXXX:{key}", [TXXX(encoding=3, desc=key, text=value)])


def _get_mp3_text(tags: ID3, key: str) -> Optional[str]:
    frame_cls = _MP3_STANDARD_FRAMES.get(key.strip().upper())
    frame_id = frame_cls.__name__ if frame_cls is not None else f"TXXX:{key}"
    frames = tags.getall(frame_id)
    if frames and getattr(frames[0], "text", None):
        return str(frames[0].text[0])
    return None


def write_styles(
    path: Path,
    styles: Sequence[str],
    *,
    key: str = DEFAULT_CUSTOM_KEY,
    joiner: str = DEFAULT_JOINER,
) -> str:
    """将风格列表写入标签，返回实际写入的值。

    FLAC 写 VorbisComment；MP3 默认写标准帧 ``TCON``（Windows 可识别），
    自定义键名则写 ``TXXX:<key>``。

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
            _set_mp3_text(tags, key, value)
            tags.save(str(path), v2_version=4)
        else:
            raise TaggerError(f"暂不支持向 {path.suffix or '<无后缀>'} 格式写标签: {path}")
    except TaggerError:
        raise
    except Exception as exc:  # mutagen 各种底层异常
        raise TaggerError(f"写 {path.name} 的标签失败: {exc}") from exc
    return value


def get_styles(path: Path, *, key: str = DEFAULT_CUSTOM_KEY) -> Optional[str]:
    """读取风格标签（MP3 优先读标准帧 ``TCON``）；不存在或读取失败返回 None。"""
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
            return _get_mp3_text(tags, key)
    except Exception:
        return None
    return None


def write_extra_tags(
    path: Path,
    *,
    bpm: Optional[int] = None,
    tempo: Optional[str] = None,
    key: Optional[str] = None,
    mood: Optional[str] = None,
    score: Optional[int] = None,
    review: Optional[str] = None,
    tempo_key: str = DEFAULT_TEMPO_KEY,
    mood_key: str = DEFAULT_MOOD_KEY,
    rating_key: str = DEFAULT_RATING_KEY,
    review_key: str = DEFAULT_REVIEW_KEY,
) -> dict[str, str]:
    """把 BPM / Tempo / Key / Mood / 评分 / 评价写入标签，返回实际写入的字段。

    - FLAC：VorbisComment ``BPM`` / ``TEMPO`` / ``KEY`` / ``MOOD`` /
      ``RATING``(0-100) / ``COMMENT``
    - MP3 ：标准帧 ``TBPM`` / ``TKEY`` / ``TMOO`` / ``POPM``(评分) / ``COMM``(评价)
      （Windows 资源管理器可显示），Tempo 无标准帧，落 ``TXXX:TEMPO``

    所有值都为空时不做任何写入（返回空 dict）；单个值非法只会被跳过。
    """
    path = Path(path)
    tempo_key = _check_key(tempo_key)
    mood_key = _check_key(mood_key)
    rating_key = _check_key(rating_key)
    review_key = _check_key(review_key)
    written: dict[str, str] = {}

    bpm_value: Optional[str] = None
    if bpm is not None:
        try:
            bpm_int = int(bpm)
        except (TypeError, ValueError):
            bpm_int = -1
        if 1 <= bpm_int <= 400:
            bpm_value = str(bpm_int)
    tempo_value = (tempo or "").strip() or None
    key_value = (key or "").strip() or None
    mood_value = (mood or "").strip() or None
    review_value = (review or "").strip() or None
    score_value: Optional[int] = None
    if score is not None:
        try:
            score_int = int(score)
        except (TypeError, ValueError):
            score_int = -1
        if 0 <= score_int <= _SCORE_MAX:
            score_value = score_int

    if (
        bpm_value is None
        and tempo_value is None
        and key_value is None
        and mood_value is None
        and score_value is None
        and review_value is None
    ):
        return written

    fmt = AudioFormat.from_path(path)
    try:
        if fmt is AudioFormat.FLAC:
            audio = FLAC(str(path))
            if bpm_value is not None:
                audio[DEFAULT_BPM_KEY] = [bpm_value]
                written["bpm"] = bpm_value
            if tempo_value is not None:
                audio[tempo_key] = [tempo_value]
                written["tempo"] = tempo_value
            if key_value is not None:
                audio[DEFAULT_KEY_KEY] = [key_value]
                written["key"] = key_value
            if mood_value is not None:
                audio[mood_key] = [mood_value]
                written["mood"] = mood_value
            if score_value is not None:
                # 0-10 -> 0-100，便于部分播放器按百分比理解
                audio[rating_key] = [str(score_value * 10)]
                written["score"] = str(score_value)
            if review_value is not None:
                audio[review_key] = [review_value]
                written["review"] = review_value
            audio.save()
        elif fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                tags = ID3()
            if bpm_value is not None:
                tags.setall("TBPM", [TBPM(encoding=3, text=bpm_value)])
                written["bpm"] = bpm_value
            if key_value is not None:
                tags.setall("TKEY", [TKEY(encoding=3, text=key_value)])
                written["key"] = key_value
            if mood_value is not None:
                tags.setall("TMOO", [TMOO(encoding=3, text=mood_value)])
                written["mood"] = mood_value
            if score_value is not None:
                # POPM：Windows 从这里读「星级」（只有 5 档，9 分会退化成 4 星）
                stars = max(1, min(5, round(score_value / 2)))
                tags.setall("POPM", [POPM(
                    email=_POPM_EMAIL, rating=_POPM_BY_STARS[stars], count=0,
                )])
                # 再存一份 0-10 的原始分，保证导出/导入无损
                tags.setall("TXXX:RATING", [TXXX(
                    encoding=3, desc="RATING", text=str(score_value),
                )])
                written["score"] = str(score_value)
            if review_value is not None:
                tags.setall("COMM", [COMM(
                    encoding=3, lang="eng", desc="", text=review_value,
                )])
                written["review"] = review_value
            if tempo_value is not None:
                tags.setall(
                    f"TXXX:{tempo_key}",
                    [TXXX(encoding=3, desc=tempo_key, text=tempo_value)],
                )
                written["tempo"] = tempo_value
            tags.save(str(path), v2_version=4)
        else:
            raise TaggerError(f"暂不支持向 {path.suffix or '<无后缀>'} 格式写音乐属性标签: {path}")
    except TaggerError:
        raise
    except Exception as exc:  # mutagen 各种底层异常
        raise TaggerError(f"写 {path.name} 的音乐属性标签失败: {exc}") from exc
    return written


def read_extra_tags(
    path: Path,
    *,
    tempo_key: str = DEFAULT_TEMPO_KEY,
    mood_key: str = DEFAULT_MOOD_KEY,
    rating_key: str = DEFAULT_RATING_KEY,
    review_key: str = DEFAULT_REVIEW_KEY,
) -> dict[str, Optional[str]]:
    """读取 BPM / Tempo / Key / Mood / 评分 / 评价；缺失返回 None（不抛异常）。"""
    path = Path(path)
    info: dict[str, Optional[str]] = {
        "bpm": None, "tempo": None, "key": None, "mood": None,
        "score": None, "review": None,
    }
    fmt = AudioFormat.from_path(path)
    try:
        if fmt is AudioFormat.FLAC:
            audio = FLAC(str(path))

            def _flac_first(tag_key: str) -> Optional[str]:
                values = audio.get(tag_key)
                if isinstance(values, list) and values:
                    return str(values[0])
                return None

            info["bpm"] = _flac_first(DEFAULT_BPM_KEY)
            info["tempo"] = _flac_first(tempo_key)
            info["key"] = _flac_first(DEFAULT_KEY_KEY)
            info["mood"] = _flac_first(mood_key)
            rating = _flac_first(rating_key)
            info["score"] = str(round(int(rating) / 10)) if rating and rating.isdigit() else None
            info["review"] = _flac_first(review_key)
        elif fmt is AudioFormat.MP3:
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                return info

            def _frame_first(frame_name: str) -> Optional[str]:
                frames = tags.getall(frame_name)
                if frames and getattr(frames[0], "text", None):
                    return str(frames[0].text[0])
                return None

            info["bpm"] = _frame_first("TBPM")
            info["key"] = _frame_first("TKEY")
            info["mood"] = _frame_first("TMOO")
            info["tempo"] = _frame_first(f"TXXX:{tempo_key}")
            exact = _frame_first("TXXX:RATING")
            if exact and exact.strip().isdigit():
                info["score"] = exact.strip()
            else:
                popm = tags.getall("POPM")
                if popm:
                    # 反查星级阈值，再换算回 0-10（近似）
                    byte = int(popm[0].rating)
                    stars = min(_POPM_BY_STARS, key=lambda s: abs(_POPM_BY_STARS[s] - byte))
                    info["score"] = str(stars * 2)
            comms = tags.getall("COMM")
            if comms and getattr(comms[0], "text", None):
                info["review"] = str(comms[0].text[0])
    except Exception:
        return info
    return info


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
