"""标签导出 / 导入。

把工作流写进音频文件的标签导出成一份 JSON，之后可以在别的机器、别的目录结构下
按 **MD5 或文件名** 重新灌回去；导入时若遇到还没切分的 TTA 整轨，会先按 CUE
切分再写入。

JSON 结构（``version 1``）::

    {
      "version": 1,
      "exported_at": "2026-11-10T10:00:00+08:00",
      "source_root": "D:\\\\music",
      "tag_keys": {"custom": "GENRE", "joiner": "; "},
      "tracks": [
        {
          "rel_path": "album/01 - Song.flac",
          "name": "01 - Song",
          "md5": "<文件 md5>",
          "size": 12345678,
          "tags": {
            "GENRE": "Pop; Dream Pop",
            "BPM": "120",
            "TEMPO": "Moderato",
            "KEY": "C major",
            "MOOD": "温暖, 舒缓",
            "RATING": "8",
            "COMMENT": "旋律抓耳，制作干净。"
          },
          "album": {                      # 来自 TTA 整轨的分轨才有
            "tta_rel_path": "album/album.tta",
            "tta_md5": "<整轨 md5>",
            "cue_rel_path": "album/album.cue",
            "cue_md5": "<cue md5>",
            "track_no": 1,
            "title": "Song"
          }
        }
      ]
    }

导入匹配优先级：``rel_path`` → ``md5`` → 文件名（不含扩展名，忽略大小写）。
带 ``album`` 的分轨按「整轨 md5 / 路径 + 轨号」定位；整轨还没切分会先切分。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .cue import parse_cue_file
from .models import TtaAlbum
from .scanner import AUDIO_EXTS, scan_directory
from .splitter import SplitError, TtaSplitter, expected_track_filename, resolve_ffmpeg
from .tagger import (
    DEFAULT_CUSTOM_KEY,
    DEFAULT_JOINER,
    get_styles,
    read_basic_info,
    read_extra_tags,
    write_extra_tags,
    write_styles,
)

logger = logging.getLogger(__name__)

#: 导出文件格式版本
EXPORT_VERSION = 1
_MD5_CHUNK = 1 << 20

#: 导出文件里使用的固定键名
TAG_GENRE = "GENRE"
TAG_BPM = "BPM"
TAG_TEMPO = "TEMPO"
TAG_KEY = "KEY"
TAG_MOOD = "MOOD"
TAG_RATING = "RATING"
TAG_COMMENT = "COMMENT"


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def file_md5(path: Path, *, chunk: int = _MD5_CHUNK) -> str:
    """计算文件 md5（按块读取，大文件不占内存）。"""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_md5(path: Path, *, ffmpeg: Optional[str] = None) -> Optional[str]:
    """按**解码后的音频内容**计算 md5（``ffmpeg -f md5``）。

    与 :func:`file_md5` 的区别：文件 md5 把标签也算进去，一旦重写标签就变了；
    音频 md5 只看声音数据，因此跨「有标签/无标签」的副本也能匹配。
    ffmpeg 不可用或解码失败时返回 ``None``。
    """
    exe = resolve_ffmpeg(ffmpeg)
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "md5", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        logger.debug("音频 md5 计算失败 %s: %s", path.name, (proc.stderr or "")[-200:])
        return None
    match = re.search(r"MD5=([0-9a-fA-F]{32})", proc.stdout or "")
    return match.group(1).lower() if match else None


def _rel(path: Path, root: Path) -> str:
    """尽量给相对路径；跨盘符时退回绝对路径。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path.resolve())


def _iter_audio(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    ]


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _split_styles(value: str, joiner: str) -> list[str]:
    if joiner and joiner in value:
        return [part.strip() for part in value.split(joiner) if part.strip()]
    for sep in (";", "、", "|"):
        if sep in value:
            return [part.strip() for part in value.split(sep) if part.strip()]
    return [value.strip()] if value.strip() else []


def read_track_tags(
    path: Path, *, custom_key: str = DEFAULT_CUSTOM_KEY
) -> dict[str, str]:
    """读出本项目写入的标签；空值不出现。"""
    tags: dict[str, str] = {}

    styles = get_styles(path, key=custom_key)
    if styles:
        tags[TAG_GENRE] = styles

    extra = read_extra_tags(path)
    for source, key in (
        ("bpm", TAG_BPM),
        ("tempo", TAG_TEMPO),
        ("key", TAG_KEY),
        ("mood", TAG_MOOD),
        ("score", TAG_RATING),
        ("review", TAG_COMMENT),
    ):
        value = extra.get(source)
        if value:
            tags[key] = str(value)
    return tags


# ----------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------
@dataclass
class ExportReport:
    output: Path
    tracks: int = 0
    with_tags: int = 0
    albums: int = 0
    skipped_untagged: int = 0


def _album_provenance(root: Path) -> dict[Path, tuple[TtaAlbum, int]]:
    """分轨文件 -> (所属 TTA 专辑, 轨号)。"""
    mapping: dict[Path, tuple[TtaAlbum, int]] = {}
    try:
        state = scan_directory(root)
    except Exception as exc:  # 目录不可读等极端情况，退化为无专辑信息
        logger.warning("扫描专辑信息失败（%s），分轨将不带 album 信息", exc)
        return mapping

    for album in state.albums:
        if album.cue is None:
            continue
        # 扫描阶段不会解析 CUE，这里按需解析
        sheet = album.sheet
        if sheet is None:
            try:
                sheet = parse_cue_file(album.cue)
            except Exception as exc:
                logger.debug("解析 CUE 失败 %s: %s", album.cue.name, exc)
                continue
            album.sheet = sheet
        for track in sheet.tracks:
            mapping[album.tta.parent / expected_track_filename(track)] = (album, track.number)
    return mapping


def export_tags(
    root: Path,
    output: Path,
    *,
    custom_key: str = DEFAULT_CUSTOM_KEY,
    joiner: str = DEFAULT_JOINER,
    include_untagged: bool = False,
    ffmpeg: Optional[str] = None,
    with_audio_hash: bool = True,
) -> ExportReport:
    """把目录下音频的标签导出为 JSON。

    :param include_untagged: 连一个标签都没有的文件也记录（默认跳过）。
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {root}")

    provenance = _album_provenance(root)
    report = ExportReport(output=Path(output))
    tracks: list[dict[str, Any]] = []
    md5_cache: dict[Path, str] = {}

    def md5_of(path: Path) -> str:
        if path not in md5_cache:
            md5_cache[path] = file_md5(path)
        return md5_cache[path]

    album_keys: set[str] = set()

    for path in _iter_audio(root):
        tags = read_track_tags(path, custom_key=custom_key)
        if not tags and not include_untagged:
            report.skipped_untagged += 1
            continue

        entry: dict[str, Any] = {
            "rel_path": _rel(path, root),
            "name": path.stem,
            "md5": md5_of(path),
            "size": path.stat().st_size,
            "tags": tags,
        }
        if with_audio_hash:
            audio_hash = audio_md5(path, ffmpeg=ffmpeg)
            if audio_hash:
                entry["audio_md5"] = audio_hash
        basic = read_basic_info(path)
        for source, key in (("title", "title"), ("artist", "artist"), ("album", "album_title")):
            # 基础标签的 album 是字符串，避免与 TTA 来源信息（album 对象）重名
            if basic.get(source):
                entry[key] = basic[source]

        source = provenance.get(path)
        if source is not None:
            album, track_no = source
            entry["album"] = {
                "tta_rel_path": _rel(album.tta, root),
                "tta_md5": md5_of(album.tta),
                "cue_rel_path": _rel(album.cue, root) if album.cue else "",
                "cue_md5": md5_of(album.cue) if album.cue and album.cue.is_file() else "",
                "track_no": track_no,
                "title": path.stem,
            }
            album_keys.add(entry["album"]["tta_rel_path"])

        tracks.append(entry)
        report.tracks += 1
        if tags:
            report.with_tags += 1

    payload = {
        "version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_root": str(root.resolve()),
        "tag_keys": {"custom": custom_key, "joiner": joiner},
        "album_count": len(album_keys),
        "tracks": tracks,
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    report.albums = len(album_keys)
    logger.info(
        "导出完成: %d 首（%d 首有标签，跳过 %d 首无标签，%d 张整轨）-> %s",
        report.tracks, report.with_tags, report.skipped_untagged, report.albums, output,
    )
    return report


def load_export(path: Path) -> dict[str, Any]:
    """读取并校验导出文件。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ValueError(f"不是有效的标签导出文件: {path}")
    version = data.get("version")
    if version is not None and int(version) > EXPORT_VERSION:
        logger.warning(
            "导出文件版本 %s 高于本程序支持的 %s，可能忽略部分字段", version, EXPORT_VERSION
        )
    return data


# ----------------------------------------------------------------------
# 导入
# ----------------------------------------------------------------------
@dataclass
class ImportReport:
    applied: int = 0          # 成功写入的条目数
    files: int = 0            # 实际写入的文件数（md5 相同可能对应多个文件）
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    split_albums: int = 0
    split_tracks: int = 0
    dry_run: bool = False


def _apply_tags(
    path: Path,
    tags: dict[str, str],
    *,
    custom_key: str,
    joiner: str,
    dry_run: bool,
) -> None:
    """把一条记录的标签写入文件。"""
    if not tags:
        return
    if dry_run:
        return

    genre = tags.get(TAG_GENRE) or tags.get(custom_key)
    if genre:
        write_styles(path, _split_styles(genre, joiner), key=custom_key, joiner=joiner)

    music = {
        "bpm": _int_or_none(tags.get(TAG_BPM)),
        "tempo": tags.get(TAG_TEMPO),
        "key": tags.get(TAG_KEY),
        "mood": tags.get(TAG_MOOD),
        "score": _int_or_none(tags.get(TAG_RATING)),
        "review": tags.get(TAG_COMMENT),
    }
    if any(value is not None for value in music.values()):
        write_extra_tags(path, **music)


def _build_file_index(
    files: list[Path], md5_cache: dict[Path, str]
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """(相对路径小写 -> 文件, 文件名小写 -> 文件)。"""
    by_rel: dict[str, list[Path]] = {}
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_rel.setdefault(path.as_posix().lower(), []).append(path)
        by_name.setdefault(path.stem.lower(), []).append(path)
    return by_rel, by_name


def import_tags(
    data: dict[str, Any],
    root: Path,
    *,
    dry_run: bool = False,
    ffmpeg: Optional[str] = None,
    custom_key: Optional[str] = None,
    joiner: Optional[str] = None,
    split_missing_tta: bool = True,
) -> ImportReport:
    """把导出文件里的标签写入 ``root`` 下的音频。

    :param split_missing_tta: 遇到尚未切分的 TTA 整轨时先切分再写入。
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {root}")

    keys = data.get("tag_keys") or {}
    custom_key = custom_key or keys.get("custom") or DEFAULT_CUSTOM_KEY
    joiner = joiner or keys.get("joiner") or DEFAULT_JOINER

    report = ImportReport(dry_run=dry_run)
    files = _iter_audio(root)
    by_rel, by_name = _build_file_index(files, {})
    md5_cache: dict[str, list[Path]] = {}
    audio_cache: dict[str, list[Path]] = {}

    # 先按相对路径校正索引（导出与导入根目录不同也能用文件名兜底）
    rel_index: dict[str, Path] = {}
    for path in files:
        rel_index[_rel(path, root).lower()] = path
        rel_index[path.as_posix().lower()] = path

    def md5_index() -> dict[str, list[Path]]:
        if not md5_cache:
            for path in files:
                md5_cache.setdefault(file_md5(path), []).append(path)
        return md5_cache

    def audio_index() -> dict[str, list[Path]]:
        """解码后的音频 md5 索引（标签被改写过也能匹配）。"""
        if not audio_cache:
            for path in files:
                digest = audio_md5(path, ffmpeg=ffmpeg)
                if digest:
                    audio_cache.setdefault(digest, []).append(path)
        return audio_cache

    # ---- 专辑：先确保整轨已切分 ----
    album_index: dict[str, TtaAlbum] = {}
    album_tracks: dict[str, dict[int, Path]] = {}
    needs_album = any((entry.get("album") or {}).get("tta_rel_path") for entry in data["tracks"])

    if needs_album:
        try:
            state = scan_directory(root)
        except Exception as exc:
            logger.warning("扫描整轨失败（%s），将退化为按文件名匹配", exc)
            state = None
        for album in (state.albums if state else []):
            if album.cue is None:
                continue
            album_index[_rel(album.tta, root).lower()] = album
            album_index[album.tta.name.lower()] = album

    splitter: Optional[TtaSplitter] = None

    def ensure_split(album: TtaAlbum) -> dict[int, Path]:
        nonlocal splitter
        key = str(album.tta)
        if key in album_tracks:
            return album_tracks[key]

        sheet = album.sheet
        expected = (
            {t.number: album.tta.parent / expected_track_filename(t) for t in sheet.tracks}
            if sheet
            else {}
        )
        if sheet and all(p.is_file() for p in expected.values()):
            album_tracks[key] = expected
            return expected

        if not split_missing_tta:
            album_tracks[key] = {n: p for n, p in expected.items() if p.is_file()}
            return album_tracks[key]

        if splitter is None:
            splitter = TtaSplitter(ffmpeg)
        logger.info("[import] 整轨尚未切分，先切分: %s", album.tta.name)
        tasks = splitter.split(album, reuse_existing=True)
        mapping = {t.track_no: t.path for t in tasks if t.track_no is not None}
        album_tracks[key] = mapping
        report.split_albums += 1
        report.split_tracks += len(tasks)
        return mapping

    # ---- 逐条写入 ----
    for entry in data["tracks"]:
        label = entry.get("rel_path") or entry.get("name") or "<未知>"
        tags = entry.get("tags") or {}
        if not tags:
            continue

        target: Optional[Path] = None
        targets: list[Path] = []

        album_info = entry.get("album") or {}
        if album_info:
            album = album_index.get(str(album_info.get("tta_rel_path", "")).lower())
            if album is None and album_info.get("tta_md5"):
                for candidate in album_index.values():
                    if file_md5(candidate.tta) == album_info["tta_md5"]:
                        album = candidate
                        break
            if album is not None:
                try:
                    mapping = ensure_split(album)
                except SplitError as exc:
                    report.failed.append(f"{label}: 切分失败 {exc}")
                    continue
                track_no = _int_or_none(album_info.get("track_no"))
                if track_no is not None:
                    target = mapping.get(track_no)
                if target is None and album.sheet is not None:
                    title = str(album_info.get("title") or "").lower()
                    for track in album.sheet.tracks:
                        if title and track.title.lower() == title:
                            target = album.tta.parent / expected_track_filename(track)
                            break
                if target is not None and not target.is_file():
                    target = None

        if target is None:
            rel = str(entry.get("rel_path") or "").lower()
            if rel and rel in rel_index:
                target = rel_index[rel]
        if target is None:
            name = str(entry.get("name") or "").lower()
            candidates = by_name.get(name, []) if name else []
            if len(candidates) == 1:
                target = candidates[0]
            elif len(candidates) > 1:
                # 同名多份时用 md5 进一步收敛
                md5 = entry.get("md5")
                if md5:
                    same = [p for p in candidates if file_md5(p) == md5]
                    if len(same) >= 1:
                        targets = same
        if target is None and not targets:
            md5 = entry.get("md5")
            if md5:
                targets = md5_index().get(md5, [])
        if target is None and not targets:
            # 文件 md5 只对字节完全一致的副本有效；标签被改写过时用音频 md5
            digest = entry.get("audio_md5")
            if digest:
                targets = audio_index().get(digest, [])

        if target is not None:
            targets = [target]
        if not targets:
            report.missing.append(label)
            continue

        for path in targets:
            try:
                _apply_tags(
                    path, tags, custom_key=custom_key, joiner=joiner, dry_run=dry_run
                )
            except Exception as exc:
                report.failed.append(f"{label}: {exc}")
                break
        else:
            report.applied += 1
            report.files += len(targets)

    logger.info(
        "导入完成: 应用 %d 条（写入 %d 个文件），新建切分 %d 张整轨 / %d 轨，缺失 %d 条，失败 %d 条",
        report.applied, report.files, report.split_albums, report.split_tracks,
        len(report.missing), len(report.failed),
    )
    return report
