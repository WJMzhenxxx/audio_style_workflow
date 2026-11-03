"""已扫描歌曲的状态记录文件（skip-scanned 断点续跑机制）。

与 ``--skip-tagged`` 互补：

- **skip-tagged** 依据音频文件内部的自定义标签（换了 tag key / 标签写入
  失败时无法续跑）；
- **skip-scanned** 依据独立的 JSON 状态文件，记录每个文件的处理指纹
  （``大小 + mtime``）。即使标签没写成功（如风格解析失败），也不会被
  误判为已处理；换 tag key 重跑也不受影响；文件被改动后指纹变化，
  会自动重新分析。

记录两级粒度::

    曲目级: <相对路径> -> {"size", "mtime", "styles", "marked_at"}
    专辑级: TTA 相对路径 -> 上述字段 + {"cue_size", "cue_mtime"}
            —— 曲目级让单曲跳过分析；专辑级让整张 TTA 连切分都跳过。

状态文件默认 ``<root>/.audio_style_scan.json``，JSON 格式，原子写入。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATE_VERSION = 1
DEFAULT_STATE_FILENAME = ".audio_style_scan.json"


class ScanStateStore:
    """扫描记录的加载 / 查询 / 标记 / 持久化。"""

    #: skip_reason 前缀（报告据此区分"已扫描跳过"）
    SKIP_PREFIX = "已在扫描记录中"

    def __init__(self, root: Path, path: Optional[Path] = None) -> None:
        self._root = Path(root).resolve()
        self._path = Path(path) if path else self._root / DEFAULT_STATE_FILENAME
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def size(self) -> int:
        return len(self._entries)

    def __contains__(self, relative_key: str) -> bool:
        return relative_key in self._entries

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load(self) -> None:
        """从磁盘加载（文件不存在 / 损坏时视为空记录并给出警告）。"""
        self._entries = {}
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            entries = payload.get("files") if isinstance(payload, dict) else None
            if not isinstance(entries, dict):
                raise ValueError("缺少 files 字段")
            self._entries = {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}
            logger.info("[state] 加载扫描记录: %d 项 (%s)", len(self._entries), self._path)
        except (ValueError, OSError) as exc:
            logger.warning("[state] 扫描记录无法读取，视为空记录（%s）: %s", self._path, exc)

    def save(self) -> None:
        """原子写入状态文件（.tmp -> replace）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "root": str(self._root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": self._entries,
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=self._path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp_name, self._path)
        except OSError:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        self._dirty = False
        logger.info("[state] 扫描记录已保存: %d 项 -> %s", len(self._entries), self._path)

    def reset(self) -> None:
        """清空全部记录（内存与磁盘）。"""
        self._entries = {}
        self._dirty = True
        if self._path.is_file():
            try:
                self._path.unlink()
                logger.info("[state] 已删除扫描记录: %s", self._path)
            except OSError as exc:
                logger.warning("[state] 删除扫描记录失败: %s", exc)

    # ------------------------------------------------------------------
    # 指纹与查询
    # ------------------------------------------------------------------
    @staticmethod
    def fingerprint(path: Path) -> Optional[tuple[int, int]]:
        try:
            st = Path(path).stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def is_scanned(self, path: Path) -> bool:
        """曲目已成功处理过且指纹未变化。"""
        entry = self._entries.get(self._key(path))
        if entry is None:
            return False
        return self._fingerprint_matches(path, entry)

    def is_album_scanned(self, tta: Path, cue: Optional[Path]) -> bool:
        """整张 TTA 专辑（含 CUE）已成功处理且源文件未变化。"""
        entry = self._entries.get(self._key(tta))
        if entry is None or not entry.get("is_album"):
            return False
        if not self._fingerprint_matches(tta, entry):
            return False
        if entry.get("cue_size") is not None:
            if cue is None:
                return False  # 记录时有 CUE，现在没了 -> 源变了
            cue_fp = self.fingerprint(cue)
            if cue_fp is None or cue_fp != (entry.get("cue_size"), entry.get("cue_mtime")):
                return False
        elif cue is not None and Path(cue).is_file():
            return False  # 记录时无 CUE，现在出现了 -> 源变了
        return True

    # ------------------------------------------------------------------
    # 标记
    # ------------------------------------------------------------------
    def mark_scanned(self, path: Path, *, styles: Optional[list[str]] = None) -> None:
        """把一首已成功处理的歌记入状态。"""
        fp = self.fingerprint(path)
        if fp is None:
            return
        entry = self._entries.get(self._key(path), {})
        entry.update(
            {
                "size": fp[0],
                "mtime": fp[1],
                "marked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if styles:
            entry["styles"] = list(styles)
        self._entries[self._key(path)] = entry
        self._dirty = True

    def mark_album(self, tta: Path, cue: Optional[Path]) -> None:
        """把一张整轨专辑记入状态（专辑所有分轨都成功后调用）。"""
        self.mark_scanned(tta)
        entry = self._entries[self._key(tta)]
        if cue is not None:
            cue_fp = self.fingerprint(cue)
            if cue_fp is not None:
                entry["cue_size"], entry["cue_mtime"] = cue_fp
        entry["is_album"] = True
        entry.setdefault("marked_at", datetime.now(timezone.utc).isoformat())

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _key(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return "rel:" + resolved.relative_to(self._root).as_posix()
        except ValueError:
            return "abs:" + str(resolved)

    @staticmethod
    def _fingerprint_matches(path: Path, entry: dict[str, Any]) -> bool:
        fp = ScanStateStore.fingerprint(path)
        return fp is not None and fp == (entry.get("size"), entry.get("mtime"))
