"""日志与逐曲分析明细。

目录结构（默认在运行目录下）::

    log/
    ├── latest.log          # 本次运行的「全量日志」（每次运行覆盖，UTF-8）
    └── details/
        ├── 某首歌__<hash>.json
        └── ...

``latest.log`` 汇总整个工作流的日志；``details/`` 下每首歌一个 JSON，
记录该曲**分析的原始输入**（音频路径/大小、客户端、端点、模式、提示词等）
与**原始输出**（分析端点返回的完整响应），以及耗时与错误。

设计要点：

- 日志目录可配置（:func:`setup_logging` / :class:`AnalysisDetailRecorder`
  均接受 ``log_dir``，默认 ``Path("log")``）；
- 明细文件名由「文件名 + 绝对路径哈希」决定，同一首歌多次分析会覆盖同一个
  文件，便于追踪，也不会因并发顺序而漂移；
- 明细写入加锁，分析阶段并发安全；
- 本模块只依赖标准库，不引入额外运行依赖。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

#: 默认日志根目录
DEFAULT_LOG_DIR = Path("log")
#: 全量日志文件名
LATEST_LOG_NAME = "latest.log"
#: 逐曲明细子目录名
DETAILS_DIR_NAME = "details"

_LOGGER_NAME = "audio_style_workflow"
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogPaths:
    """一次日志布局的路径集合。"""

    root: Path
    latest: Path
    details: Path


# ----------------------------------------------------------------------
# 目录与日志配置
# ----------------------------------------------------------------------
def prepare_log_dirs(log_dir: Optional[Path] = None) -> LogPaths:
    """确保 ``<log_dir>`` 与 ``<log_dir>/details`` 存在，返回路径集合。"""
    root = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    details = root / DETAILS_DIR_NAME
    details.mkdir(parents=True, exist_ok=True)
    return LogPaths(root=root, latest=root / LATEST_LOG_NAME, details=details)


def _drop_managed_handlers(target: logging.Logger) -> None:
    for handler in list(target.handlers):
        if getattr(handler, "_asw_managed", False):
            target.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - 关闭失败不影响主流程
                pass


def setup_logging(
    log_dir: Optional[Path] = None,
    *,
    verbose: bool = False,
    level: Optional[int] = None,
    console: bool = True,
) -> LogPaths:
    """配置根日志：控制台 + ``log/latest.log``（覆盖写）。

    :returns: :class:`LogPaths`，可用于在启动信息里展示日志位置。
    """
    paths = prepare_log_dirs(log_dir)
    resolved_level = level if level is not None else (logging.DEBUG if verbose else logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    _drop_managed_handlers(root)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(paths.latest, mode="w", encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    file_handler._asw_managed = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler._asw_managed = True  # type: ignore[attr-defined]
        root.addHandler(console_handler)

    return paths


# ----------------------------------------------------------------------
# 逐曲分析明细
# ----------------------------------------------------------------------
def _sanitize_stem(stem: str, max_len: int = 80) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", stem or "").strip().strip(". ")
    return cleaned[:max_len].rstrip() or "audio"


def _stable_digest(audio_path: Path) -> str:
    try:
        resolved = str(Path(audio_path).resolve())
    except OSError:  # pragma: no cover - 极端路径异常
        resolved = str(audio_path)
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]


def describe_request(audio_path: Path, client: Any) -> dict[str, Any]:
    """把「分析输入」整理成可序列化 dict（客户端元信息 + 音频文件信息）。"""
    audio = Path(audio_path)
    try:
        size = audio.stat().st_size
    except OSError:
        size = None
    info: dict[str, Any] = {
        "audio": str(audio),
        "audio_size": size,
        "client": getattr(client, "name", type(client).__name__),
    }
    # 各客户端可能有/没有这些属性，按存在性收集
    for attr in ("endpoint", "api_mode", "model_name", "max_new_tokens",
                 "temperature", "thinking_budget", "file_field"):
        value = getattr(client, attr, None)
        if value is not None:
            info[attr] = value
    prompt = getattr(client, "prompt", None)
    if prompt:
        info["prompt"] = prompt
    return info


class AnalysisDetailRecorder:
    """把每首歌的分析输入/输出写入 ``<log_dir>/details/*.json``（线程安全）。"""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self._paths = prepare_log_dirs(log_dir)
        self._lock = threading.Lock()

    @property
    def details_dir(self) -> Path:
        return self._paths.details

    def path_for(self, audio_path: Path) -> Path:
        """同一首歌稳定映射到同一个明细文件。"""
        audio = Path(audio_path)
        return self._paths.details / f"{_sanitize_stem(audio.stem)}__{_stable_digest(audio)}.json"

    def record(
        self,
        audio_path: Path,
        *,
        request: dict[str, Any],
        response: Any = None,
        error: Optional[str] = None,
        elapsed: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        """写一条明细，返回文件路径。"""
        audio = Path(audio_path)
        payload: dict[str, Any] = {
            "audio": str(audio),
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "request": request,
            "response": response,
            "error": error,
        }
        if elapsed is not None:
            payload["elapsed_sec"] = round(float(elapsed), 4)
        if extra:
            payload.update(extra)

        target = self.path_for(audio)
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            tmp.replace(target)
        logger.debug("[detail] 已写入分析明细: %s", target)
        return target


__all__: Iterable[str] = (
    "DEFAULT_LOG_DIR",
    "LATEST_LOG_NAME",
    "DETAILS_DIR_NAME",
    "LogPaths",
    "prepare_log_dirs",
    "setup_logging",
    "describe_request",
    "AnalysisDetailRecorder",
)
