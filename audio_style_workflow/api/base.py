"""分析端点抽象接口（只定义契约，不含任何传输实现）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SongAnalysisError(RuntimeError):
    """与分析端点交互失败（网络错误、HTTP 4xx/5xx、响应协议不符合约定）。"""


class SongAnalysisClient(ABC):
    """将一首歌曲文件提交到分析服务并返回原始响应。

    约定的响应为 JSON 对象，推荐结构::

        {
          "status": "ok",                 # "ok" / "error"
          "styles": ["Pop", "R&B"],       # 风格标签（也接受逗号分隔字符串）
          "confidence": 0.87,             # 可选，0~1
          "message": "..."                # 可选
        }

    实现方的职责边界:

    1. 负责把音频文件安全送达分析服务（multipart / base64 / 对象存储引用均可）；
    2. 一切失败都用 :class:`SongAnalysisError` 表达（网络、鉴权、协议、超时），
       错误消息应包含文件名与原因，便于汇总报告定位；
    3. 只返回**原始 JSON dict**，不解释风格字段——那是 StyleParser 的职责。
    """

    name: str = "base"

    def __init__(self, *, timeout: float = 180.0) -> None:
        self.timeout = timeout

    @abstractmethod
    def analyze(self, audio_path: Path) -> dict[str, Any]:
        """上传并分析一首歌，返回原始响应 dict。

        :raises SongAnalysisError: 任何失败情形
        """
        raise NotImplementedError

    # 便于 with 语法管理底层资源
    def __enter__(self) -> "SongAnalysisClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """释放资源（默认无操作）。"""


def create_client(kind: str = "http", **kwargs: Any) -> SongAnalysisClient:
    """客户端工厂。

    :param kind:
        - ``"http"``   -> HTTP multipart 上传实现
        - ``"mock"``   -> 进程内演示实现（无需网络）
        - ``"local"``  -> 本地 MusicFlamingo 模型推理实现（离线，需 transformers/torch）
        - ``"sglang"`` -> sglang 部署的音乐理解服务（音频传本地路径）
    """
    kind = (kind or "http").strip().lower()
    if kind == "http":
        from .http_client import HttpSongAnalysisClient

        return HttpSongAnalysisClient(**kwargs)
    if kind == "mock":
        from .mock_client import MockSongAnalysisClient

        return MockSongAnalysisClient(**kwargs)
    if kind == "local":
        from .local_model_client import LocalMusicFlamingoClient

        return LocalMusicFlamingoClient(**kwargs)
    if kind == "sglang":
        from .sglang_client import SglangMusicClient

        return SglangMusicClient(**kwargs)
    raise ValueError(f"未知分析端点类型: {kind!r}（可选: http / mock / local / sglang）")
