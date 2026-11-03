"""进程内 Mock 分析端点（无需网络，用于演示、联调与测试）。

- ``mode="json"`` 返回约定结构的 JSON（走确定性解析路径）；
- ``mode="freeform"`` 返回非结构化数据（用于验证 LLM 兜底解析路径）。

风格标签由文件指纹（前 64KB + 文件名）稳定生成，同一文件多次分析结果一致。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from .base import SongAnalysisClient

STYLE_POOL = [
    "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop",
    "R&B", "Folk", "Metal", "Blues", "Ambient", "City Pop",
]


def _fingerprint(audio_path: Path) -> str:
    hasher = hashlib.md5()
    hasher.update(audio_path.name.lower().encode("utf-8", "replace"))
    with audio_path.open("rb") as fh:
        hasher.update(fh.read(65536))
    return hasher.hexdigest()


class MockSongAnalysisClient(SongAnalysisClient):
    name = "mock"

    def __init__(self, mode: Literal["json", "freeform"] = "json", *, timeout: float = 180.0) -> None:
        super().__init__(timeout=timeout)
        if mode not in ("json", "freeform"):
            raise ValueError(f"未知的 mock 模式: {mode!r}")
        self.mode = mode

    def analyze(self, audio_path: Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        digest = _fingerprint(audio_path)
        pool_size = len(STYLE_POOL)
        picks = [
            STYLE_POOL[int(digest[i * 4:i * 4 + 4], 16) % pool_size] for i in range(3)
        ]
        styles: list[str] = []
        for s in picks:
            if s not in styles:
                styles.append(s)

        if self.mode == "freeform":
            return {
                "caption": f"这首歌听起来像是 {styles[0]} 与 {styles[-1]} 的混合体。",
                "note": "非结构化文本响应，用于演示 LLM 兜底风格解析路径",
            }
        return {
            "status": "ok",
            "styles": styles,
            "confidence": 0.87 + (int(digest[:2], 16) % 10) / 100,
            "engine": "mock-v1",
        }
