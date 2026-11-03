"""基于 HTTP multipart 上传的歌曲分析端点实现（httpx）。

用法::

    client = HttpSongAnalysisClient(
        endpoint="https://example.com/api/analyze",
        api_key="sk-...",            # 可选，默认 Bearer 鉴权
        file_field="file",           # 服务端接收文件的表单字段名
    )
    response = client.analyze(Path("song.flac"))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from .base import SongAnalysisClient, SongAnalysisError

logger = logging.getLogger(__name__)

_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".tta": "audio/x-tta",
}


class HttpSongAnalysisClient(SongAnalysisClient):
    """把音频文件以 ``multipart/form-data`` POST 到分析服务的实现。

    - 连接级错误自动重试（``retries``，仅重连不重放请求体）；
    - HTTP >= 400 / 非 JSON / 非 JSON 对象 / service 层 status=error
      一律抛 :class:`SongAnalysisError` 并附响应片段；
    - 内部 ``httpx.Client`` 线程安全，可配合工作流 ``--concurrency``。
    """

    name = "http"

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: Optional[str] = None,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        file_field: str = "file",
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_form: Optional[Mapping[str, Any]] = None,
        timeout: float = 300.0,
        retries: int = 2,
        verify: bool = True,
        user_agent: str = "audio-style-workflow/0.1",
        follow_redirects: bool = True,
    ) -> None:
        super().__init__(timeout=timeout)
        if not endpoint or not str(endpoint).startswith(("http://", "https://")):
            raise ValueError(f"endpoint 必须是 http(s) URL，当前: {endpoint!r}")
        self.endpoint = str(endpoint)
        self.file_field = file_field
        self.retries = max(0, int(retries))

        self.headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        if extra_headers:
            self.headers.update(dict(extra_headers))
        if api_key:
            self.headers[auth_header] = f"{auth_scheme} {api_key}".strip()

        self.extra_form: dict[str, Any] = dict(extra_form) if extra_form else {}
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            verify=verify,
            follow_redirects=follow_redirects,
            transport=httpx.HTTPTransport(retries=self.retries),
        )

    def analyze(self, audio_path: Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise SongAnalysisError(f"音频文件不存在: {audio_path}")

        content_type = _CONTENT_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")
        logger.info("[analyze] 上传 %s (%.1f MB)", audio_path.name, audio_path.stat().st_size / 1e6)

        try:
            with audio_path.open("rb") as fh:
                response = self._client.post(
                    self.endpoint,
                    headers=self.headers,
                    files={self.file_field: (audio_path.name, fh, content_type)},
                    data=self.extra_form,
                )
        except httpx.HTTPError as exc:
            raise SongAnalysisError(f"请求分析端点失败（{audio_path.name}）: {type(exc).__name__}: {exc}") from exc
        except OSError as exc:
            raise SongAnalysisError(f"读取音频文件失败（{audio_path.name}）: {exc}") from exc

        if response.status_code >= 400:
            raise SongAnalysisError(
                f"分析端点返回 HTTP {response.status_code}（{audio_path.name}）: {self._snippet(response)}"
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise SongAnalysisError(
                f"分析端点响应不是 JSON（{audio_path.name}）: {self._snippet(response)}"
            ) from exc
        if not isinstance(payload, dict):
            raise SongAnalysisError(
                f"分析端点响应应为 JSON 对象（{audio_path.name}），实为 {type(payload).__name__}"
            )

        status = str(payload.get("status", "ok")).lower()
        if status in ("error", "failed", "failure"):
            raise SongAnalysisError(
                f"分析端点报告失败（{audio_path.name}）: {payload.get('message') or self._snippet(response)}"
            )
        return payload

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _snippet(response: httpx.Response, limit: int = 300) -> str:
        text = response.text.replace("\n", " ").strip()
        return text[:limit] + ("..." if len(text) > limit else "")
