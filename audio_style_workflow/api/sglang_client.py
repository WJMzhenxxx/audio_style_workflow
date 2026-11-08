"""基于 sglang 部署的音乐理解模型（如 moss-music-model）的分析端点实现。

对应服务端启动方式（音频不经 HTTP 上传，服务直接读文件路径）::

    sglang serve --model-path /path/to/moss-music-model --trust-remote-code

支持两种 API 模式：

1. ``generate``（默认，最简）—— sglang 原生 REST::

       POST <base>/generate
       {
         "text": "<分析提示词>",
         "audio_data": "<音频在本机的路径>",
         "sampling_params": {"max_new_tokens": 1024, "temperature": 0.0}
       }
       -> {"text": "<模型输出>", ...}

2. ``chat``（OpenAI 兼容 + thinking 分离 + budget 控制）::

       POST <base>/v1/chat/completions
       {
         "model": "default",
         "messages": [{"role": "user", "content": [
             {"type": "audio_url", "audio_url": {"url": "<音频路径>"}},
             {"type": "text", "text": "<分析提示词>"}]}],
         "max_tokens": 1024, "temperature": 0.0,
         "separate_reasoning": true,
         "custom_logit_processor": "Qwen3InstructionInjectionThinkingBudgetLogitProcessor",
         "custom_params": {"thinking_budget": 50}
       }
       -> {"choices": [{"message": {"content": "...", "reasoning_content": "..."}}]}

重要约束：sglang 在**服务器侧**读取 ``audio_data`` / ``audio_url`` 指向的
文件，因此要求音乐库与 sglang 服务共享文件系统（同机部署或挂载同一卷）。
若部署在不同机器，请用 NFS/SSHFS 挂载，或改用上传型实现
（HttpSongAnalysisClient）。

thinking budget 说明：设置 ``thinking_budget`` 时自动附带
``separate_reasoning`` 与 ``custom_logit_processor``；processor 字符串默认
取 ``Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()`` 的惯用值，
如你的 sglang 版本 ``to_str()`` 输出不同，用 ``custom_logit_processor``
参数显式覆盖（或本机装 sglang 后把 ``to_str()`` 结果传入）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from ..style_parser import DEFAULT_ANALYSIS_PROMPT, extract_styles_from_text
from .base import SongAnalysisClient, SongAnalysisError

logger = logging.getLogger(__name__)

DEFAULT_SGLANG_BASE = "http://localhost:30000"
DEFAULT_THINKING_PROCESSOR = "Qwen3InstructionInjectionThinkingBudgetLogitProcessor"

ApiMode = Literal["generate", "chat"]


class SglangMusicClient(SongAnalysisClient):
    """调用 sglang 音乐理解服务的分析客户端（音频传本机路径）。

    - ``api_mode="generate"``: sglang 原生 ``/generate``；
    - ``api_mode="chat"``: OpenAI 兼容 ``/v1/chat/completions``，
      可选 thinking 分离与 budget 控制；
    - 响应统一解析为工作流契约（``styles`` + ``description``），
      StyleParser 可直接确定性提取。
    """

    name = "sglang-music"

    def __init__(
        self,
        endpoint: str = DEFAULT_SGLANG_BASE,
        *,
        api_mode: ApiMode = "generate",
        prompt: str = DEFAULT_ANALYSIS_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        model_name: str = "default",
        thinking_budget: Optional[int] = None,
        custom_logit_processor: Optional[str] = None,
        timeout: float = 600.0,
        retries: int = 2,
        transport: Optional[Any] = None,
    ) -> None:
        super().__init__(timeout=timeout)
        if api_mode not in ("generate", "chat"):
            raise ValueError(f"api_mode 只支持 generate / chat，当前: {api_mode!r}")
        if not endpoint or not str(endpoint).startswith(("http://", "https://")):
            raise ValueError(f"endpoint 必须是 http(s) URL，当前: {endpoint!r}")

        self.endpoint = str(endpoint).rstrip("/")
        self.api_mode = api_mode
        self.prompt = prompt or DEFAULT_ANALYSIS_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.model_name = model_name
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None
        self.custom_logit_processor = (
            custom_logit_processor or DEFAULT_THINKING_PROCESSOR
        ) if self.thinking_budget is not None else custom_logit_processor

        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._url_generate = f"{self.endpoint}/generate"
        self._url_chat = f"{self.endpoint}/v1/chat/completions"
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            transport=transport if transport is not None else httpx.HTTPTransport(retries=max(0, retries)),
        )

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    def analyze(self, audio_path: Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise SongAnalysisError(f"音频文件不存在: {audio_path}")
        # 服务端按路径读文件，必须传服务端可见的绝对路径
        audio_url = str(audio_path.resolve())

        logger.info(
            "[sglang] %s POST %s (%s, max_new_tokens=%d)",
            audio_path.name,
            self._url_generate if self.api_mode == "generate" else self._url_chat,
            audio_url,
            self.max_new_tokens,
        )
        if self.api_mode == "generate":
            text, raw_response = self._analyze_via_generate(audio_url)
            reasoning = None
        else:
            text, reasoning, raw_response = self._analyze_via_chat(audio_url)

        payload: dict[str, Any] = {
            "status": "ok",
            "engine": self.name,
            "endpoint": self.endpoint,
            "api_mode": self.api_mode,
            "description": text.strip(),
            # 保留服务端原始响应（原始输出），供分析明细与排查使用
            "raw_response": raw_response,
        }
        if reasoning:
            payload["reasoning"] = reasoning
        styles = extract_styles_from_text(text)
        if styles:
            payload["styles"] = styles
        else:
            payload["message"] = (
                "未能从模型输出第一行解析出风格标签；已保留 description，"
                "可配置 --llm-model 让 StyleParser 用 LLM 兜底提取"
            )
        return payload

    # ------------------------------------------------------------------
    # generate 模式（基础音乐描述）
    # ------------------------------------------------------------------
    def _analyze_via_generate(self, audio_url: str) -> tuple[str, dict[str, Any]]:
        body = {
            "text": self.prompt,
            "audio_data": audio_url,
            "sampling_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }
        raw = self._post(self._url_generate, body, field_hint="text")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SongAnalysisError(f"sglang /generate 响应缺少 text 字段: {_preview(raw)}")
        return text, raw

    # ------------------------------------------------------------------
    # chat 模式（OpenAI 兼容 + thinking 分离 + budget 控制）
    # ------------------------------------------------------------------
    def _analyze_via_chat(self, audio_url: str) -> tuple[str, Optional[str], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.thinking_budget is not None:
            body["separate_reasoning"] = True
            body["custom_logit_processor"] = self.custom_logit_processor or DEFAULT_THINKING_PROCESSOR
            body["custom_params"] = {"thinking_budget": self.thinking_budget}

        raw = self._post(self._url_chat, body, field_hint="choices[0].message.content")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SongAnalysisError(f"sglang /v1/chat/completions 响应缺少 choices: {_preview(raw)}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise SongAnalysisError(f"响应缺少 content 字段: {_preview(raw)}")
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        if isinstance(reasoning, str) and reasoning.strip():
            logger.debug("[sglang] thinking 分离输出(%d 字符)已保留在 raw_response/reasoning", len(reasoning))
            return content, reasoning, raw
        return content, None, raw

    # ------------------------------------------------------------------
    # 传输
    # ------------------------------------------------------------------
    def _post(self, url: str, body: dict[str, Any], *, field_hint: str) -> dict[str, Any]:
        try:
            response = self._client.post(url, json=body, headers=self.headers)
        except httpx.HTTPError as exc:
            raise SongAnalysisError(f"请求 sglang 服务失败（{url}）: {type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise SongAnalysisError(
                f"sglang 服务返回 HTTP {response.status_code}（期望字段 {field_hint}）: "
                f"{response.text[:300]}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise SongAnalysisError(f"sglang 响应不是 JSON: {response.text[:300]}") from exc
        if not isinstance(payload, dict):
            raise SongAnalysisError(f"sglang 响应应为 JSON 对象，实为 {type(payload).__name__}")
        return payload

    def close(self) -> None:
        self._client.close()


def _preview(payload: Any, limit: int = 240) -> str:
    try:
        import json

        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    return text[:limit] + ("..." if len(text) > limit else "")
