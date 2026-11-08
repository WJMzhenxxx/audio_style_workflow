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

thinking budget 说明：设置 ``thinking_budget`` 时必须同时提供
``custom_logit_processor`` 的**序列化字符串**——即
``Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()`` 的 JSON 输出
（内含 dill 序列化的 callable）。解析顺序：显式参数 >
环境变量 ``SGLANG_CUSTOM_LOGIT_PROCESSOR`` > 本机 sglang 的 ``to_str()``。
三者都拿不到时**直接报错**，绝不把裸类名发给服务端：sglang 会
``orjson.loads`` 该字段，传类名会让 scheduler 抛 JSONDecodeError 崩溃。
另需服务端以 ``--enable-custom-logit-processor`` 启动。
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional, Sequence

import httpx

from ..style_parser import (
    DEFAULT_ANALYSIS_PROMPT,
    extract_genre_rationale,
    extract_musical_meta,
    extract_styles_from_text,
)
from .base import SongAnalysisClient, SongAnalysisError

logger = logging.getLogger(__name__)

DEFAULT_SGLANG_BASE = "http://localhost:30000"

#: OpenMOSS (MOSS-Music) sglang 分支里的 thinking budget 处理器类名。
#: 注意：请求里必须传 ``to_str()`` 产出的 **JSON 字符串**（内含 dill 序列化
#: 的 callable），不能传类名——服务端会用 ``orjson.loads`` 解析，传类名会直接
#: 让 sglang scheduler 崩溃。
THINKING_PROCESSOR_CLASS = "Qwen3InstructionInjectionThinkingBudgetLogitProcessor"
SGLANG_PROCESSOR_MODULE = "sglang.srt.sampling.custom_logit_processor"
#: 直接提供序列化串的环境变量（避免在命令行里粘贴超长字符串）
ENV_CUSTOM_LOGIT_PROCESSOR = "SGLANG_CUSTOM_LOGIT_PROCESSOR"

ApiMode = Literal["generate", "chat"]

#: 后缀 -> MIME（sglang 只认 ``data:`` 前缀，MIME 本身不参与解码，写对更规范）
_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".tta": "audio/x-tta",
    ".wma": "audio/x-ms-wma",
}
#: base64 内联超过该大小时提示（体积膨胀约 33%）
_BASE64_WARN_BYTES = 20 * 1024 * 1024
#: 纯文本 prompt 的 token 数上限估计；prompt_tokens 低于它说明音频没进模型
_TEXT_ONLY_PROMPT_TOKENS = 200


class ProcessorResolveError(ValueError):
    """无法得到可用的 custom_logit_processor 序列化串。"""


def _read_processor_value(value: str) -> str:
    """支持 ``@path/to/file`` 形式：从文件读取序列化串。"""
    value = (value or "").strip()
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProcessorResolveError(f"读取 custom_logit_processor 文件失败: {path}: {exc}") from exc
    return value


def validate_processor_string(value: str) -> str:
    """校验自定义 logit processor 是否是 sglang 能解析的 JSON 对象。

    服务端 ``CustomLogitProcessor.from_str`` 会执行 ``orjson.loads(json_str)``，
    因此非 JSON 字符串（例如误传类名）会让服务端抛异常甚至崩溃。这里提前拦截。
    """
    value = _read_processor_value(value)
    if not value:
        raise ProcessorResolveError("custom_logit_processor 不能为空")
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProcessorResolveError(
            "custom_logit_processor 不是合法 JSON（sglang 服务端会 orjson.loads 解析它）。"
            "不要传类名，请传 `Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()` "
            f"的输出。解析错误: {exc}"
        ) from exc
    if not isinstance(data, dict) or not data:
        raise ProcessorResolveError(
            f"custom_logit_processor 应为非空 JSON 对象，实际为 {type(data).__name__}"
        )
    return value


def _load_processor_from_sglang() -> Optional[str]:
    """本机若装了 OpenMOSS sglang 分支，直接调用 ``to_str()`` 获取序列化串。"""
    try:
        module = __import__(SGLANG_PROCESSOR_MODULE, fromlist=[THINKING_PROCESSOR_CLASS])
    except Exception:
        return None
    processor_cls = getattr(module, THINKING_PROCESSOR_CLASS, None)
    if processor_cls is None:
        return None
    try:
        return processor_cls.to_str()
    except Exception as exc:  # pragma: no cover - 取决于本机 sglang 状态
        logger.warning("调用 %s.to_str() 失败: %s", THINKING_PROCESSOR_CLASS, exc)
        return None


def resolve_thinking_processor(explicit: Optional[str] = None) -> str:
    """得到 thinking budget 处理器的序列化字符串（并校验为 JSON）。

    解析顺序：显式参数 > 环境变量 ``SGLANG_CUSTOM_LOGIT_PROCESSOR`` >
    本机 sglang 的 ``to_str()``。全部不可用时抛 :class:`ProcessorResolveError`，
    而不是把类名发给服务端（那会导致 sglang 崩溃）。
    """
    candidate = explicit or os.environ.get(ENV_CUSTOM_LOGIT_PROCESSOR)
    if candidate:
        return validate_processor_string(candidate)

    from_sglang = _load_processor_from_sglang()
    if from_sglang:
        return validate_processor_string(from_sglang)

    raise ProcessorResolveError(
        "无法获取 thinking budget 的 custom_logit_processor 序列化串。\n"
        "该值是 sglang（OpenMOSS/moss-audio 分支）里 "
        "`Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()` 的 JSON 输出，"
        "内含 dill 序列化数据，必须由与服务端一致的 sglang 版本生成，不能手写类名"
        "（否则会让 sglang scheduler 崩溃）。\n"
        "解决办法（二选一）：\n"
        "  1) 在装有该 sglang 分支的环境执行：\n"
        "       python -c \"from sglang.srt.sampling.custom_logit_processor import "
        "Qwen3InstructionInjectionThinkingBudgetLogitProcessor as P; "
        "print(P.to_str())\"\n"
        f"     把输出通过 --sglang-custom-logit-processor 传入、或写入文件后用 "
        f"--sglang-custom-logit-processor @processor.txt，或设置环境变量 "
        f"{ENV_CUSTOM_LOGIT_PROCESSOR}；\n"
        "  2) 去掉 thinking budget（不传 --sglang-thinking-budget）。\n"
        "另请确认服务端以 --enable-custom-logit-processor 启动。"
    )


#: 服务端拒绝 custom logit processor 时错误信息里常见的片段
_UNSUPPORTED_PROCESSOR_HINTS = (
    "custom_logit_processor",
    "custom logit processor",
    "enable-custom-logit-processor",
    "enable_custom_logit_processor",
)


def _looks_like_unsupported_processor(message: str) -> bool:
    """判断错误是否由「服务端未开启 custom logit processor」引起。"""
    low = (message or "").lower()
    return any(h in low for h in _UNSUPPORTED_PROCESSOR_HINTS)


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
        audio_path_map: Optional[Sequence[tuple[str, str]]] = None,
        audio_base64: bool = False,
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
        # thinking_budget 需要合法的 processor 序列化串；解析/校验失败立即报错，
        # 避免把非法值发给服务端导致其崩溃。
        self.custom_logit_processor: Optional[str] = None
        if self.thinking_budget is not None:
            self.custom_logit_processor = resolve_thinking_processor(custom_logit_processor)
        #: 是否真正向服务端下发 thinking budget；若服务端未开
        #: ``--enable-custom-logit-processor`` 而被拒绝，会自动降级为 False。
        self.thinking_budget_applied = self.thinking_budget is not None

        #: 本地路径前缀 -> 服务端可见路径前缀；服务端读不到本地路径时需要它
        self.audio_path_map: list[tuple[str, str]] = [
            (os.path.normpath(str(local)), str(remote).rstrip("/\\") or "/")
            for local, remote in (audio_path_map or [])
        ]
        #: 为 True 时把音频以 data URL（base64）内联进请求，无需共享文件系统
        self.audio_base64 = bool(audio_base64)

        #: 连续返回相同文本的计数（用于诊断"音频没被读到"）
        self._last_text_digest: Optional[str] = None
        self._identical_streak = 0
        self._identical_warned = False
        self._last_sent_path: Optional[str] = None
        #: 最近一次 chat 响应的 finish_reason（"stop" / "length" / None）
        self._last_finish_reason: Optional[str] = None
        #: 最近一次 chat 响应的 usage.prompt_tokens
        self._last_prompt_tokens: Optional[int] = None
        self._audio_consumed_warned = False
        #: 该服务的 chat 端点不吃 audio_url 时置 True，之后直接走 /generate
        self._chat_audio_broken = False

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
    def _audio_payload(self, audio_path: Path) -> str:
        """把音频转成发给服务端的字符串。

        - 默认：服务端可见的路径（``audio_path_map`` 映射后）；
        - ``audio_base64=True``：``data:<mime>;base64,<...>`` 内联，
          不依赖共享文件系统。sglang 的 ``load_audio`` 只认 ``data:`` 前缀，
          裸 base64 会被当成文件路径，所以这里必须带 ``data:`` 头。
        """
        audio_path = Path(audio_path)
        if not self.audio_base64:
            return self._server_path(audio_path)

        size = audio_path.stat().st_size
        if size > _BASE64_WARN_BYTES:
            logger.warning(
                "[sglang] %s 体积 %.1f MB，base64 内联后约 %.1f MB，请求会明显变慢；"
                "大文件建议改用 --sglang-path-map + 共享文件系统",
                audio_path.name, size / 1e6, size * 4 / 3 / 1e6,
            )
        mime = _AUDIO_MIME.get(audio_path.suffix.lower(), "application/octet-stream")
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _describe_audio(audio_payload: str) -> str:
        """日志/诊断里用的短描述（避免把几 MB 的 base64 打进日志）。"""
        if audio_payload.startswith("data:"):
            head, _, body = audio_payload.partition(",")
            return f"{head},<{len(body)} base64 chars>"
        return audio_payload

    def _server_path(self, audio_path: Path) -> str:
        """把本地音频路径换算成服务端可见的路径。

        服务端与客户端不共享文件系统时（例如客户端 Windows、服务端 Linux），
        直接用本地绝对路径必然读不到。可用 ``audio_path_map`` 做前缀映射::

            [("D:\\\\music", "/data/music")]  ->  D:\\music\\a.flac  =>  /data/music/a.flac
        """
        resolved = str(Path(audio_path).resolve())
        normalized = os.path.normpath(resolved)
        for local_root, remote_root in self.audio_path_map:
            if normalized == local_root:
                relative = ""
            elif normalized.startswith(local_root + os.sep):
                relative = os.path.relpath(normalized, local_root)
            else:
                continue
            remote = PurePosixPath(remote_root)
            if relative:
                remote = remote / PurePosixPath(*Path(relative).parts)
            return str(remote)
        return resolved

    def _check_identical_streak(self, text: str) -> None:
        """连续多首返回完全相同的文本，通常意味着服务端根本没读到音频。"""
        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if digest == self._last_text_digest:
            self._identical_streak += 1
        else:
            self._last_text_digest = digest
            self._identical_streak = 1
        if self._identical_streak >= 2 and not self._identical_warned:
            self._identical_warned = True
            preview = " ".join(text.split())[:200]
            logger.warning(
                "[sglang] 连续 %d 首歌返回完全相同的文本（%d 字符），疑似服务端没有读到音频。"
                "请检查音频路径对服务端是否可见（客户端 Windows 路径 vs 服务端 Linux 路径？），"
                "或用 --sglang-path-map 本地前缀=服务端前缀 做映射；"
                "也可用 --sglang-audio-base64 内联发送。"
                "已发送: %s；文本预览: %s",
                self._identical_streak,
                len(text),
                self._last_sent_path or "(未知)",
                preview or "(空)",
            )

    def _audio_missing(self) -> bool:
        """用 usage.prompt_tokens 判断音频有没有真正进入模型。

        纯文本 prompt 约 100 token；一段音频至少要几百个 token。若
        prompt_tokens 仍然很小，说明服务端没解析请求里的音频。
        """
        tokens = self._last_prompt_tokens
        if tokens is None:
            return False
        return tokens < _TEXT_ONLY_PROMPT_TOKENS

    def _disable_thinking_budget(self, reason: str) -> None:
        """服务端拒绝该字段时自动降级（只提示一次）。"""
        if not self.thinking_budget_applied:
            return
        self.thinking_budget_applied = False
        logger.warning(
            "[sglang] 服务端未启用 custom logit processor，已自动降级为**不带 thinking budget**重试。"
            "如需 budget 控制，请用 --enable-custom-logit-processor 启动服务端。原因: %s",
            (reason or "")[:300],
        )

    def analyze(self, audio_path: Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise SongAnalysisError(f"音频文件不存在: {audio_path}")
        # 服务端按路径/内联数据读音频：路径模式需换算成服务端可见路径
        audio_url = self._audio_payload(audio_path)
        self._last_sent_path = self._describe_audio(audio_url)

        # chat 端点在部分 sglang 版本里不会把 audio_url 转成音频输入；
        # 一旦发现就记住并改走 /generate（audio_data 路径）。
        use_chat = self.api_mode == "chat" and not self._chat_audio_broken
        if use_chat:
            logger.info(
                "[sglang] %s POST %s (%s, max_new_tokens=%d)",
                audio_path.name, self._url_chat, self._last_sent_path, self.max_new_tokens,
            )
            text, reasoning, raw_response, content_source = self._analyze_via_chat(audio_url)
            if self._audio_missing():
                self._chat_audio_broken = True
                logger.warning(
                    "[sglang] 该服务的 /v1/chat/completions 没有把音频送进模型"
                    "（prompt_tokens=%s，纯文本规模），已自动改用 /generate（audio_data）。"
                    "后续歌曲将直接走 generate，无需再试 chat。",
                    self._last_prompt_tokens,
                )
                use_chat = False
        if not use_chat:
            logger.info(
                "[sglang] %s POST %s (%s, max_new_tokens=%d)",
                audio_path.name, self._url_generate, self._last_sent_path, self.max_new_tokens,
            )
            text, raw_response = self._analyze_via_generate(audio_url)
            reasoning = None
            content_source = "text"

        used_mode = "chat" if use_chat else "generate"
        self._check_identical_streak(text)

        payload: dict[str, Any] = {
            "status": "ok",
            "engine": self.name,
            "endpoint": self.endpoint,
            "api_mode": used_mode,
            "description": text.strip(),
            # content / reasoning_content（思考未闭合时只能从后者取答案）
            "content_source": content_source,
            # 保留服务端原始响应（原始输出），供分析明细与排查使用
            "raw_response": raw_response,
        }
        if reasoning:
            payload["reasoning"] = reasoning
        if self.thinking_budget is not None:
            payload["thinking_budget_applied"] = bool(self.thinking_budget_applied)
        if used_mode == "chat" and self._last_finish_reason:
            payload["finish_reason"] = self._last_finish_reason
        if self._last_prompt_tokens is not None:
            payload["prompt_tokens"] = self._last_prompt_tokens
        styles = extract_styles_from_text(text)
        if styles:
            payload["styles"] = styles
        else:
            payload["message"] = (
                "未能从模型输出第一行解析出风格标签；已保留 description，"
                "可配置 --llm-model 让 StyleParser 用 LLM 兜底提取"
            )
        payload.update(extract_musical_meta(text))
        rationale = extract_genre_rationale(text)
        if rationale:
            payload["genre_rationale"] = rationale
        return payload

    # ------------------------------------------------------------------
    # generate 模式（基础音乐描述）
    # ------------------------------------------------------------------
    def _analyze_via_generate(self, audio_url: str) -> tuple[str, dict[str, Any]]:
        sampling: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        body: dict[str, Any] = {
            "text": self.prompt,
            "audio_data": audio_url,
            "sampling_params": sampling,
        }
        if self.thinking_budget_applied:
            # MOSS 的 /generate 也支持 budget：custom_params 放 sampling_params，
            # processor 放请求顶层（见 MOSS-Music 用法说明）
            sampling["custom_params"] = {"thinking_budget": self.thinking_budget}
            body["custom_logit_processor"] = self.custom_logit_processor

        try:
            raw = self._post(self._url_generate, body, field_hint="text")
        except SongAnalysisError as exc:
            if not (self.thinking_budget_applied and _looks_like_unsupported_processor(str(exc))):
                raise
            self._disable_thinking_budget(str(exc))
            sampling.pop("custom_params", None)
            body.pop("custom_logit_processor", None)
            raw = self._post(self._url_generate, body, field_hint="text")

        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SongAnalysisError(f"sglang /generate 响应缺少 text 字段: {_preview(raw)}")
        meta = raw.get("meta_info") if isinstance(raw.get("meta_info"), dict) else {}
        prompt_tokens = meta.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            self._last_prompt_tokens = prompt_tokens
        return text, raw

    # ------------------------------------------------------------------
    # chat 模式（OpenAI 兼容 + thinking 分离 + budget 控制）
    # ------------------------------------------------------------------
    def _analyze_via_chat(
        self, audio_url: str
    ) -> tuple[str, Optional[str], dict[str, Any], str]:
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
        if self.thinking_budget_applied:
            body["separate_reasoning"] = True
            body["custom_logit_processor"] = self.custom_logit_processor
            body["custom_params"] = {"thinking_budget": self.thinking_budget}

        try:
            raw = self._post(self._url_chat, body, field_hint="choices[0].message.content")
        except SongAnalysisError as exc:
            # 服务端未用 --enable-custom-logit-processor 启动时会拒绝该字段。
            # 这是服务端的安全开关，客户端无法绕过；自动降级为不带 budget 重试。
            if not (self.thinking_budget_applied and _looks_like_unsupported_processor(str(exc))):
                raise
            self._disable_thinking_budget(str(exc))
            for key in ("separate_reasoning", "custom_logit_processor", "custom_params"):
                body.pop(key, None)
            raw = self._post(self._url_chat, body, field_hint="choices[0].message.content")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SongAnalysisError(f"sglang /v1/chat/completions 响应缺少 choices: {_preview(raw)}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        self._last_finish_reason = finish_reason

        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            if isinstance(prompt_tokens, int):
                self._last_prompt_tokens = prompt_tokens

        has_content = isinstance(content, str) and content.strip()
        has_reasoning = isinstance(reasoning, str) and reasoning.strip()

        if has_content:
            return content, (reasoning if has_reasoning else None), raw, "content"

        if has_reasoning:
            # thinking 模型常见：思考未闭合（思考段吃光 max_tokens）时 content 为 null，
            # 真正的答案落在 reasoning_content 里。直接失败会白白丢掉可用结果。
            logger.warning(
                "[sglang] 响应 content 为空（finish_reason=%s），回退使用 reasoning_content"
                "（%d 字符）；如需正常 content，可增大 --max-new-tokens 或启用 thinking budget 强制思考收尾",
                finish_reason,
                len(reasoning),
            )
            return reasoning, reasoning, raw, "reasoning_content"

        raise SongAnalysisError(
            f"响应缺少 content 与 reasoning_content 字段: {_preview(raw)}"
        )

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
