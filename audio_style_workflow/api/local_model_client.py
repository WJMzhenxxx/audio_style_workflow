"""基于本地 MusicFlamingo 模型（transformers）的歌曲分析端点实现。

完全离线：不经过任何 HTTP 服务，直接在本机/本卡上推理。
调用链与 HuggingFace 官方示例一致::

    processor.apply_chat_template(conversation, tokenize=True,
                                  add_generation_prompt=True, return_dict=True)
        .to(model.device)
    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    model.generate(**inputs, max_new_tokens=...)
    processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True)

返回契约与其余实现一致::

    {
      "status": "ok",
      "styles": ["Pop", "R&B"],       # 从模型输出第一行解析的风格标签
      "description": "<模型的完整分析文本>",
      "engine": "music-flamingo",
      "model_id": "nvidia/music-flamingo-2601-hf"
    }

- 若 prompt 引导模型把风格标签列在第一行，则走 StyleParser 的
  确定性 JSON 提取路径（零 LLM 成本）；
- 若第一行解析不到标签，styles 缺省、保留 description，
  配置 ``--llm-model`` 后由 LLM 兜底从 description 提取。

依赖（可选安装，见 requirements-local.txt）::

    pip install -r requirements-local.txt
    # 国内镜像加速下载权重:
    set HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..style_parser import DEFAULT_ANALYSIS_PROMPT, extract_styles_from_text
from .base import SongAnalysisClient, SongAnalysisError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/music-flamingo-2601-hf"

#: 兼容旧导入路径：风格 prompt 与提取函数已提升到 style_parser 共用
DEFAULT_PROMPT = DEFAULT_ANALYSIS_PROMPT

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT",
    "LocalMusicFlamingoClient",
    "extract_styles_from_text",
]


class LocalMusicFlamingoClient(SongAnalysisClient):
    """调用本地 MusicFlamingo 模型分析歌曲（离线推理）。

    - 模型**懒加载**：首次 ``analyze`` 时才下载/载入权重；
    - ``generate`` 非线程安全，内部持锁串行化（工作流开 --concurrency
      也只会排队，不会显存雪崩）；
    - 与 HTTP 实现同构，可无缝替换接入工作流。
    """

    name = "music-flamingo"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 500,
        device_map: str = "auto",
        timeout: float = 3600.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.device_map = device_map
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """下载/载入模型与处理器（幂等，可在首次 analyze 前手动预热）。"""
        if self._loaded:
            return
        model_cls, processor_cls = self._import_backend()
        logger.info("[local] 加载模型 %s（device_map=%s）...", self.model_id, self.device_map)
        try:
            self._processor = processor_cls.from_pretrained(self.model_id)
            self._model = model_cls.from_pretrained(self.model_id, device_map=self.device_map)
        except Exception as exc:
            self._model = self._processor = None
            raise SongAnalysisError(
                f"加载模型失败（{self.model_id}）: {exc}；"
                f"国内可用镜像: set HF_ENDPOINT=https://hf-mirror.com"
            ) from exc
        self._loaded = True
        logger.info("[local] 模型就绪")

    def _import_backend(self) -> tuple[Any, Any]:
        """分离出 import 便于测试注入 fake 后端。"""
        try:
            from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
        except ImportError as exc:
            raise SongAnalysisError(
                "缺少本地模型依赖：请先 pip install -r requirements-local.txt"
                "（transformers / torch / accelerate）"
            ) from exc
        return MusicFlamingoForConditionalGeneration, AutoProcessor

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    def analyze(self, audio_path: Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise SongAnalysisError(f"音频文件不存在: {audio_path}")
        self.load()

        logger.info("[local] 推理 %s（max_new_tokens=%d）", audio_path.name, self.max_new_tokens)
        with self._lock:
            try:
                description = self._generate(audio_path)
            except SongAnalysisError:
                raise
            except Exception as exc:
                raise SongAnalysisError(f"本地模型推理失败（{audio_path.name}）: {exc}") from exc

        payload: dict[str, Any] = {
            "status": "ok",
            "engine": self.name,
            "model_id": self.model_id,
            "description": description.strip(),
        }
        styles = extract_styles_from_text(description)
        if styles:
            payload["styles"] = styles
        else:
            payload["message"] = (
                "未能从模型输出第一行解析出风格标签；已保留 description，"
                "可配置 --llm-model 让 StyleParser 用 LLM 兜底提取"
            )
        return payload

    # ------------------------------------------------------------------
    # 推理内核（照 HF 官方调用链）
    # ------------------------------------------------------------------
    def _generate(self, audio_path: Path) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._model.device)
        inputs["input_features"] = inputs["input_features"].to(self._model.dtype)

        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        decoded = self._processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return str(decoded[0]).strip()
