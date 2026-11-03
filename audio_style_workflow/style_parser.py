"""从分析端点响应中解析歌曲风格（两级策略）。

1. **确定性 JSON 提取**：在常见键名（styles/style/genres/genre/music_style/
   tags）与常见嵌套层（result/analysis/data/payload/output）中查找风格字段，
   字符串按 ``, ; / | 、`` 等分隔符拆分；
2. **LLM 兑底**：注入任意 LangChain ChatModel 后，对上述找不到风格字段的
   响应（自然语言等非结构化内容）用结构化输出提取风格列表。

另含本地模型实现共用的辅助件：

- :data:`DEFAULT_ANALYSIS_PROMPT` —— 引导生成式音乐模型把风格标签写在
  输出首行，便于确定性解析；
- :func:`extract_styles_from_text` —— 从模型自由文本输出的首行提取
  风格标签列表（启发式，无需 LLM）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: 顶层风格字段候选（比较时归一化为小写下划线）
_STYLE_KEYS = (
    "styles", "style", "genres", "genre",
    "music_style", "music_styles", "music_genre", "music_genres",
    "tags",
)
#: 允许向下钻一层的容器字段
_NESTED_KEYS = ("result", "analysis", "data", "payload", "output", "body")
#: 字符串风格值的分隔符（保留 "R&B" 一类不可拆的复合词）
_SPLIT_RE = ",|;|/|\\||、|·|\\s{2,}"

_MAX_STYLES = 8


# ======================================================================
# 本地/自托管生成式音乐模型（MusicFlamingo、sglang 部署模型等）共用辅助件
# ======================================================================
#: 引导模型把风格标签写在第一行（便于确定性解析），后续行输出完整描述
DEFAULT_ANALYSIS_PROMPT = (
    "Listen to this track and analyze it. On the FIRST line, list its musical "
    "styles/genres as a short comma-separated line (for example: Pop, R&B, Jazz). "
    "Then describe in full detail the genre, tempo, key, instruments, production "
    "style, and the overall mood it creates."
)

#: 解析风格标签的启发式参数
_MAX_LABELS = 8
_MAX_LABEL_LEN = 48
_MAX_LABEL_WORDS = 4  # 真实风格标签很少超过 4 个词；超长子句视为描述句
_PREFIX_RE = re.compile(
    r"^\s*(styles?|genres?|genres?\s*styles?|music\s*styles?|tags?|音乐风格|风格|流派|标签)\s*[:：\-—]\s*",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"[,，;；、/|\t]|\s{2,}")
_TRAILING_PUNCT_RE = re.compile(r"[.。!！?？\"'\”’]+\s*$")


def extract_styles_from_text(text: str, *, limit: int = _MAX_LABELS) -> list[str]:
    """从生成式模型输出的第一行提取风格标签（确定性解析，无需 LLM）。

    规则：

    1. 取第一条非空行；剥掉 "Styles:" / "Genres:" / "风格:" 等前缀；
    2. 按 `, ， ; ； 、 / |` 及连续空格拆分；
    3. 每个标签长度须 <= 48 字符、词数 <= 4（排除被误拆的完整句子）；
    4. 标签数为 1~8 才可信（过长通常是普通描述句）。
    """
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return []
    first_line = _PREFIX_RE.sub("", first_line)

    labels: list[str] = []
    for raw in _SPLIT_RE.split(first_line):
        label = _TRAILING_PUNCT_RE.sub("", raw.strip().strip('"“”').strip())
        if not label or len(label) > _MAX_LABEL_LEN or len(label.split()) > _MAX_LABEL_WORDS:
            return []  # 超长/多词片段 -> 整行大概率是描述句，放弃
        if label not in labels:
            labels.append(label)
    if not (1 <= len(labels) <= _MAX_LABELS):
        return []
    return labels[:limit]


class StyleParseError(RuntimeError):
    """无法从响应中确定歌曲风格。"""


class StyleList(BaseModel):
    """LLM 兜底解析的结构化输出模式。"""

    styles: list[str] = Field(
        default_factory=list,
        description="歌曲风格标签列表，例如 ['Pop', 'R&B']；无法判断时为空列表",
    )


def _norm_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _split_style_text(value: str) -> list[str]:
    import re

    parts = re.split(_SPLIT_RE, value)
    found: list[str] = []
    for part in parts:
        text = part.strip().strip(" \"'[]")
        if text and text != "-" and text not in found:
            found.append(text)
    return found


def _extract_styles(payload: Mapping[str, Any]) -> list[str]:
    """确定性提取：返回风格列表（空列表表示没找到）。"""
    style_key_set = set(_STYLE_KEYS)
    for key, value in payload.items():
        if _norm_key(key) in style_key_set:
            if isinstance(value, (list, tuple)):
                styles: list[str] = []
                for item in value:
                    text = str(item).strip()
                    if text and text not in styles:
                        styles.append(text)
            elif isinstance(value, str):
                styles = _split_style_text(value)
            elif isinstance(value, (int, float)):
                continue
            else:
                continue
            if styles:
                return styles
    for nested_key in _NESTED_KEYS:
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            styles = _extract_styles(nested)
            if styles:
                return styles
    return []


def _extract_confidence(payload: Mapping[str, Any]) -> Optional[float]:
    for key in ("confidence", "score", "probability", "prob"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if 0.0 <= float(value) <= 1.0:
                return float(value)
    return None


class StyleParser:
    """响应 -> 风格列表的解析器。

    :param llm: 可选的 LangChain ChatModel（任何实现了
        ``with_structured_output`` 的语言模型）。不传时仅确定性解析。
    :param max_styles: 单首歌最多保留的风格标签数。
    """

    def __init__(self, llm: Optional[Any] = None, *, max_styles: int = _MAX_STYLES) -> None:
        self._llm = llm
        self.max_styles = max(1, int(max_styles))

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None

    def parse(
        self,
        payload: Mapping[str, Any],
        *,
        source_name: str = "",
    ) -> tuple[list[str], str, Optional[float]]:
        """解析响应，返回 ``(styles, parser_name, confidence)``。

        :raises StyleParseError: 两级策略都无法提取风格
        """
        styles = _extract_styles(payload)
        if styles:
            return styles[: self.max_styles], "json", _extract_confidence(payload)

        if self._llm is None:
            preview = _preview(payload)
            raise StyleParseError(
                f"响应中没有可识别的风格字段（styles/genres/style/tags…），"
                f"且未启用 LLM 兜底解析: {preview}"
            )
        return self._parse_with_llm(payload, source_name=source_name)

    # ------------------------------------------------------------------
    def _parse_with_llm(
        self, payload: Mapping[str, Any], *, source_name: str = ""
    ) -> tuple[list[str], str, Optional[float]]:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是音乐元数据提取助手。分析服务返回了以下原始内容（可能不是结构化"
                    "JSON，含中文/英文自然语言）。请从中提取该歌曲的风格标签列表。"
                    "风格名保持原始语言的通行写法；确实无法判断就返回空列表。"
                    "不要捏造响应中不存在的风格。",
                ),
                (
                    "human",
                    "歌曲: {name}\n分析服务响应:\n```\n{response}\n```",
                ),
            ]
        )
        messages = prompt.format_messages(
            name=source_name or "(unknown)",
            response=_preview(payload, limit=4000),
        )
        try:
            structured = self._llm.with_structured_output(StyleList)
            result = structured.invoke(messages)
        except Exception as exc:
            raise StyleParseError(f"LLM 风格提取失败（{exc}）") from exc

        styles = [s.strip() for s in (result.styles or []) if s and s.strip()]
        if not styles:
            raise StyleParseError("LLM 认为响应中没有可提取的歌曲风格")
        return styles[: self.max_styles], "llm", None


def _preview(payload: Any, limit: int = 240) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
