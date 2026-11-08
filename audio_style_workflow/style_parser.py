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
#: 引导模型把风格/速度/调性/理由写在固定位置（便于确定性解析），后续输出完整描述
DEFAULT_ANALYSIS_PROMPT = (
    "Listen to this track and analyze it. Reply in this exact format:\n"
    "Line 1: its musical styles/genres as a short comma-separated list "
    "(for example: Pop, R&B, Jazz).\n"
    "Line 2: BPM: <integer beats per minute, e.g. 120>\n"
    "Line 3: Tempo: <short tempo descriptor, e.g. Slow / Moderato / Fast>\n"
    "Line 4: Key: <key signature, e.g. C major / A minor>\n"
    "Line 5: Mood: <one or two mood/atmosphere words, e.g. Warm, Dreamy, Melancholic>\n"
    "Line 6: Score: <integer 0-10 rating, 10 being best>\n"
    "Line 7: Review: <one sentence overall review>\n"
    "Line 8: Why: <one or two sentences explaining why these styles fit this track>\n"
    "Then leave one blank line and describe in detail the genre, instruments, "
    "production style and the overall mood it creates."
)

#: 中文版 prompt：风格名 / Tempo / Key 保持英文，Mood 与 Why 及正文用中文。
#: 这样能沿用通行写法（Dream Pop、Moderato、C major），也便于按风格检索。
DEFAULT_ANALYSIS_PROMPT_ZH = (
    "请聆听这段音频并分析。严格按下面 8 行输出，不要加任何多余前缀：\n"
    "<风格标签，用英文书写，逗号分隔，例如 Pop, R&B, Dream Pop, Synth Pop>\n"
    "BPM: <每分钟节拍数，整数，例如 120>\n"
    "Tempo: <速度描述，用英文，例如 Slow / Moderato / Fast>\n"
    "Key: <调性，用英文，例如 C major / A minor>\n"
    "Mood: <用中文写一两个词描述氛围，例如 温暖, 舒缓, 忧郁, 空灵>\n"
    "Score: <0-10 的整数评分，10 为最佳>\n"
    "Review: <用中文写一句话总体评价这首歌>\n"
    "Why: <用中文写一两句话，说明为什么判定为这些风格>\n"
    "然后空一行，用中文详细描述曲风、配器、制作与整体情绪；"
    "描述里提到风格名、速度术语、调性时仍保留英文原词。"
)

#: 语言 -> prompt 模板
ANALYSIS_PROMPTS = {
    "en": DEFAULT_ANALYSIS_PROMPT,
    "zh": DEFAULT_ANALYSIS_PROMPT_ZH,
}


def get_analysis_prompt(lang: str = "en") -> str:
    """按语言取分析 prompt（``zh`` / ``en``）。"""
    key = (lang or "en").strip().lower()
    if key not in ANALYSIS_PROMPTS:
        raise ValueError(f"未知的 prompt 语言: {lang!r}（可选: {', '.join(ANALYSIS_PROMPTS)}）")
    return ANALYSIS_PROMPTS[key]

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


#: 指令回声里常见的"字段名"（单/双词，也要当成前缀剥掉）
_ECHO_HEAD_WORDS = {
    "list", "output", "answer", "result", "format", "response",
    "genres", "genre", "styles", "style", "tags", "tag",
    "风格", "流派", "标签", "输出", "结果", "答案",
}
#: "第1行" / "line 2" / "第N条" 这类编号前缀
_ECHO_HEAD_RE = re.compile(r"^(?:第[一二三四五六七八九十\d]+[行条点]|line\s*\d+|row\s*\d+)$", re.IGNORECASE)
#: 行首列表编号：``1.`` / ``2、`` / ``-`` / ``•``
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[.)、]|[-•*])\s*")
#: 标签里出现这些词，基本是回声的说明文字，丢弃
_INSTRUCTIONY_RE = re.compile(
    r"(只写|不要|例如|第一行|第[一二三四五六七八九十\d]+行|输出|格式|分隔|前缀|"
    r"风格标签|comma|separated|short\s+list|line\s*\d+)",
    re.IGNORECASE,
)


def _strip_instruction_echo(line: str) -> str:
    """去掉模型回声的指令前缀。

    生成式模型常把 prompt 里的说明原样接续，例如::

        its musical styles/genres as a short comma-separated list: Pop, R&B
        list: Pop, R&B
        第1行：流行, 节奏布鲁斯

    冒号前是一整句说明（词数多）、一个指令字段名（list/output/…）或
    ``第N行`` 这类编号时，冒号后才是真正答案。普通单个字段名
    （``Tempo: 100``）不会被误切。
    """
    line = _LIST_MARKER_RE.sub("", line)
    if ":" not in line and "：" not in line:
        return line
    separator = ":" if ":" in line else "："
    head, _, tail = line.rpartition(separator)
    tail = tail.strip()
    if not tail:
        return line
    head_text = head.strip()
    if (
        len(head_text.split()) >= 3
        or head_text.lower() in _ECHO_HEAD_WORDS
        or _ECHO_HEAD_RE.match(head_text)
        or _INSTRUCTIONY_RE.search(head_text)
    ):
        return _LIST_MARKER_RE.sub("", tail)
    return line


def extract_styles_from_text(text: str, *, limit: int = _MAX_LABELS) -> list[str]:
    """从生成式模型输出的第一行提取风格标签（确定性解析，无需 LLM）。

    规则：

    1. 取第一条非空行；剥掉 "Styles:" / "Genres:" / "风格:" 等前缀，
       以及回声的指令前缀（``...list:``）；
    2. 按 `, ， ; ； 、 / |` 及连续空格拆分；
    3. 每个标签长度须 <= 48 字符、词数 <= 4（排除被误拆的完整句子）；
    4. 标签数为 1~8 才可信（过长通常是普通描述句）。
    """
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return []
    first_line = _PREFIX_RE.sub("", first_line)
    first_line = _strip_instruction_echo(first_line)

    labels: list[str] = []
    for raw in _SPLIT_RE.split(first_line):
        label = _TRAILING_PUNCT_RE.sub("", raw.strip().strip('"“”（）()').strip())
        if not label:
            continue
        # 回声的说明文字（"只写风格标签" / "comma-separated" 等）直接丢弃
        if _INSTRUCTIONY_RE.search(label):
            continue
        if len(label) > _MAX_LABEL_LEN or len(label.split()) > _MAX_LABEL_WORDS:
            return []  # 超长/多词片段 -> 整行大概率是描述句，放弃
        if label not in labels:
            labels.append(label)
    if not (1 <= len(labels) <= _MAX_LABELS):
        return []
    return labels[:limit]


# ----------------------------------------------------------------------
# 音乐属性：BPM / Tempo / Key
# ----------------------------------------------------------------------
#: 单个属性的最大长度（放进标签不宜过长）
_MAX_META_LEN = 48
_BPM_MIN, _BPM_MAX = 20, 400

#: 行首标签形式：``BPM: 120`` / ``速度：120``
_BPM_LABEL_RE = re.compile(
    r"^\s*(?:bpm|beats?\s*per\s*minute|速度|节拍)\s*[:：=\-—]?\s*(\d{2,3})\b",
    re.IGNORECASE,
)
#: 行内形式：``120 BPM`` / ``around 128 bpm``
_BPM_INLINE_RE = re.compile(r"(\d{2,3})\s*(?:bpm|beats?\s*per\s*minute)\b", re.IGNORECASE)
#: ``Tempo: Moderato`` / ``速度：慢板``
_TEMPO_LABEL_RE = re.compile(
    r"^\s*(?:tempo|速度|节奏|节拍速度)\s*[:：=\-—]\s*(.+?)\s*$", re.IGNORECASE
)
#: ``Key: C major`` / ``调性：A小调``
_KEY_LABEL_RE = re.compile(
    r"^\s*(?:key|tonality|调性|调号|调)\s*[:：=\-—]\s*(.+?)\s*$", re.IGNORECASE
)
#: 纯 BPM 值（``120`` / ``120 BPM``），出现在 Tempo 标签后时按 BPM 处理
_BPM_VALUE_RE = re.compile(r"^\s*(\d{2,3})(?:\s*bpm)?\s*$", re.IGNORECASE)
#: 文本里直接出现的调性（``C major`` / ``F# minor`` / ``降B大调``）
_KEY_INLINE_RE = re.compile(
    r"\b([A-Ga-g][#b♯♭]?\s*(?:major|minor|maj|min))\b"
    r"|([A-Ga-g][#b♯♭]?\s*(?:大调|小调))"
)


def _clean_meta_line(line: str) -> str:
    """去掉 Markdown 装饰（``**``、``__``、反引号）与列表符号。"""
    text = line.strip()
    for token in ("**", "__", "`"):
        text = text.replace(token, "")
    return text.strip().lstrip("-•*>").strip()


def _clean_meta_value(value: str) -> str:
    return value.strip().strip("\"'“”‘’").strip().rstrip("。.,，;；").strip()


def _valid_bpm(value: int) -> Optional[int]:
    return value if _BPM_MIN <= value <= _BPM_MAX else None


def _looks_like_key(value: str) -> bool:
    if not value or len(value) > _MAX_META_LEN:
        return False
    if any(token in value for token in ("大调", "小调", "大調", "小調", "调", "調")):
        return True
    return bool(re.search(r"[A-Ga-g][#b♯♭]?", value)) and bool(
        re.search(r"major|minor|maj|min", value, re.IGNORECASE)
    )


def extract_musical_meta(text: str) -> dict[str, Any]:
    """从模型输出里提取 BPM / Tempo / Key（确定性解析，无需 LLM）。

    支持带标签的写法（``BPM: 120`` / ``Tempo: 慢板`` / ``Key: C major``，
    含中文标签 ``速度:`` / ``调性:``）以及行内写法（``120 BPM``、``F# minor``）。
    未识别到的字段不会出现在返回值里。

    :returns: 形如 ``{"bpm": 120, "tempo": "Moderato", "key": "C major"}``
    """
    if not text:
        return {}

    bpm: Optional[int] = None
    tempo: Optional[str] = None
    key: Optional[str] = None

    for raw_line in text.splitlines():
        line = _clean_meta_line(raw_line)
        if not line:
            continue

        # --- 行首 BPM 标签 ---
        if bpm is None:
            match = _BPM_LABEL_RE.match(line)
            if match:
                bpm = _valid_bpm(int(match.group(1)))
                if bpm is not None:
                    continue

        # --- 行内 x BPM（可能出现在同一行的其它描述里）---
        if bpm is None:
            match = _BPM_INLINE_RE.search(line)
            if match:
                bpm = _valid_bpm(int(match.group(1)))

        # --- Tempo 标签（值是纯数字时归入 BPM）---
        if tempo is None:
            match = _TEMPO_LABEL_RE.match(line)
            if match:
                value = _clean_meta_value(match.group(1))
                if value:
                    bpm_only = _BPM_VALUE_RE.match(value)
                    if bpm_only:
                        if bpm is None:
                            bpm = _valid_bpm(int(bpm_only.group(1)))
                    else:
                        tempo = value[:_MAX_META_LEN]
                continue

        # --- Key 标签 ---
        if key is None:
            match = _KEY_LABEL_RE.match(line)
            if match:
                value = _clean_meta_value(match.group(1))
                if value and _looks_like_key(value):
                    key = value[:_MAX_META_LEN]
                continue

    # 兜底：整段文本里找
    if key is None:
        match = _KEY_INLINE_RE.search(text)
        if match:
            key = (match.group(1) or match.group(2))[:_MAX_META_LEN]
    if bpm is None:
        match = _BPM_INLINE_RE.search(text)
        if match:
            bpm = _valid_bpm(int(match.group(1)))

    result: dict[str, Any] = {}
    if bpm is not None:
        result["bpm"] = bpm
    if tempo:
        result["tempo"] = tempo
    if key:
        result["key"] = key
    return result


# ----------------------------------------------------------------------
# 氛围（Mood）
# ----------------------------------------------------------------------
#: ``Mood: 温暖, 舒缓`` / ``氛围：空灵``
_MOOD_LABEL_RE = re.compile(
    r"^\s*(?:mood|atmosphere|vibe|feeling|氛围|情绪|气氛|情感)\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def extract_mood(text: str) -> Optional[str]:
    """提取氛围/情绪（如 ``温暖, 舒缓``）。

    只认带标签的写法（``Mood:`` / ``氛围：``），避免把正文里随便一句
    情绪描述误当标签。取不到返回 ``None``。
    """
    if not text:
        return None
    for raw_line in text.splitlines():
        line = _clean_meta_line(raw_line)
        if not line:
            continue
        match = _MOOD_LABEL_RE.match(line)
        if not match:
            continue
        value = _clean_meta_value(match.group(1))
        value = value.strip("[]【】").strip()
        if value and len(value) <= _MAX_META_LEN:
            return value
    return None


# ----------------------------------------------------------------------
# 评分（Score）与歌曲评价（Review）
# ----------------------------------------------------------------------
#: ``Score: 8`` / ``评分：8/10`` / ``rating: 4/5``
_SCORE_LABEL_RE = re.compile(
    r"^\s*(?:score|rating|评分|打分|得分)\s*[:：]\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?",
    re.IGNORECASE,
)
#: ``Review: 一句话评价`` / ``评价：…``
_REVIEW_LABEL_RE = re.compile(
    r"^\s*(?:review|comment|overall|评价|简评|点评|乐评|总评)\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
#: 评分统一到 0-10
_SCORE_MAX = 10
#: 评价最大长度（写进标签不宜过长）
_MAX_REVIEW_LEN = 200


def extract_score(text: str) -> Optional[int]:
    """提取评分并归一化到 0-10。

    支持 ``Score: 8``、``评分：8/10``、``rating: 4/5``（按分母换算到 10 分制）。
    """
    if not text:
        return None
    for raw_line in text.splitlines():
        line = _clean_meta_line(raw_line)
        if not line:
            continue
        match = _SCORE_LABEL_RE.match(line)
        if not match:
            continue
        try:
            value = int(match.group(1))
            scale = int(match.group(2)) if match.group(2) else _SCORE_MAX
        except (TypeError, ValueError):
            continue
        if scale <= 0 or value < 0 or value > scale:
            continue
        # 按分母换算到 10 分制
        normalized = round(value * _SCORE_MAX / scale)
        return max(0, min(_SCORE_MAX, normalized))
    return None


def extract_review(text: str) -> Optional[str]:
    """提取一句话歌曲评价（``Review:`` / ``评价：``）。"""
    if not text:
        return None
    for raw_line in text.splitlines():
        line = _clean_meta_line(raw_line)
        if not line:
            continue
        match = _REVIEW_LABEL_RE.match(line)
        if not match:
            continue
        value = match.group(1).strip().strip("\"'“”‘’").strip()
        if value:
            return value[:_MAX_REVIEW_LEN]
    return None


# ----------------------------------------------------------------------
# 风格判定理由（"为什么是这个风格"）
# ----------------------------------------------------------------------
#: 理由字段的标签（中英文）
_RATIONALE_LABEL_RE = re.compile(
    r"^\s*(?:why|reasons?|rationale|justification|"
    r"genre\s+rationale|style\s+rationale|"
    r"为什么是这个风格|为什么是这种风格|风格理由|判定理由|理由|依据)\s*[:：]\s*(.*)$",
    re.IGNORECASE,
)
#: 其它已知字段：遇到就说明理由块结束了
_NEXT_FIELD_RE = re.compile(
    r"^\s*(?:bpm|tempo|key|tonality|beats?\s*per\s*minute|速度|节奏|调性|调号)\s*[:：]",
    re.IGNORECASE,
)
#: 启发式：含因果词、且像风格判断的句子
_CAUSAL_RE = re.compile(
    r"(because|since|due\s+to|which\s+is\s+why|as\s+it\s+(?:has|features|combines))"
    r"|(因为|由于|之所以|原因是)",
    re.IGNORECASE,
)
_GENRE_HINT_RE = re.compile(
    r"(style|genre|风格|流派|节奏|旋律|配器|和声|嗓音|人声|"
    r"vocal|beat|groove|harmon|synth|chorus|drum|guitar|piano|bass|melod|rhythm|instrument)",
    re.IGNORECASE,
)
_MAX_RATIONALE_LEN = 600


def extract_genre_rationale(text: str) -> Optional[str]:
    """提取「为什么是这个风格」的理由文本。

    优先取带标签的段落（``Why:`` / ``Reason:`` / ``理由：`` / ``依据：``…），
    支持跨多行，直到空行或下一个已知字段（BPM/Tempo/Key）为止。
    没有标签时退化为启发式：找第一句同时含因果词（because/因为…）与
    音乐要素词（style/vocal/节奏…）的句子。
    """
    if not text:
        return None

    lines = text.splitlines()
    collected: list[str] = []
    for index, raw in enumerate(lines):
        line = _clean_meta_line(raw)
        match = _RATIONALE_LABEL_RE.match(line)
        if not match:
            continue
        # 理由是一整句，不能像 key/tempo 那样剥掉句末标点
        first = match.group(1).strip().strip("\"'“”‘’").strip()
        if first:
            collected.append(first)
        # 继续收集后续行，直到空行/下一个字段/下一个标签
        for follow in lines[index + 1:]:
            candidate = _clean_meta_line(follow)
            if not candidate:
                break
            if _NEXT_FIELD_RE.match(candidate) or _RATIONALE_LABEL_RE.match(candidate):
                break
            collected.append(candidate)
        break

    if collected:
        rationale = " ".join(part for part in collected if part).strip()
        if rationale:
            return rationale[:_MAX_RATIONALE_LEN]

    # 启发式兜底：按句切分，找同时含因果词与音乐要素的那句
    for sentence in re.split(r"(?<=[.!?。！？])\s*|\n+", text):
        candidate = sentence.strip()
        if not candidate or len(candidate) > _MAX_RATIONALE_LEN:
            continue
        if _CAUSAL_RE.search(candidate) and _GENRE_HINT_RE.search(candidate):
            return candidate[:_MAX_RATIONALE_LEN]
    return None


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
