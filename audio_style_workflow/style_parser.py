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
    "Describe this track by covering the overall atmosphere, instrumentation, "
    "structural development, tonal character and emotional expression. "
    "Keep the whole answer under 400 words.\n"
    "Use exactly the section headers below (keep the ** markers), one section per line:\n"
    "**Genre**: <1-3 English genre tags, comma-separated, most specific first; "
    "only styles you can justify from the audio itself, no guessing the artist's scene>\n"
    "**Why**: <one or two sentences on why these genres fit>\n"
    "**Tempo**: <integer BPM + slow/medium/fast, e.g. \"125 BPM / medium\">\n"
    "**Harmony & Key**: <key signature, e.g. \"G major\"; add the chord movement if useful>\n"
    "**Mood & Theme**: <2-3 short mood words>\n"
    "**Score**: <integer 0-10, 10 being best>\n"
    "**Review**: <one or two sentences of overall review, in Chinese>\n"
    "**Instrument**: <main instruments and timbres>\n"
    "**Vocals & Lyrics**: <vocal characteristics; state clearly if instrumental>\n"
    "**Song Structure & Dynamics**: <3-5 sentences summarizing the main sections and how "
    "the arrangement develops; do NOT list timestamped segments one by one>"
)

DEFAULT_ANALYSIS_PROMPT_ZH = (
    "请从整体氛围、乐器音色、结构发展、调性色彩和情绪表达几个方面描述这段音乐，"
    "总长度控制在 400 字以内。\n"
    "请严格使用下面的小标题（保留 ** 号），每个小标题占一行：\n"
    "**Genre**: <英文流派标签，逗号分隔，1-3 个，最具体的放最前；"
    "只写能从音频本身听出来的风格，不要根据艺人产地或产业背景推测>\n"
    "**Why**: <一到两句说明为什么判定为这些风格>\n"
    "**Tempo**: <BPM 整数 + 慢/中/快，例如 \"125 BPM / 中速\">\n"
    "**Harmony & Key**: <调性，例如 \"G major\"；如有必要可附带和声走向>\n"
    "**Mood & Theme**: <2-3 个简短的中文情绪词>\n"
    "**Score**: <0-10 整数评分>\n"
    "**Review**: <用中文写一到两句总体评价>\n"
    "**Instrument**: <主要乐器与音色>\n"
    "**Vocals & Lyrics**: <人声特征；纯器乐请注明>\n"
    "**Song Structure & Dynamics**: <用 3-5 句话概括主要段落与动态发展；"
    "不要逐段罗列时间戳>"
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

#: 常见别名 -> 规范写法（子流派尤其容易写成多种形式）
_GENRE_ALIASES = {
    "poprock": "Pop-Rock", "pop rock": "Pop-Rock", "pop-rock": "Pop-Rock",
    "electropop": "Electropop", "electro pop": "Electropop", "electro-pop": "Electropop",
    "dancepop": "Dance-Pop", "dance pop": "Dance-Pop", "dance-pop": "Dance-Pop",
    "synthpop": "Synth-Pop", "synth pop": "Synth-Pop", "synth-pop": "Synth-Pop",
    "indierock": "Indie Rock", "indie rock": "Indie Rock",
    "altrock": "Alt Rock", "alt rock": "Alt Rock", "alt-rock": "Alt Rock",
    "progrock": "Prog Rock", "prog rock": "Prog Rock",
    "idm": "IDM", "uk garage": "UK Garage", "uk drill": "UK Drill",
    "kpop": "K-Pop", "k-pop": "K-Pop", "k pop": "K-Pop", "korean pop": "K-Pop",
    "korean idol": "K-Pop", "케이팝": "K-Pop", "韩流": "K-Pop", "韩国流行": "K-Pop",
    "citypop": "City Pop", "city pop": "City Pop", "シティポップ": "City Pop",
    "城市流行": "City Pop", "都市流行": "City Pop",
    "jpop": "J-Pop", "j-pop": "J-Pop", "j pop": "J-Pop", "japanese pop": "J-Pop",
    "日本流行": "J-Pop",
    "cpop": "C-Pop", "c-pop": "C-Pop", "mandopop": "Mandopop", "国语流行": "Mandopop",
    "cantopop": "Cantopop", "粤语流行": "Cantopop",
    "rnb": "R&B", "r&b": "R&B", "rhythm and blues": "R&B", "节奏布鲁斯": "R&B",
    "hip hop": "Hip-Hop", "hiphop": "Hip-Hop", "hip-hop": "Hip-Hop", "嘻哈": "Hip-Hop",
    "edm": "EDM", "electronic dance music": "EDM",
    "lofi": "Lo-Fi", "lo fi": "Lo-Fi", "lo-fi": "Lo-Fi",
    "drum and bass": "Drum & Bass", "dnb": "Drum & Bass", "d&b": "Drum & Bass",
    "synthwave": "Synthwave", "synth pop": "Synth-Pop", "synthpop": "Synth-Pop",
    "indie pop": "Indie Pop", "indiepop": "Indie Pop",
    "dream pop": "Dream Pop", "dreampop": "Dream Pop",
    "neo soul": "Neo-Soul", "neosoul": "Neo-Soul",
    "trip hop": "Trip-Hop", "triphop": "Trip-Hop",
}


#: 拆分连写流派用的词素（progressivehouse -> progressive + house）
_GENRE_WORDS = {
    "progressive", "house", "trance", "techno", "ambient", "pop", "rock", "synth",
    "dance", "indie", "melodic", "vocal", "electro", "tech", "deep", "drum",
    "bass", "jazz", "metal", "soul", "funk", "disco", "city", "dream", "folk",
    "classical", "acoustic", "alternative", "alt", "punk", "gospel", "world",
    "urban", "chill", "lounge", "smooth", "trip", "hop", "hip", "lo", "fi",
    "neo", "hard", "soft", "light", "dark", "wave", "core", "pop", "rnb",
}

#: 从描述文本里兜底识别流派时使用的词表（按长度倒序，优先匹配长词）
_GENRE_TERMS: tuple[str, ...] = (
    "progressive house", "progressive trance", "melodic house", "melodic trance",
    "vocal trance", "vocal house", "bass house", "electro house", "deep house",
    "tech house", "drum & bass", "drum and bass", "city pop", "dream pop",
    "synth-pop", "synth pop", "synthpop", "electropop", "dance-pop", "dance pop",
    "indie pop", "indie rock", "pop rock", "pop-rock", "alt rock", "art pop",
    "bedroom pop", "trip hop", "trip-hop", "lo-fi", "lofi", "hip-hop", "hip hop",
    "r&b", "rnb", "neo-soul", "new age", "film score", "soundtrack", "orchestral",
    "house", "trance", "techno", "ambient", "electronic", "edm", "dubstep",
    "disco", "funk", "soul", "jazz", "blues", "classical", "baroque", "romantic",
    "ballad", "acoustic", "folk", "country", "metal", "punk", "rock", "pop",
    "cinematic", "instrumental", "chillout", "lounge",
    "流行", "摇滚", "爵士", "古典", "民谣", "电子", "说唱", "嘻哈", "灵魂",
    "放克", "迪斯科", "氛围", "器乐", "新世纪", "原声",
)


def _split_concatenated(word: str) -> Optional[str]:
    """把 ``progressivehouse`` 这类连写拆成 ``Progressive House``。"""
    low = word.lower()
    parts: list[str] = []
    index = 0
    while index < len(low):
        for end in range(len(low), index, -1):
            if low[index:end] in _GENRE_WORDS:
                parts.append(low[index:end])
                index = end
                break
        else:
            return None
    if len(parts) >= 2:
        return " ".join(part.capitalize() for part in parts)
    return None


def _scan_genre_terms(text: str) -> list[str]:
    """从自然语言描述里找出提到的流派（兜底用）。"""
    lowered = (text or "").lower()
    found: list[str] = []
    for term in _GENRE_TERMS:
        if term not in lowered:
            continue
        canon = normalize_genre(term)
        if canon and canon not in found:
            found.append(canon)
        if len(found) >= 3:
            break
    return found


def normalize_genre(name: str) -> str:
    """把别名/大小写变体规范成通行写法。

    先查别名表（``kpop`` -> ``K-Pop``、``poprock`` -> ``Pop-Rock``），
    命中不了就统一成 Title Case（``progressive house`` -> ``Progressive House``、
    ``rock`` -> ``Rock``），避免同一个流派出现大小写混杂。
    """
    text = (name or "").strip()
    if not text:
        return text
    alias = _GENRE_ALIASES.get(text.lower())
    if alias:
        return alias
    # 全部大写的缩写（EDM/IDM）保持原样
    if text.isupper() and len(text) <= 5:
        return text
    # 连写形式（progressivehouse / ambientpop）拆成单词
    if " " not in text and "-" not in text and text.isalpha():
        split = _split_concatenated(text)
        if split:
            return split
    return text.title()


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


#: 摘要区里的标签行（可出现在任意行，而不只是第一行）
_TAGS_LABEL_LINE_RE = re.compile(
    r"^\s*(?:tags?|genres?|styles?|风格标签|风格|标签|流派|曲风)\s*[:：]\s*(.+)$",
    re.IGNORECASE,
)


def _looks_like_tag_line(line: str) -> bool:
    """判断一行是否像「标签列表」而不是散文句子。

    中文散文按逗号切出来的片段往往很短，容易骗过长度检查（例如
    ``这是一首典型的渐进式Trance作品，融合了…``），因此这里额外要求：
    没有句末标点、片段不过长、片段数在范围内。
    """
    if not line or len(line) > 140:
        return False
    if re.search(r"[。！？!?]", line):
        return False
    parts = [part.strip() for part in _SPLIT_RE.split(line) if part.strip()]
    if not (1 <= len(parts) <= _MAX_LABELS):
        return False
    return all(len(part) <= 24 for part in parts)


def _parse_labels(segment: str, *, limit: int) -> list[str]:
    """把 ``Pop, R&B, Jazz`` 这类片段切成规范化标签列表。"""
    labels: list[str] = []
    for raw in _SPLIT_RE.split(segment):
        label = _TRAILING_PUNCT_RE.sub("", raw.strip().strip('"“”（）()<>').strip())
        if not label:
            continue
        if _INSTRUCTIONY_RE.search(label):
            continue
        if len(label) > _MAX_LABEL_LEN or len(label.split()) > _MAX_LABEL_WORDS:
            return []
        label = normalize_genre(label)
        if label not in labels:
            labels.append(label)
    if not (1 <= len(labels) <= _MAX_LABELS):
        return []
    return labels[:limit]


def extract_styles_from_text(text: str, *, limit: int = _MAX_LABELS) -> list[str]:
    """从生成式模型输出的第一行提取风格标签（确定性解析，无需 LLM）。

    规则：

    1. 取第一条非空行；剥掉 "Styles:" / "Genres:" / "风格:" 等前缀，
       以及回声的指令前缀（``...list:``）；
    2. 按 `, ， ; ； 、 / |` 及连续空格拆分；
    3. 每个标签长度须 <= 48 字符、词数 <= 4（排除被误拆的完整句子）；
    4. 标签数为 1~8 才可信（过长通常是普通描述句）。
    """
    # 1) 优先取「描述后摘要」里的标签行（TAGS: / 风格标签: / Genres: …）
    for raw_line in text.splitlines():
        match = _TAGS_LABEL_LINE_RE.match(_clean_meta_line(raw_line))
        if not match:
            continue
        labels = _parse_labels(match.group(1), limit=limit)
        if labels:
            return labels

    # 2) 回退：取第一行（兼容"第一行就是标签"的模型/端点）
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return []
    first_line = _PREFIX_RE.sub("", first_line)
    first_line = _strip_instruction_echo(first_line)
    if _looks_like_tag_line(first_line):
        labels = _parse_labels(first_line, limit=limit)
        if labels:
            return labels

    # 3) 最后一招：模型只给了描述、没给摘要行 -> 从描述里扫流派词
    scanned = _scan_genre_terms(text)
    if scanned:
        logger.warning(
            "[styles] 模型未输出 TAGS 摘要行，已从描述中兜底提取: %s", scanned
        )
    return scanned[:limit]


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
    r"^\s*(?:tempo|speed|速度|节奏|节拍速度|速度与节奏)\s*[:：=\-—]\s*(.+?)\s*$",
    re.IGNORECASE,
)
#: 复合值里的速度词（"125 BPM / Moderato" -> Moderato）
_TEMPO_WORD_RE = re.compile(
    r"(largo|adagio|andante|moderato|allegro|vivace|presto|"
    r"slow|medium|moderate|mid-?tempo|fast|"
    r"慢板|慢速|中速|快速|快板|行板|小快板|急板|慢|快)",
    re.IGNORECASE,
)
#: ``Key: C major`` / ``调性：A小调``
_KEY_LABEL_RE = re.compile(
    r"^\s*(?:harmony\s*(?:&|and|/|与)\s*key|key|tonality|调性|调号|"
    r"和声\s*(?:与|和|/)\s*调性|调性色彩|调)\s*[:：=\-—]\s*(.+?)\s*$",
    re.IGNORECASE,
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
                        # "125 BPM / Moderato" 这类复合值：速度词单独取
                        word = _TEMPO_WORD_RE.search(value)
                        tempo = (word.group(1) if word else value)[:_MAX_META_LEN]
                        if bpm is None:
                            inline = _BPM_INLINE_RE.search(value)
                            if inline:
                                bpm = _valid_bpm(int(inline.group(1)))
                continue

        # --- Key 标签 ---
        if key is None:
            match = _KEY_LABEL_RE.match(line)
            if match:
                value = _clean_meta_value(match.group(1))
                if value:
                    # 值里常带和声说明（"G major，I–V–vi–IV 循环"）——优先抽出调性本身
                    inline = _KEY_INLINE_RE.search(value)
                    if inline:
                        key = (inline.group(1) or inline.group(2))[:_MAX_META_LEN]
                    elif _looks_like_key(value):
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
    r"^\s*(?:mood\s*(?:&|and|/|与)\s*theme|mood|atmosphere|vibe|feeling|"
    r"情绪\s*(?:与|和|/)\s*主题|氛围|情绪|气氛|情感|主题)\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def tempo_label_from_bpm(bpm: int) -> str:
    """按传统速度术语把 BPM 映射成标签（模型给的 Tempo 常与 BPM 打架）。"""
    try:
        value = int(bpm)
    except (TypeError, ValueError):
        return ""
    if value < 60:
        return "Largo"
    if value < 76:
        return "Adagio"
    if value < 108:
        return "Andante"
    if value < 120:
        return "Moderato"
    if value < 156:
        return "Allegro"
    if value < 176:
        return "Vivace"
    return "Presto"


#: 情绪值里常见的"解释性尾巴"，从这些词开始截断
_MOOD_TAIL_MARKERS = (
    "，带有", "，营造", "，整体", "，并", "，同时", "，适合",
    ", with", ", creating", ", and it", ", giving", ", suitable",
)
#: 中文情绪词一般很短，超过这个长度多半是描述句
_MAX_MOOD_PART_LEN = 6
#: 英文情绪词上限
_MAX_MOOD_PART_LEN_EN = 16


def _normalize_mood(value: str, limit: int = 3) -> str:
    """把 ``明亮、现代、充满能量的电子流行氛围`` 这类长句裁成短情绪词。

    模型在 ``Mood & Theme`` 里常写一整句解释，直接当标签会污染 ``TMOOD``。
    这里按分隔符切分，丢弃过长的"描述性"片段。
    """
    text = (value or "").strip().strip("。.；;").strip()
    if not text:
        return ""
    for marker in _MOOD_TAIL_MARKERS:
        index = text.find(marker)
        if index > 0:
            text = text[:index]
            break

    parts = [part.strip() for part in re.split(r"[、,，/;；|]", text) if part.strip()]
    kept: list[str] = []
    for part in parts:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", part))
        cap = _MAX_MOOD_PART_LEN if has_cjk else _MAX_MOOD_PART_LEN_EN
        if len(part) <= cap:
            kept.append(part)
    if not kept:                      # 全是长句时退回前两个片段（整段保留，别从词中间切）
        kept = parts[:2]
    joined = ", ".join(kept[:limit])
    if len(joined) > 24:
        joined = joined[:24].rstrip("，, ") + "…"
    return joined


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
        if not value:
            continue
        normalized = _normalize_mood(value)
        if normalized:
            return normalized[:_MAX_META_LEN]
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

        # 客户端只返回正文（无 styles 字段）时，先做确定性文本解析
        text = _payload_text(payload)
        if text:
            text_styles = extract_styles_from_text(text)
            if text_styles:
                return text_styles[: self.max_styles], "text", _extract_confidence(payload)

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


#: 可能承载模型原始文本的字段（客户端只回文本时用）
_TEXT_FIELDS = ("description", "text", "content", "raw_response", "output", "result")


def _payload_text(payload: Mapping[str, Any]) -> str:
    """取出响应里的自然语言正文（用于确定性兜底解析）。"""
    for field in _TEXT_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _preview(payload: Any, limit: int = 240) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
