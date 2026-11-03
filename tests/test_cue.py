"""CUE 解析器单元测试。"""
import pytest

from audio_style_workflow.cue import (
    CueParseError,
    frames_to_seconds,
    parse_cue_file,
    parse_cue_text,
    parse_time,
)

SAMPLE = """\
REM GENRE Pop
REM DATE 2001
PERFORMER "周杰伦"
TITLE "范特西"
FILE "album.tta" WAVE
  TRACK 01 AUDIO
    TITLE "爱在西元前"
    PERFORMER "周杰伦"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "简单爱"
    INDEX 01 04:22:35
  TRACK 03 AUDIO
    TITLE "忍者"
    INDEX 00 08:01:74
    INDEX 01 08:01:74
  TRACK 04 AUDIO
    TITLE "开不了口"
    INDEX 01 009:00:00
  TRACK 05 AUDIO
    TITLE "上海一九四三"
    INDEX 01 510:00:00
"""


def test_frames_to_seconds():
    assert frames_to_seconds(0, 0, 0) == 0.0
    assert frames_to_seconds(1, 2, 3) == 62 + 3 / 75
    assert frames_to_seconds(4, 22, 35) == 262 + 35 / 75


def test_parse_time():
    assert parse_time(" 04:22:35 ") == 262 + 35 / 75
    with pytest.raises(CueParseError):
        parse_time("12:34")
    with pytest.raises(CueParseError):
        parse_time("ab:cd:ef")


def test_parse_basic_structure():
    sheet = parse_cue_text(SAMPLE)
    assert sheet.performer == "周杰伦"
    assert sheet.title == "范特西"
    assert sheet.file_name == "album.tta"
    assert [t.number for t in sheet.tracks] == [1, 2, 3, 4, 5]


def test_parse_track_starts():
    sheet = parse_cue_text(SAMPLE)
    starts = [t.start_sec for t in sheet.tracks]
    assert starts == pytest.approx([0.0, 262 + 35 / 75, 481 + 74 / 75, 540.0, 30600.0])


def test_index00_with_chooses_index01():
    sheet = parse_cue_text(SAMPLE)
    track3 = sheet.tracks[2]
    # 08:01:74 -> 8*60+1+74/75，INDEX 00 与 01 相同，取 01
    assert track3.start_sec == pytest.approx(481 + 74 / 75)


def test_index00_fallback_when_no_index01():
    text = """\
FILE "a.tta" WAVE
  TRACK 01 AUDIO
    TITLE "one"
    INDEX 00 00:00:00
  TRACK 02 AUDIO
    TITLE "two"
    INDEX 00 01:00:00
"""
    sheet = parse_cue_text(text)
    assert sheet.tracks[1].start_sec == pytest.approx(60.0)


def test_per_track_performer_overrides_album():
    sheet = parse_cue_text(SAMPLE)
    assert sheet.tracks[0].title == "爱在西元前"
    assert sheet.tracks[0].performer == "周杰伦"
    assert sheet.tracks[2].performer == ""  # 未定义时为空，切分层回退专辑艺人


def test_no_tracks_raises():
    with pytest.raises(CueParseError):
        parse_cue_text("REM GENRE Pop\nPERFORMER \"x\"\n")


def test_file_without_quotes():
    sheet = parse_cue_text('FILE album.tta WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n')
    assert sheet.file_name == "album.tta"


def test_gbk_encoded_cue(tmp_path):
    gbk_text = (
        'PERFORMER "周杰伦"\nTITLE "范特西"\nFILE "album.tta" WAVE\n'
        "  TRACK 01 AUDIO\n    TITLE \"爱在西元前\"\n    INDEX 01 00:00:00\n"
    )
    cue = tmp_path / "album.cue"
    cue.write_bytes(gbk_text.encode("gbk"))
    sheet = parse_cue_file(cue)
    assert sheet.performer == "周杰伦"
    assert sheet.tracks[0].title == "爱在西元前"


def test_utf8_with_bom(tmp_path):
    cue = tmp_path / "bom.cue"
    cue.write_bytes(('TITLE "t"\nFILE "a.tta" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n').encode("utf-8-sig"))
    sheet = parse_cue_file(cue)
    assert sheet.title == "t"
