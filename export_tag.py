#!/usr/bin/env python
"""导出音乐库里的分类标签到 JSON。

用法::

    python export_tag.py -i /path/to/taggedmusic -o ./tags.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audio_style_workflow.logging_setup import setup_logging
from audio_style_workflow.tag_io import export_tags
from audio_style_workflow.tagger import DEFAULT_CUSTOM_KEY, DEFAULT_JOINER


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_tag.py",
        description="把音乐库中的分类标签（流派/氛围/BPM/调性/评分/评价）导出为 JSON",
    )
    parser.add_argument("-i", "--input", required=True, metavar="DIR", help="音乐目录（递归扫描）")
    parser.add_argument("-o", "--output", default="tags.json", metavar="FILE", help="输出 JSON，默认 tags.json")
    parser.add_argument("--custom-tag", default=DEFAULT_CUSTOM_KEY, help=f"标签键名，默认 {DEFAULT_CUSTOM_KEY}")
    parser.add_argument("--joiner", default=DEFAULT_JOINER, help="多风格连接符，默认 '; '")
    parser.add_argument("--include-untagged", action="store_true", help="连没有标签的文件也记录")
    parser.add_argument("--no-audio-hash", action="store_true",
                        help="不计算音频内容 md5（默认计算，标签被改写过也能匹配）")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg 可执行文件路径（算音频 md5 用）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(None, verbose=args.verbose)

    try:
        report = export_tags(
            Path(args.input),
            Path(args.output),
            custom_key=args.custom_tag,
            joiner=args.joiner,
            include_untagged=args.include_untagged,
            ffmpeg=args.ffmpeg,
            with_audio_hash=not args.no_audio_hash,
        )
    except (OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    print("\n================ 导出摘要 ================")
    print(f"输出文件      : {report.output}")
    print(f"记录歌曲      : {report.tracks}")
    print(f"其中有标签    : {report.with_tags}")
    print(f"跳过无标签    : {report.skipped_untagged}")
    print(f"涉及整轨专辑  : {report.albums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
