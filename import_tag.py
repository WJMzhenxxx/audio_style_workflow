#!/usr/bin/env python
"""把 JSON 里的分类标签导入音乐库。

按「相对路径 → MD5 → 文件名」匹配；遇到尚未切分的 TTA 整轨会先按 CUE 切分再写入。

用法::

    python import_tag.py -i ./tags.json -o /path/to/music
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audio_style_workflow.logging_setup import setup_logging
from audio_style_workflow.tag_io import import_tags, load_export


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import_tag.py",
        description="把导出的标签 JSON 写回音乐文件（TTA 整轨未切分会先切分）",
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE", help="标签 JSON")
    parser.add_argument("-o", "--output", required=True, metavar="DIR", help="目标音乐目录")
    parser.add_argument("--custom-tag", default=None, help="覆盖 JSON 里的标签键名")
    parser.add_argument("--joiner", default=None, help="覆盖 JSON 里的多风格连接符")
    parser.add_argument("--no-split", action="store_true", help="不自动切分 TTA 整轨，只导入已有分轨")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg 可执行文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只显示匹配结果，不写文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(None, verbose=args.verbose)

    try:
        data = load_export(Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"[错误] 读取标签文件失败: {exc}", file=sys.stderr)
        return 1

    try:
        report = import_tags(
            data,
            Path(args.output),
            dry_run=args.dry_run,
            ffmpeg=args.ffmpeg,
            custom_key=args.custom_tag,
            joiner=args.joiner,
            split_missing_tta=not args.no_split,
        )
    except (OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    print("\n================ 导入摘要 ================")
    print(f"应用条目      : {report.applied}")
    print(f"写入文件      : {report.files}")
    print(f"新建切分整轨  : {report.split_albums}（{report.split_tracks} 轨）")
    print(f"未匹配        : {len(report.missing)}")
    print(f"失败          : {len(report.failed)}")
    if report.dry_run:
        print("模式          : dry-run（未写任何文件）")

    for label in report.missing[:20]:
        print(f"  [MISS] {label}")
    if len(report.missing) > 20:
        print(f"  ... 其余 {len(report.missing) - 20} 条缺失略")
    for item in report.failed[:20]:
        print(f"  [FAIL] {item}")

    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
