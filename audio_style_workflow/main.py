"""命令行入口。

示例::

    python -m audio_style_workflow.main D:\\Music \\
        --endpoint https://analyzer.example.com/api/analyze \\
        --custom-tag GENRE --skip-tagged
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，避免打印异常
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .api.base import SongAnalysisClient, SongAnalysisError, create_client
from .api.sglang_client import DEFAULT_SGLANG_BASE
from .logging_setup import DEFAULT_LOG_DIR, setup_logging
from .models import PipelineState, TrackTask
from .splitter import SplitError, TtaSplitter
from .style_parser import StyleParser, get_analysis_prompt
from .tagger import DEFAULT_CUSTOM_KEY, DEFAULT_JOINER
from .workflow import RunReport, WorkflowConfig, run_workflow

logger = logging.getLogger("audio_style_workflow")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-style-workflow",
        description="扫描目录中的歌曲->(TTA 按 CUE 切分)->提交分析端点->解析风格->写回自定义标签",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_argument_group("目标")
    target.add_argument("root", type=Path, help="音乐库根目录（递归扫描子目录）")

    client = parser.add_argument_group("分析端点")
    client.add_argument(
        "-e", "--endpoint",
        default=os.environ.get("SONG_ANALYZER_ENDPOINT"),
        help="分析端点 URL（也可用环境变量 SONG_ANALYZER_ENDPOINT）",
    )
    client.add_argument("--api-key", default=os.environ.get("SONG_ANALYZER_API_KEY"), help="Bearer 鉴权密钥")
    client.add_argument("--file-field", default=os.environ.get("SONG_ANALYZER_FILE_FIELD", "file"), help="上传文件的表单字段名")
    client.add_argument("--timeout", type=float, default=300.0, help="单次请求超时秒数")
    client.add_argument("--retries", type=int, default=2, help="连接级重试次数")
    client.add_argument("--mock", action="store_true", help="使用进程内 Mock 端点（无需网络，用于联调）")
    client.add_argument("--mock-freeform", action="store_true", help="Mock 返回非结构化内容（配合 --llm-model 验证 LLM 兑底）")
    client.add_argument("--local", action="store_true", help="使用本地 MusicFlamingo 模型分析（离线推理，需 pip install -r requirements-local.txt）")
    client.add_argument("--sglang", nargs="?", const=os.environ.get("SGLANG_BASE", DEFAULT_SGLANG_BASE), metavar="BASE_URL", help="使用 sglang 部署的音乐理解模型（音频传本地路径，需与服务共享文件系统；不带值时读环境变量 SGLANG_BASE，再回退 http://localhost:30000）")
    client.add_argument("--sglang-base", default=None, metavar="BASE_URL", help="显式指定 sglang 服务基地址（等价于 --sglang BASE_URL；也可用环境变量 SGLANG_BASE）")
    client.add_argument("--sglang-thinking-budget", type=int, default=None, metavar="N", help="限制思考 token 数（如 50）；需同时提供 --sglang-custom-logit-processor")
    client.add_argument("--sglang-custom-logit-processor", default=os.environ.get("SGLANG_CUSTOM_LOGIT_PROCESSOR"), metavar="JSON_OR_@FILE", help="thinking budget 所需的 custom_logit_processor 序列化串（Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str() 的输出）；也可写 @文件路径，或设环境变量 SGLANG_CUSTOM_LOGIT_PROCESSOR")
    client.add_argument("--sglang-path-map", action="append", default=None, metavar="LOCAL=REMOTE",
                        help="把本地路径前缀映射为服务端可见路径，可重复。例：--sglang-path-map \"C:\\music=/data/music\"（服务端读不到客户端路径时必须用）")
    client.add_argument("--sglang-clip", type=int, default=0, metavar="SECONDS",
                        help="只取这么长的音频送模型（0=整首）。仅用于降低请求体积；实测模型本就会完整摄入整首，截取会丢信息")
    client.add_argument("--sglang-clip-mode", choices=["middle", "start"], default="middle",
                        help="--sglang-clip 的截取位置：middle=中间窗口（默认）；start=从头")
    client.add_argument("--sglang-audio-base64", action="store_true",
                        help="把音频以 data URL（base64）内联进请求，无需共享文件系统（体积增大约 33%%）")
    client.add_argument("--prompt-lang", choices=["zh", "en"], default="zh",
                        help="分析 prompt 的语言，从而决定模型用什么语言输出（默认 zh）")
    client.add_argument("--model-id", default=os.environ.get("SONG_ANALYZER_MODEL_ID", "nvidia/music-flamingo-2601-hf"), help="本地 MusicFlamingo 模型 ID（配合 --local）")
    client.add_argument("--max-new-tokens", type=int, default=500, help="模型生成 token 上限（--local / --sglang 共用；sglang 建议 1024）")
    client.add_argument("--concurrency", type=int, default=1, help="分析端点并发请求数（本地模型会自动串行化）")

    tags = parser.add_argument_group("标签")
    tags.add_argument("--custom-tag", default=DEFAULT_CUSTOM_KEY, help="自定义风格标签键名（FLAC: VorbisComment 字段 / MP3: TXXX 帧）")
    tags.add_argument("--joiner", default=DEFAULT_JOINER, help="多风格连接符")
    tags.add_argument("--skip-tagged", action="store_true", help="已含该自定义标签的歌曲跳过分析（依据文件内标签）")
    tags.add_argument("--no-musical-tags", dest="musical_tags", action="store_false",
                      help="不写入 BPM/Tempo/Key 标签（默认写入：FLAC BPM/TEMPO/KEY；MP3 TBPM/TKEY/TXXX:TEMPO）")
    tags.add_argument("--dry-run", action="store_true", help="只分析并展示将写入的标签，不落盘")

    state = parser.add_argument_group("扫描记录（skip-scanned 断点续跑）")
    state.add_argument("--skip-scanned", action="store_true", help="跳过扫描记录中已成功处理且未变化的歌曲（TTA 专辑整张跳过切分；文件变动后自动重处理）")
    state.add_argument("--state-file", type=Path, default=os.environ.get("SONG_ANALYZER_STATE_FILE"), help="扫描记录文件路径（默认 <root>/.audio_style_scan.json）")
    state.add_argument("--reset-state", action="store_true", help="运行前清空扫描记录（重新全量处理）")
    state.add_argument("--per-song", dest="per_song", action="store_true",
                       help="逐曲闭环：每首分析完立即解析并写标签、记扫描记录（默认先全部分析再统一写）")
    state.add_argument("--state-flush-every", type=int, default=10, metavar="N",
                       help="扫描记录每累计 N 条就落盘一次（0 = 只在结束时保存）；中断后可用 --skip-scanned 续跑")

    split = parser.add_argument_group("TTA 切分")
    split.add_argument("--split-dir", type=Path, default=None, help="分轨输出目录（默认与 TTA 整轨同目录）")
    split.add_argument("--ffmpeg", default=None, help="ffmpeg 可执行文件（默认从 PATH / FFMPEG_PATH 查找）")
    split.add_argument("--resplit", action="store_true", help="忽略已存在的分轨文件，强制重新切分")

    llm = parser.add_argument_group("LLM 兜底解析")
    llm.add_argument("--llm-model", default=os.environ.get("SONG_ANALYZER_LLM_MODEL"), help="启用 LLM 解析非结构化响应（如 gpt-4o-mini，需 OPENAI_API_KEY）")

    misc = parser.add_argument_group("日志")
    misc.add_argument(
        "--log-dir", type=Path,
        default=Path(os.environ.get("SONG_ANALYZER_LOG_DIR", str(DEFAULT_LOG_DIR))),
        metavar="DIR",
        help="日志目录：<DIR>/latest.log 记录全量日志，<DIR>/details/ 存放逐曲分析明细"
             "（也可用环境变量 SONG_ANALYZER_LOG_DIR）",
    )
    misc.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def parse_path_maps(values: Optional[list[str]]) -> Optional[list[tuple[str, str]]]:
    """把 ``--sglang-path-map LOCAL=REMOTE`` 解析成 (本地前缀, 服务端前缀) 列表。"""
    if not values:
        return None
    maps: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--sglang-path-map 需要 LOCAL=REMOTE 形式，当前: {raw!r}")
        local, remote = raw.split("=", 1)
        local, remote = local.strip(), remote.strip()
        if not local or not remote:
            raise ValueError(f"--sglang-path-map 两侧都不能为空: {raw!r}")
        maps.append((local, remote))
    return maps


def build_client(args: argparse.Namespace):
    sglang_endpoint = args.sglang_base or args.sglang
    modes = [bool(args.local), bool(args.mock), sglang_endpoint is not None]
    if sum(modes) > 1:
        raise ValueError("--local / --mock / --sglang 互斥，请只选一种分析端点")
    prompt = get_analysis_prompt(getattr(args, "prompt_lang", "en"))
    if sglang_endpoint is not None:
        return create_client(
            "sglang",
            endpoint=sglang_endpoint,
            api_mode="generate",          # 固定走 /generate（chat 端点会丢音频）
            prompt=prompt,
            thinking_budget=args.sglang_thinking_budget,
            max_new_tokens=args.max_new_tokens,
            custom_logit_processor=args.sglang_custom_logit_processor,
            audio_path_map=parse_path_maps(args.sglang_path_map),
            audio_base64=getattr(args, "sglang_audio_base64", False),
            clip_seconds=getattr(args, "sglang_clip", 0),
            clip_mode=getattr(args, "sglang_clip_mode", "middle"),
            ffmpeg=args.ffmpeg,
        )
    if args.local:
        logger.info("已选择本地模型分析（忽略 --endpoint）")
        return create_client(
            "local",
            prompt=prompt,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
        )
    if args.mock:
        mode = "freeform" if args.mock_freeform else "json"
        return create_client("mock", mode=mode)
    if not args.endpoint:
        raise ValueError(
            "未提供 --endpoint（或环境变量 SONG_ANALYZER_ENDPOINT）；"
            "联调可加 --mock，离线推理可加 --local，"
            "sglang 部署的音乐模型可加 --sglang [BASE_URL]（或 --sglang-base BASE_URL）"
        )
    return create_client(
        "http",
        endpoint=args.endpoint,
        api_key=args.api_key,
        file_field=args.file_field,
        timeout=args.timeout,
        retries=args.retries,
    )


def build_style_parser(args: argparse.Namespace) -> StyleParser:
    if not args.llm_model:
        return StyleParser()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("未安装 langchain-openai，LLM 兜底解析不可用（pip install langchain-openai）")
        return StyleParser()
    return StyleParser(ChatOpenAI(model=args.llm_model, temperature=0))


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    log_paths = setup_logging(args.log_dir, verbose=args.verbose)

    if not args.root.is_dir():
        print(f"[错误] 目录不存在或不是目录: {args.root}", file=sys.stderr)
        return 2

    try:
        client = build_client(args)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    try:
        splitter: Optional[TtaSplitter] = TtaSplitter(args.ffmpeg)
    except SplitError as exc:
        # ffmpeg 缺失时仍可处理 mp3/flac，TTA 会记入警告
        logger.warning("%s", exc)
        splitter = None

    config = WorkflowConfig(
        root=args.root,
        split_dir=args.split_dir,
        custom_tag_key=args.custom_tag,
        style_joiner=args.joiner,
        skip_tagged=args.skip_tagged,
        skip_scanned=args.skip_scanned,
        state_file=args.state_file,
        reset_state=args.reset_state,
        dry_run=args.dry_run,
        concurrency=max(1, args.concurrency),
        reuse_split_tracks=not args.resplit,
        log_dir=log_paths.root,
        write_musical_tags=getattr(args, "musical_tags", True),
        per_song=getattr(args, "per_song", False),
        state_flush_every=max(0, getattr(args, "state_flush_every", 10)),
    )

    print(f"目标目录 : {config.root}")
    print(f"分析端点 : {client.name}（{getattr(client, 'endpoint', 'in-process mock')}）")
    print(f"自定义标签: TXXX:{config.custom_tag_key} / {config.custom_tag_key}"
          + ("（dry-run，不写入）" if config.dry_run else ""))
    print(f"日志      : {log_paths.latest}（明细 {log_paths.details}，评价 {log_paths.descriptions}）")
    if config.skip_scanned:
        state_desc = "启用" + ("（重置记录）" if config.reset_state else "") \
            + ("；dry-run 下不读取/不写入记录" if config.dry_run else "")
        print(f"扫描记录  : {state_desc}")
    print()

    report: RunReport = run_workflow(
        client=client,
        style_parser=build_style_parser(args),
        config=config,
        splitter=splitter,
    )
    report.print_summary()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
