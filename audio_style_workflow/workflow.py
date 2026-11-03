"""基于 LangChain(LCEL) 的歌曲风格工作流。

``RunnableLambda`` 把每个处理步骤封装为 Runnable，再以 ``|``（pipe）
串成一条可独立运行、可观测的 RunnableSequence::

    None ──▶ scan ──▶ split_tta ──▶ analyze ──▶ parse_styles ──▶ write_tags ──▶ list[TrackTask]

设计要点:

- 每步都是"输入状态 -> 输出状态"的纯函数（便于单测与替换）；
- 单文件失败只记录 ``error``，不中断整批任务；
- ``skip_tagged``、``dry_run``、``concurrency`` 等行为均由配置驱动。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda, RunnableSequence

from .api.base import SongAnalysisClient, SongAnalysisError
from .cue import CueParseError
from .models import PipelineState, TrackTask, TtaAlbum
from .scanner import scan_directory
from .splitter import SplitError, TtaSplitter
from .state_store import ScanStateStore
from .style_parser import StyleParser, StyleParseError
from .tagger import TaggerError, get_styles, write_styles

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 配置与报告
# ----------------------------------------------------------------------
@dataclass
class WorkflowConfig:
    """工作流行为配置。"""

    root: Path
    split_dir: Optional[Path] = None          # TTA 分轨输出目录（默认与整轨同目录）
    custom_tag_key: str = "STYLE"             # 自定义风格标签键名
    style_joiner: str = "; "                  # 多风格连接符
    skip_tagged: bool = False                 # 已有自定义标签的歌曲跳过分析
    skip_scanned: bool = False                # 扫描记录中已成功且未变化的歌曲跳过
    state_file: Optional[Path] = None         # 扫描记录文件（默认 <root>/.audio_style_scan.json）
    reset_state: bool = False                 # 运行前清空扫描记录
    dry_run: bool = False                     # 只分析解析，不写标签
    concurrency: int = 1                      # 分析端点并发数
    reuse_split_tracks: bool = True           # 分轨文件已存在时复用（断点续跑）


@dataclass
class RunReport:
    """一次工作流运行的结果。"""

    tasks: list[TrackTask] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    albums_skipped_scanned: int = 0           # 因扫描记录整张跳过切分的专辑数
    state_path: Optional[Path] = None         # 本次使用的扫描记录文件

    @property
    def analyzed(self) -> int:
        return sum(1 for t in self.tasks if t.analysis is not None)

    @property
    def parsed_by_json(self) -> int:
        return sum(1 for t in self.tasks if t.style_parser == "json")

    @property
    def parsed_by_llm(self) -> int:
        return sum(1 for t in self.tasks if t.style_parser == "llm")

    @property
    def tagged(self) -> int:
        return sum(1 for t in self.tasks if t.tagged)

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tasks if t.skip_reason and not t.error)

    @property
    def skipped_scanned(self) -> int:
        """因扫描记录（指纹一致）跳过的歌曲数。"""
        return sum(
            1 for t in self.tasks if (t.skip_reason or "").startswith(ScanStateStore.SKIP_PREFIX)
        )

    @property
    def failed(self) -> list[TrackTask]:
        return [t for t in self.tasks if t.error]

    def summary_lines(self) -> list[str]:
        lines = [
            "================ 处理摘要 ================",
            f"歌曲总数      : {len(self.tasks)}",
            f"已分析        : {self.analyzed}"
            f"（JSON 提取 {self.parsed_by_json} / LLM 兑底 {self.parsed_by_llm}）",
            f"已写回标签    : {self.tagged}",
            f"跳过          : {self.skipped}"
            + (f"（含扫描记录命中 {self.skipped_scanned}" 
               + (f"、整张跳过切分专辑 {self.albums_skipped_scanned} 张" if self.albums_skipped_scanned else "")
               + "）" if self.skipped_scanned or self.albums_skipped_scanned else ""),
            f"失败          : {len(self.failed)}",
            f"警告          : {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("")
            lines.append("---- 失败明细 ----")
            lines.extend(f"[FAIL] {t.path.name} — {t.error}" for t in self.failed)
        if self.warnings:
            lines.append("")
            lines.append("---- 警告明细 ----")
            lines.extend(f"[WARN] {w}" for w in self.warnings)
        return lines

    def print_summary(self) -> None:
        for line in self.summary_lines():
            print(line)


# ----------------------------------------------------------------------
# 工作流步骤（纯函数，均接收/返回 PipelineState 或其子集）
# ----------------------------------------------------------------------
def step_scan(state: Any, config: WorkflowConfig) -> PipelineState:
    """递归扫描目录：直连文件 -> tasks，TTA 整轨 -> albums。"""
    result = scan_directory(
        config.root,
        skip_tag_key=config.custom_tag_key if config.skip_tagged else None,
    )
    return result


def step_split(
    state: PipelineState,
    config: WorkflowConfig,
    splitter: Optional[TtaSplitter],
    store: Optional[ScanStateStore] = None,
) -> PipelineState:
    """把 (TTA, CUE) 专辑切分为 FLAC 分轨并追加进 tasks。

    启用 skip-scanned 时：专辑（TTA+CUE）指纹与记录一致则整张跳过切分；
    部分场景（复用旧分轨）下单曲指纹一致也会在分析阶段被跳过。
    """
    pending = list(state.tasks)
    warnings = list(state.warnings)
    albums_skipped = 0

    for album in state.albums:
        if album.error:
            message = f"[tta] {album.tta.name}: {album.error}"
            if message not in warnings:  # scan 层可能已上报，避免重复
                warnings.append(message)
            continue
        if splitter is None:
            warnings.append(
                f"[tta] {album.tta.name}: ffmpeg 不可用，跳过切分（安装 ffmpeg 后重跑）"
            )
            continue
        if store is not None and store.is_album_scanned(album.tta, album.cue):
            albums_skipped += 1
            logger.info("[state] %s: 专辑已在扫描记录中（源未变化），整张跳过切分", album.tta.name)
            continue
        try:
            tracks = splitter.split(
                album,
                out_dir=config.split_dir,
                reuse_existing=config.reuse_split_tracks,
            )
        except (SplitError, CueParseError) as exc:
            warnings.append(f"[tta] {album.tta.name}: 切分失败 — {exc}")
            continue
        if config.skip_tagged:
            for task in tracks:
                if get_styles(task.path, key=config.custom_tag_key):
                    task.skip_reason = f"已有自定义标签 {config.custom_tag_key}"
        elif store is not None:
            for task in tracks:
                if not task.skip_reason and store.is_scanned(task.path):
                    task.skip_reason = f"{ScanStateStore.SKIP_PREFIX}（指纹一致）"
        pending.extend(tracks)

    # 去重：上一轮切出的分轨在再次扫描时会同时以"直连文件"和"TTA 分轨"
    # 两种身份出现；按路径去重并保留元数据更全的分轨任务，避免重复分析。
    seen: dict[Path, int] = {}
    deduped: list[TrackTask] = []
    for task in pending:
        index = seen.get(task.path)
        if index is None:
            seen[task.path] = len(deduped)
            deduped.append(task)
        else:
            prev = deduped[index]
            if prev.title is None and task.title is not None:
                deduped[index] = task  # 分轨任务覆盖直连任务，信息更全
                logger.debug("合并重复任务: %s（保留分轨元数据）", task.path.name)

    logger.info("切分阶段完成: 待分析歌曲共 %d 首", len(deduped))
    state_out = PipelineState(tasks=deduped, albums=state.albums, warnings=warnings)
    state_out.stats["albums_skipped_scanned"] = albums_skipped
    return state_out


def _analyze_one(task: TrackTask, client: SongAnalysisClient) -> None:
    try:
        task.analysis = client.analyze(task.path)
    except SongAnalysisError as exc:
        task.error = str(exc)
    except Exception as exc:  # 客户端实现的自定义异常兜底
        task.error = f"分析异常（{type(exc).__name__}）: {exc}"


def step_analyze(
    state: PipelineState,
    config: WorkflowConfig,
    client: SongAnalysisClient,
    store: Optional[ScanStateStore] = None,
) -> PipelineState:
    """并发调用分析端点，逐任务填充 analysis / error。"""
    if store is not None:
        for task in state.tasks:
            if not task.skip_reason and store.is_scanned(task.path):
                task.skip_reason = f"{ScanStateStore.SKIP_PREFIX}（指纹一致）"
                logger.debug("[state] %s 命中扫描记录，跳过分析", task.path.name)
    pending = [t for t in state.tasks if not t.skip_reason]
    workers = max(1, min(config.concurrency, max(len(pending), 1)))
    if not pending:
        logger.info("没有需要分析的歌曲（全部跳过）")
        return state

    logger.info("提交 %d 首歌曲到分析端点（并发 %d, 端点: %s）", len(pending), workers, client.name)
    if workers == 1:
        for i, task in enumerate(pending, 1):
            _analyze_one(task, client)
            logger.info("[%d/%d] %s %s", i, len(pending), task.path.name,
                        "OK" if task.analysis is not None else f"失败: {task.error}")
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analyze") as pool:
            futures = {pool.submit(_analyze_one, task, client): task for task in pending}
            done = 0
            for future in as_completed(futures):
                task = futures[future]
                future.result()  # _analyze_one 不抛异常，这里只做同步点
                done += 1
                logger.info("[%d/%d] %s %s", done, len(pending), task.path.name,
                            "OK" if task.analysis is not None else f"失败: {task.error}")
    return state


def step_parse(state: PipelineState, config: WorkflowConfig, parser: StyleParser) -> PipelineState:
    """从分析响应中提取风格（JSON 优先，LLM 兜底）。"""
    for task in state.tasks:
        if task.error or task.skip_reason or task.analysis is None:
            continue
        try:
            styles, source, confidence = parser.parse(task.analysis, source_name=task.path.name)
        except StyleParseError as exc:
            task.error = str(exc)
            continue
        task.styles = styles
        task.style_parser = source
        task.confidence = confidence
        logger.info(
            "[styles] %s -> %s（%s 解析%s）",
            task.path.name,
            "/".join(styles),
            source,
            f", 置信度 {confidence:.2f}" if confidence is not None else "",
        )
    return state


def step_tag(state: PipelineState, config: WorkflowConfig, store: Optional[ScanStateStore] = None) -> PipelineState:
    """把风格写回音频自定义标签；dry_run 只记录将写入的值。

    写入成功后同时记入扫描状态（skip-scanned 机制的标记点）。
    """
    for task in state.tasks:
        if task.skip_reason or task.error or not task.styles:
            continue
        if config.dry_run:
            task.would_write = config.style_joiner.join(task.styles)
            logger.info("[dry-run] %s: 将写入 %s=%r", task.path.name, config.custom_tag_key, task.would_write)
            continue
        try:
            task.would_write = write_styles(
                task.path, task.styles, key=config.custom_tag_key, joiner=config.style_joiner
            )
            task.tagged = True
            if store is not None:
                store.mark_scanned(task.path, styles=task.styles)
            logger.info(
                "[tag] %s <= %s=%r", task.path.name, config.custom_tag_key, task.would_write
            )
        except TaggerError as exc:
            task.error = str(exc)
    return state


# ----------------------------------------------------------------------
# 组装与运行
# ----------------------------------------------------------------------
def build_workflow(
    client: SongAnalysisClient,
    style_parser: StyleParser,
    config: WorkflowConfig,
    *,
    splitter: Optional[TtaSplitter] = None,
    store: Optional[ScanStateStore] = None,
) -> RunnableSequence:
    """把五个步骤组装为 LangChain RunnableSequence。"""
    return (
        RunnableLambda(lambda _: step_scan(_, config), name="scan")
        | RunnableLambda(lambda state: step_split(state, config, splitter, store), name="split_tta")
        | RunnableLambda(lambda state: step_analyze(state, config, client, store), name="analyze")
        | RunnableLambda(lambda state: step_parse(state, config, style_parser), name="parse_styles")
        | RunnableLambda(lambda state: step_tag(state, config, store), name="write_tags")
    )


def run_workflow(
    client: SongAnalysisClient,
    style_parser: StyleParser,
    config: WorkflowConfig,
    *,
    splitter: Optional[TtaSplitter] = None,
    store: Optional[ScanStateStore] = None,
) -> RunReport:
    """端到端执行工作流并返回报告。

    - 未显式传入 splitter 时自动构造（ffmpeg 不可用则优雅降级，仅处理 mp3/flac）；
    - ``config.skip_scanned`` 且非 dry-run 时自动维护扫描记录：
      运行前加载（``reset_state`` 则先清空），成功任务在写标签后标记，
      专辑则在全部分轨成功后标记，结束时原子保存；dry-run 不产生任何
      状态记录（保持预览无副作用）。
    """
    if splitter is None:
        try:
            splitter = TtaSplitter()
        except SplitError as exc:
            logger.warning("%s（该异常不致命，将只直连处理 mp3/flac）", exc)
            splitter = None

    owns_store = False
    if store is None and config.skip_scanned and not config.dry_run:
        store = ScanStateStore(config.root, config.state_file)
        owns_store = True
        if config.reset_state:
            store.reset()
        else:
            store.load()

    workflow = build_workflow(client, style_parser, config, splitter=splitter, store=store)
    logger.info("LangChain 工作流启动: %s", " -> ".join(getattr(step, "name", "?") for step in workflow.steps))
    final_state: PipelineState = workflow.invoke(None)
    logger.info("工作流执行完毕")

    marked = 0
    if store is not None and not config.dry_run:
        marked = sum(1 for t in final_state.tasks if t.tagged)
        # 专辑级标记：该 TTA 的全部分轨都成功处理才跳过下次切分
        for album in final_state.albums:
            tracks = [t for t in final_state.tasks if t.source == album.tta]
            if tracks and all(t.tagged or (t.skip_reason and not t.error) for t in tracks):
                store.mark_album(album.tta, album.cue)
    if owns_store and store is not None:
        store.save()

    report = RunReport(
        tasks=final_state.tasks,
        warnings=final_state.warnings,
        albums_skipped_scanned=int(final_state.stats.get("albums_skipped_scanned", 0)),
        state_path=store.path if store is not None else None,
    )
    if config.dry_run:
        report.warnings.append("[mode] dry-run：风格仅解析与展示，未写入任何文件")
    elif store is not None:
        logger.info("[state] 本次标记 %d 首歌曲，扫描记录共 %d 项", marked, store.size)
    return report
