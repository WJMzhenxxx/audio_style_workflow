"""audio_style_workflow —— 基于 LangChain 的歌曲风格分析与标签回写工作流。

核心流程::

    scan(扫描目录) -> split_tta(整数轨切分) -> analyze(提交分析端点)
        -> parse_styles(解析风格) -> write_tags(写回自定义标签)

入口::

    python -m audio_style_workflow.main <音乐目录> --endpoint http://...
"""

__version__ = "0.1.0"
