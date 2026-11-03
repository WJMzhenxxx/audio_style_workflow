# Audio Style Workflow（LangChain 版）

扫描目录（含子目录）中的音乐文件 -> TTA 整轨按 CUE 切分 -> 提交歌曲分析端点
（HTTP 上传）-> 解析歌曲风格 -> 写回自定义标签（MP3 `TXXX:<key>` / FLAC
VorbisComment `<key>`）。

## 架构

```
        ┌────────────────────────  LangChain RunnableSequence  ────────────────────────┐
        │                                                                              │
None ──▶│  scan ──▶ split_tta ──▶ analyze ──▶ parse_styles ──▶ write_tags ──▶ [TrackTask]
        │  (扫描   (TTA+CUE     (HTTP 上传     (JSON 提取 /       (mutagen
        │  目录)   ffmpeg 切分    到分析端点,    LLM 兜底)          写自定义标签,
        │          为 FLAC 分轨)  支持并发)                          支持 dry-run)
        └──────────────────────────────────────────────────────────────────────────────┘
```

- 每步是一个 `RunnableLambda`（纯函数：状态进、状态出），用 `|` 串接，
  可单测、可替换、可用 langchain 的回调体系观测。
- 单文件失败只记录错误，不中断整批；最终输出可读摘要报告。

```
audio_style_workflow/
├── models.py          # 数据模型（TrackTask / TtaAlbum / PipelineState...）
├── cue.py             # CUE 解析（UTF-8/GBK/帧时间）
├── scanner.py         # 递归扫描 & TTA-CUE 配对
├── splitter.py        # ffmpeg 按 CUE 切分 TTA 为带标签的 FLAC 分轨
├── api/
│   ├── base.py              # ★ 分析端点接口（先接口）
│   ├── http_client.py       # ★ 基于 HTTP multipart 上传的实现
│   ├── local_model_client.py# ★ 本地 MusicFlamingo 模型推理实现（离线）
│   ├── sglang_client.py     # ★ sglang 部署的音乐理解模型实现（音频传本机路径）
│   └── mock_client.py       # 进程内演示实现（无需网络）
├── style_parser.py    # 风格解析（确定 JSON 提取 + LLM 兜底）
├── tagger.py          # 自定义标签读写（mutagen）
├── workflow.py        # LangChain(LCEL) 工作流组装
└── main.py            # CLI 入口
examples/mock_server.py # FastAPI Mock 分析端点（本地联调）
tests/                   # 单元测试 + ffmpeg 端到端测试
```

## 安装

1. Python 3.10+，ffmpeg 可用（用于 TTA 切分；Windows 参见 `winget install ffmpeg`）。
2. 安装依赖（本目录下）：

```bash
pip install -r requirements.txt          # 运行
pip install -r requirements-dev.txt      # 测试 / mock 服务端 / LLM 兑底
```

## 快速开始

```bash
# 1) 起 mock 分析端点（可选，有真实端点可跳过）
python -m uvicorn examples.mock_server:app --port 8010

# 2) 跑工作流
python -m audio_style_workflow.main "D:\我的音乐" \
    --endpoint http://127.0.0.1:8010/analyze \
    --custom-tag STYLE \
    --skip-tagged
```

输出示例：

```
================ 处理摘要 ================
歌曲总数      : 23
已分析        : 23（JSON 提取 23 / LLM 兜底 0）
已写回标签    : 23
跳过          : 0
失败          : 0
警告          : 1
```

### 离线联调（无端点 / 无网络）

```bash
python -m audio_style_workflow.main .\music --mock          # 端点用进程内 mock
python -m audio_style_workflow.main .\music --mock --mock-freeform --llm-model gpt-4o-mini
#   freeform 返回非结构化文本，验证 LLM 兑底解析（需 OPENAI_API_KEY 环境变量）
```

### sglang 部署的音乐理解模型（moss-music 等）

适配 sglang serve 的两种 API；音频不走 HTTP 上传，服务端直接读路径，
因此要求音乐库与 sglang 服务**共享文件系统**（同机或挂载同一卷）：

```bash
# 服务端：sglang serve --model-path /path/to/moss-music-model --trust-remote-code

# 模式1：/generate 基础模式（默认）
python -m audio_style_workflow.main "D:\我的音乐" --sglang --skip-scanned

# 模式2：/v1/chat/completions（OpenAI 兼容）+ thinking 分离 + budget 控制
python -m audio_style_workflow.main "D:\我的音乐" --sglang http://localhost:30000 \
    --sglang-api chat --sglang-thinking-budget 50 --max-new-tokens 1024
```

参数说明：`--sglang [BASE_URL]` 不带值时默认 `http://localhost:30000`；
`--sglang-thinking-budget N` 自动附带 `separate_reasoning` 与
`custom_logit_processor`（Qwen3InstructionInjectionThinkingBudgetLogitProcessor
的惯用字符串，可用 `custom_logit_processor` 参数编程覆盖）。
推理结果同样统一解析为 `styles` + `description`，下游流程不变。

### 本地模型分析（离线推理，MusicFlamingo）

不依赖任何 HTTP 服务，用本地 transformers 权重直接推理（GPU 推荐，自动 `device_map=auto`）：

```bash
pip install -r requirements-local.txt        # torch / transformers / accelerate
set HF_ENDPOINT=https://hf-mirror.com        # 可选：国内镜像加速权重下载

python -m audio_style_workflow.main "D:\我的音乐" --local --skip-scanned
#   可选: --model-id nvidia/music-flamingo-2601-hf  --max-new-tokens 500
```

工作原理：prompt 引导模型把风格标签写在输出第一行，客户端确定性解析为
`styles`（零额外 LLM 成本），完整分析文本保留在 `description`；
若第一行解析不到标签，可加 `--llm-model` 从 description 兑底提取。
模型首次分析时才加载（懒加载），推理内部持锁串行化，`--concurrency` 下安全。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `root` | 音乐库根目录（递归） |
| `-e / --endpoint` | 分析端点 URL（或环境变量 `SONG_ANALYZER_ENDPOINT`） |
| `--api-key` | Bearer 鉴权（或 `SONG_ANALYZER_API_KEY`） |
| `--file-field` | 上传字段名，默认 `file` |
| `--custom-tag` | 自定义风格标签键名，默认 `STYLE`（FLAC 直接字段；MP3 写 `TXXX:STYLE`） |
| `--joiner` | 多风格连接符，默认 `"; "` |
| `--skip-tagged` | 跳过已有该自定义标签的歌曲（依据文件内标签） |
| `--skip-scanned` | 跳过扫描记录中已成功处理且未变化的歌曲（TTA 专辑整张跳过切分；文件变动后自动重处理） |
| `--state-file` | 扫描记录文件路径，默认 `<root>/.audio_style_scan.json` |
| `--reset-state` | 运行前清空扫描记录，重新全量处理 |
| `--dry-run` | 全流程演练但不写任何文件，仅展示将写入的值（也不产生扫描记录） |
| `--split-dir` | TTA 分轨输出目录（默认与整轨同目录） |
| `--resplit` | 强制重新切分（默认复用已有分轨） |
| `--mock` | 使用进程内 Mock 端点（联调用，无网络） |
| `--local` | 使用本地 MusicFlamingo 模型离线推理（需 `pip install -r requirements-local.txt`） |
| `--sglang [BASE_URL]` | 使用 sglang 部署的音乐理解模型（默认 `http://localhost:30000`；音频传本机路径，需共享文件系统） |
| `--sglang-api {generate,chat}` | sglang API 模式：`generate`=/generate；`chat`=/v1/chat/completions |
| `--sglang-thinking-budget N` | chat 模式开启 thinking 分离并限制思考 token 数 |
| `--model-id` | 本地 MusicFlamingo 模型 HuggingFace ID，默认 `nvidia/music-flamingo-2601-hf` |
| `--max-new-tokens` | 模型生成 token 上限（--local / --sglang 共用；sglang 建议 1024） |
| `--concurrency` | 分析端点并发请求数（本地模型自动串行化） |
| `--llm-model` | 启用 LLM 兜底解析非结构化响应（需 `langchain-openai` + `OPENAI_API_KEY`） |

环境变量：`SONG_ANALYZER_ENDPOINT`、`SONG_ANALYZER_API_KEY`、
`SONG_ANALYZER_FILE_FIELD`、`SONG_ANALYZER_LLM_MODEL`、`FFMPEG_PATH`。

## 分析端点契约

`POST <endpoint>`（`multipart/form-data`，字段名 `--file-field`，默认 `file`），
鉴权可选 `Authorization: Bearer <api-key>`。响应 JSON：

```json
{
  "status": "ok",
  "styles": ["Pop", "R&B"],
  "confidence": 0.87
}
```

- 风格字段兼容键名：`styles` / `style` / `genres` / `genre` /
  `music_style(s)` / `music_genre(s)` / `tags`；字符串值按 `, ; / | 、` 拆分；
  亦支持一层嵌套（`result/analysis/data/payload/output` 内部）。
- `status` 为 `error/failed` 或 HTTP ≥400 时按失败处理并在报告里列出原因。
- 完全非结构化的响应可配 `--llm-model` 用 LLM 提取风格。

### 接入自定义分析服务

已内置五种实现：`http`（默认）/ `mock`（联调）/ `local`（MusicFlamingo 离线推理）/
`sglang`（音乐理解模型服务）/ 自定义继承。接入新服务只需实现接口：

```python
from audio_style_workflow.api.base import SongAnalysisClient, SongAnalysisError

class MyClient(SongAnalysisClient):
    name = "my-service"
    def analyze(self, audio_path):
        ...  # 你的传输方式（预签名 URL、gRPC、消息队列……）
        return {"styles": [...], "description": "..."}
```

然后把它传给 `run_workflow(client=MyClient(), ...)` 即可，工作流其余部分不变。

## 断点续跑：两种跳过机制

| | `--skip-tagged` | `--skip-scanned` |
| --- | --- | --- |
| 依据 | 音频文件内的自定义标签 | 独立状态文件里的指纹（大小+mtime） |
| 换 tag key 重跑 | 失效（重新全量） | 照常命中 |
| 标签写入失败的服务重试 | 无法区分 | 失败不记录，仅成功的重跑时才跳过 |
| TTA 专辑级跳过 | 无 | 专辑（TTA+CUE）指纹一致时连切分都跳过 |
| 文件变动检测 | 无 | 大小/mtime变化后自动重新处理 |
| 产生文件 | 无（写入音频本身） | `<root>/.audio_style_scan.json` |

推荐命令：

```bash
python -m audio_style_workflow.main "D:\我的音乐" -e http://... --skip-scanned
```

首次全量处理并生成记录；之后重跑时未变动的歌曲（和源头未变的整轨专辑）
直接跳过，只处理新增/变动的文件。想重新全量分析时加 `--reset-state`。

dry-run 模式不读也不写扫描记录，保持预览无副作用。

## 自定义标签长什么样

| 格式 | 写入位置 | 查看方式 |
| --- | --- | --- |
| FLAC | VorbisComment 自由字段 `STYLE` | `ffprobe file.flac 2>&1 \| findstr STYLE` |
| MP3 | ID3v2.4 `TXXX:STYLE` 用户文本帧 | foobar2000 Columns UI / mp3tag 均可读 |

TTA 分轨额外会带上 `TITLE / ARTIST / ALBUM / ALBUMARTIST / TRACKNUMBER` 基础标签。

## 流程细节

- **TTA 切分**：TTA 与 FLAC 皆为无损编码，逐轨 `tta -> FLAC` 无质量损失。
  切点取每轨 `INDEX 01`（缺失时回退第一个 INDEX），帧时间按 75fps 换算；
  使用输出级精确 seek + `.part` 临时文件，避免半成品。
- **CUE 配对**：优先同名 `foo.tta` + `foo.cue`（忽略大小写）；否则解析目录
  内未匹配的 CUE，按其 `FILE` 指令精确配对；配不上的 TTA 记警告跳过（不动原文件）。
- **编码兼容**：CUE 自动尝试 UTF-8(BOM) -> GBK -> Latin-1。
- **失败隔离**：网络失败 / 响应异常 / 解析失败 / 写标签失败都只标记该曲目，
  在结束报告中汇总；进程退出码非 0 表示存在失败。

## 测试

```bash
python -m pytest tests/ -v
```

- 单元测试覆盖 CUE 解析、风格提取、标签读写、扫描配对；
- 端到端测试用 ffmpeg 实际生成 TTA 整轨，验证切分时长、专辑标签、
  断点续跑（--skip-tagged）与 dry-run 行为。
