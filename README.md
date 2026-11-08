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
├── logging_setup.py   # 日志（log/latest.log）+ 逐曲分析明细（log/details/）
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

# 基础用法（固定走 /generate，音频以 base64 内联，无需共享文件系统）
python -m audio_style_workflow.main "D:\我的音乐" --sglang --sglang-audio-base64 --skip-scanned

# 或用独立参数 --sglang-base 指定地址（等价写法）
python -m audio_style_workflow.main "D:\我的音乐" --sglang-base http://192.168.1.10:30000

# 或用环境变量 SGLANG_BASE（优先级：--sglang-base > --sglang BASE_URL > SGLANG_BASE > 默认）
set SGLANG_BASE=http://192.168.1.10:30000
python -m audio_style_workflow.main "D:\我的音乐" --sglang
```

> **固定使用 `/generate`**：`/v1/chat/completions` 在部分 sglang 版本里不会把
> 音频送进模型（响应的 `prompt_tokens` 只有纯文本规模），因此 CLI 不再提供
> `--sglang-api`，一律走 `/generate`（`audio_data` 字段）。

#### 让模型输出中文

默认 `--prompt-lang zh`：**风格名、Tempo、Key 保持英文**（`Dream Pop`、
`Moderato`、`F# major`），只有 `Why` 和评价正文用中文——这样风格标签可检索、
可与国际元数据对齐，评述则是母语。字段名（`BPM:` / `Tempo:` / `Key:` / `Why:`）
也保持英文以便确定性解析：

```bash
python -m audio_style_workflow.main "D:\我的音乐" --sglang http://10.0.0.5:43449 \
    --sglang-audio-base64            # 默认就是中文
```

想整段都用英文就加 `--prompt-lang en`。
中文场景下的风格解析依然稳定：`第1行：…`、`只写风格标签，…：` 这类回声会被
自动剥掉；即使模型偶尔给出中文风格名（`流行`、`节奏布鲁斯`）也能正常入库。

#### thinking budget（需要服务端 `--enable-custom-logit-processor`）

Thinking budget 必须同时提供 `custom_logit_processor` 的**序列化字符串**。
它由 sglang（OpenMOSS `moss-audio` 分支）的
`Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()` 产出，
内容形如 `{"callable": "<dill 序列化 hex>"}`——**不能只写类名**：
服务端会用 `orjson.loads` 解析该字段，传类名会让 sglang scheduler 抛
`JSONDecodeError` 并崩溃（SIGQUIT）。

```bash
# 1) 在装有该 sglang 分支的环境里导出序列化串
python -c "from sglang.srt.sampling.custom_logit_processor import \
Qwen3InstructionInjectionThinkingBudgetLogitProcessor as P; print(P.to_str())" > processor.json

# 2) 传给工作流（二选一）
python -m audio_style_workflow.main "D:\我的音乐" --sglang http://10.0.0.5:43449 \
    --sglang-thinking-budget 50 --sglang-custom-logit-processor @processor.json

set SGLANG_CUSTOM_LOGIT_PROCESSOR=<上面输出的一整行 JSON>
python -m audio_style_workflow.main "D:\我的音乐" --sglang http://10.0.0.5:43449 \
    --sglang-thinking-budget 50
```

工作流会在**发送前**校验该字段是合法 JSON，不合法立即报错退出（不会波及服务端）；
若本机恰好装了该 sglang 分支，则会自动调用 `to_str()`，无需手工传参。
想跳过 budget 直接去掉 `--sglang-thinking-budget` 即可。

> 为什么服务端必须开 `--enable-custom-logit-processor`：该字段会被服务端
> `dill.loads()` **反序列化并在 scheduler 进程里执行**，等于把代码执行权交给请求方，
> 所以 sglang 默认关闭，必须由运维显式开启——这是服务端的安全边界，客户端无法绕过。
> 客户端能做的**自适应**：若服务端拒绝该字段，工作流会自动去掉 budget 重新请求一次，
> 并在 `latest.log` 打一条 warning、在分析明细里标注 `thinking_budget_applied=false`，
> 保证整批任务不因此失败。仍可用 `separate_reasoning`（只需 `--reasoning-parser`，
> 不需要 `--enable-custom-logit-processor`）拿到思考过程。

参数说明：`--sglang [BASE_URL]` 不带值时读环境变量 `SGLANG_BASE`，再回退
`http://localhost:30000`；`--sglang-base BASE_URL` 是显式指定地址的等价形式
（优先级最高）。推理结果统一解析为 `styles` + `description`，下游流程不变。

#### 跨机部署：base64 内联（推荐）或路径映射

`sglang` 默认**在服务端按路径读音频**。如果客户端是 Windows、服务端是 Linux
（或两台不同机器），直接发本地绝对路径 `C:\...` 服务端打不开——模型拿不到
音频，返回一段与歌曲无关的通用描述（典型表现：**不同歌曲返回完全相同的文本**）。

```bash
# A) base64 内联：不依赖共享文件系统（推荐）
--sglang-audio-base64
```

客户端会把音频编码成 `data:<mime>;base64,<...>` 放进 `audio_url`。
sglang 的 `load_audio` 原生支持 `data:` URL（注意：**裸 base64 不支持**，
会被当成本地路径）。代价是请求体增大约 33%，大文件会更慢。

```bash
# B) 路径映射：服务端确实能看到这份文件时才用
--sglang-path-map "C:\proj\audio_style_workflow\music=/data/music"

# C) 服务端挂载同一份目录（NFS / SMB / sshfs），路径本身一致
```

客户端还会检测"连续多首返回完全相同文本"并打印 warning 提示输入可能没送达，
同时把响应文本前 200 字打进日志、把 `finish_reason` 记进 `details/*.json`
（`length` 表示被 `max_tokens` 截断，`stop` 表示模型自己结束）。

### 单曲探测（排查「输出与歌曲无关」）

```bash
# 直接对服务端发一次请求，打印 finish_reason / usage / content / reasoning / 原始 JSON
python examples/probe_sglang.py "music/Aphonia.mp3" \
    --base-url http://10.0.0.5:43449 --base64 --max-tokens 2048

# 走 /generate（audio_data）路径
python examples/probe_sglang.py "music/Aphonia.mp3" \
    --base-url http://10.0.0.5:43449 --base64 --api generate
```

#### 已知问题：部分 sglang 版本的 chat 端点会丢弃音频

某些版本（实测 MOSS-Music + `moss-audio` 分支）的 `/v1/chat/completions`
**不会把 `audio_url` 转成音频输入**：请求发出 19 MB，响应里
`prompt_tokens` 却只有 ~105（纯文本规模），模型只是在续写 prompt，
于是不同歌曲得到几乎相同的答案。同一服务端的 `/generate`（`audio_data`）
则正常：`prompt_tokens` 跳到 5000+，输出随歌曲变化。

因此 **CLI 已固定为 `/generate`**，不再提供 `--sglang-api` 开关。
（库层面 `SglangMusicClient(api_mode="chat")` 仍保留，且会检测到音频缺失后
自动回落到 `/generate`，但命令行不会走这条路。）

#### `content` 为 null（答案落在 `reasoning_content`）

thinking 模型（或 chat 模板默认开启 thinking 时）常见现象：思考段没闭合就用完了
`max_tokens`，于是 `content: null`，真正的答案全在 `reasoning_content` 里。
客户端会**自动回退**使用 `reasoning_content`，并在分析明细里标注
`content_source: "reasoning_content"`；日志同时打一条 warning。

根因仍然是"思考太长"，可选处置：

```bash
# 1) 增大正文预算，让思考自然闭合并写出 content
--max-new-tokens 4096

# 2) 用 thinking budget 强制思考收尾（需服务端 --enable-custom-logit-processor）
--sglang-thinking-budget 0 --sglang-custom-logit-processor @processor.json

# 3) 让 prompt 只要求风格标签，减少无谓的描述
```

> `--sglang-thinking-budget 0` 表示"不允许思考"，会立即闭合思考段，对打标签最快。

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
| `--no-musical-tags` | 不写入 BPM/Tempo/Key 标签（默认写入） |
| `--skip-tagged` | 跳过已有该自定义标签的歌曲（依据文件内标签） |
| `--skip-scanned` | 跳过扫描记录中已成功处理且未变化的歌曲（TTA 专辑整张跳过切分；文件变动后自动重处理） |
| `--state-file` | 扫描记录文件路径，默认 `<root>/.audio_style_scan.json` |
| `--reset-state` | 运行前清空扫描记录，重新全量处理 |
| `--dry-run` | 全流程演练但不写任何文件，仅展示将写入的值（也不产生扫描记录） |
| `--split-dir` | TTA 分轨输出目录（默认与整轨同目录） |
| `--resplit` | 强制重新切分（默认复用已有分轨） |
| `--mock` | 使用进程内 Mock 端点（联调用，无网络） |
| `--local` | 使用本地 MusicFlamingo 模型离线推理（需 `pip install -r requirements-local.txt`） |
| `--sglang [BASE_URL]` | 使用 sglang 部署的音乐理解模型（默认读 `SGLANG_BASE` 再回退 `http://localhost:30000`；音频传本机路径，需共享文件系统） |
| `--sglang-base BASE_URL` | 显式指定 sglang 服务基地址（等价 `--sglang BASE_URL`，优先级最高） |
| `--sglang-thinking-budget N` | 限制思考 token 数（需同时提供下面的处理器串） |
| `--sglang-custom-logit-processor` | thinking budget 的 `custom_logit_processor` 序列化串（`to_str()` 输出）；支持 `@文件` 或环境变量 `SGLANG_CUSTOM_LOGIT_PROCESSOR` |
| `--sglang-path-map LOCAL=REMOTE` | 把本地路径前缀映射为服务端可见路径，可重复（共享文件系统时用） |
| `--sglang-audio-base64` | 音频以 `data:` URL（base64）内联发送，无需共享文件系统 |
| `--prompt-lang {zh,en}` | 默认 `zh`：风格名/Tempo/Key 用英文，Why 与正文用中文；`en` 则整段英文 |
| `--model-id` | 本地 MusicFlamingo 模型 HuggingFace ID，默认 `nvidia/music-flamingo-2601-hf` |
| `--max-new-tokens` | 模型生成 token 上限（--local / --sglang 共用；sglang 建议 1024） |
| `--concurrency` | 分析端点并发请求数（本地模型自动串行化） |
| `--llm-model` | 启用 LLM 兜底解析非结构化响应（需 `langchain-openai` + `OPENAI_API_KEY`） |
| `--log-dir DIR` | 日志目录，默认 `log`（或环境变量 `SONG_ANALYZER_LOG_DIR`） |

环境变量：`SONG_ANALYZER_ENDPOINT`、`SONG_ANALYZER_API_KEY`、
`SONG_ANALYZER_FILE_FIELD`、`SONG_ANALYZER_LLM_MODEL`、`FFMPEG_PATH`、
`SGLANG_BASE`、`SGLANG_CUSTOM_LOGIT_PROCESSOR`、`SONG_ANALYZER_LOG_DIR`。

## 日志与逐曲分析明细

每次运行都会在 `--log-dir`（默认 `log/`）下产出：

```
log/
├── latest.log              # 本次运行的全量日志（每次运行覆盖，UTF-8）
├── details/
│   ├── 某首歌__<hash>.json
│   └── ...
└── Genre description/
    ├── 某首歌__<hash>.txt  # 该曲的完整评价（可读文本）
    └── ...
```

- `latest.log`：完整的运行日志（含扫描、切分、分析、解析、写标签各阶段）；
- `details/*.json`：每首歌一次分析的**原始输入输出**，内容形如

```json
{
  "audio": "D:\\music\\song.flac",
  "recorded_at": "2026-11-09T00:20:00+08:00",
  "request": {
    "audio": "D:\\music\\song.flac",
    "audio_size": 10485760,
    "client": "sglang-music",
    "endpoint": "http://localhost:30000",
    "api_mode": "chat",
    "thinking_budget": 50,
    "max_new_tokens": 1024,
    "prompt": "Listen to this track and analyze it..."
  },
  "response": {
    "status": "ok",
    "styles": ["Pop", "R&B"],
    "description": "...",
    "raw_response": {"choices": [{"message": {"content": "...", "reasoning_content": "..."}}]}
  },
  "error": null,
  "elapsed_sec": 3.21
}
```

- `Genre description/*.txt`：每首歌的**完整评价**（人类可读），分两节：

  ```text
  歌曲: Aphonia.mp3
  路径: C:\music\Aphonia.mp3
  风格: Pop, R&B
  BPM: 118
  Tempo: Moderato
  Key: C major
  来源: content
  ============================================================

  【为什么是这个风格】

  中板律动与温暖和声是 R&B 的典型骨架。

  ------------------------------------------------------------

  【完整评价】

  Genre: Pop, R&B
  ...
  ```

  「为什么是这个风格」由 `extract_genre_rationale` 确定性解析：优先取带标签的
  段落（`Why:` / `Reason:` / `理由：` / `依据：`…，可跨多行），没有标签时退化为
  找同时含因果词（because/因为）与音乐要素词（style/vocal/节奏…）的句子；
  解析不到就不输出该节。「来源」标明正文取自 `content` 还是 `reasoning_content`。
  文件名与 `details` 同名前缀，可直接对照原始请求/响应。

- 文件名由「歌名 + 绝对路径哈希」决定：同一首歌多次分析覆盖同一文件，并发安全；
- 分析失败也会落一份明细（`error` 字段记录原因）；
- 库/Notebook 里通过 `WorkflowConfig(log_dir=Path("log"))` 启用（默认不产生文件）；
  `main.py` 默认写入 `log/`，可用 `--log-dir` 改路径；
- 目录内容已加入 `.gitignore`（`log/`）。

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

除风格外，还会写入从模型输出里提取的 **BPM / Tempo / Key**（可用
`--no-musical-tags` 关闭）：

| 属性 | FLAC (VorbisComment) | MP3 (ID3v2.4) | 示例 |
| --- | --- | --- | --- |
| BPM | `BPM` | 标准帧 `TBPM` | `120` |
| Tempo | `TEMPO` | `TXXX:TEMPO` | `Moderato` / `慢板` |
| Key | `KEY` | 标准帧 `TKEY` | `C major` / `A小调` |

提取是**确定性的**（无需 LLM）：既认带标签的写法 `BPM: 120` / `Tempo: Moderato`
/ `Key: C major`（含中文 `速度:` / `调性:`），也认行内写法 `120 BPM`、`F# minor`；
BPM 限定在 20–400，越界视为无效。三个值都没提取到时不会写入任何额外字段。

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
