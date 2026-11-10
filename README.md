# Audio Style Workflow

扫描音乐目录（mp3 / flac / wav）→ TTA 整轨按 CUE 切分 → 调用音乐理解模型 →
解析风格/BPM/调性/氛围/评分 → 写回音频标签（Windows 资源管理器可见）。

工作流用 LangChain LCEL 组装成 `RunnableSequence`，每一步是纯函数（状态进、状态出），
可单测、可替换、可单步调用。

```text
None ──▶ scan ──▶ split_tta ──▶ analyze ──▶ parse_styles ──▶ write_tags ──▶ [TrackTask]
         (扫描目录) (TTA+CUE    (调用分析     (确定性解析 /     (mutagen 写标签,
                    切分为 FLAC)  端点)         LLM 兜底)        支持 dry-run)
```

- 单曲失败只记录错误，不中断整批；
- 分析器可插拔：HTTP 端点 / sglang 部署的音乐模型 / 本地 MusicFlamingo / 进程内 mock。

## 安装

```bash
pip install -r requirements.txt          # 运行
pip install -r requirements-dev.txt      # 测试 / mock 服务端 / LLM 兜底
```

需要 Python 3.10+ 与 ffmpeg（TTA 切分；Windows：`winget install ffmpeg`）。

## 快速开始

```bash
# 1) 有真实端点
python -m audio_style_workflow.main "D:\我的音乐" \
    --endpoint http://127.0.0.1:8010/analyze --skip-tagged

# 2) 没有端点：进程内 mock 跑通链路
python -m audio_style_workflow.main .\music --mock

# 3) sglang 部署的音乐模型（推荐，跨机也能用）
python -m audio_style_workflow.main "D:\我的音乐" \
    --sglang http://10.0.0.5:43449 --sglang-audio-base64 --per-song --skip-scanned

# 4) 本地 MusicFlamingo 离线推理
pip install -r requirements-local.txt
python -m audio_style_workflow.main "D:\我的音乐" --local --skip-scanned
```

输出摘要：

```text
================ 处理摘要 ================
歌曲总数      : 23
已分析        : 23（JSON 提取 23 / LLM 兜底 0）
已写回标签    : 23
跳过          : 0
失败          : 0
```

`audio_style_workflow.ipynb` 是同一套流程的 Notebook 版本，含交互式单曲演示与逐段说明。

## 两种执行模式

| | 分阶段批处理（默认） | 逐曲闭环（`--per-song`） |
| --- | --- | --- |
| 顺序 | 先全部分析 → 再统一解析 → 再统一写标签 | 每首：分析 → 解析 → 写标签 → 记扫描记录 |
| 中断后 | 标签已写但扫描记录可能没落盘 | 标签与扫描记录都已落盘，`--skip-scanned` 直接续跑 |
| 全局视角 | 可先看全部分析结果 | 无法先汇总再决定 |
| 并发 | `--concurrency` 在本阶段内并行 | 同样支持 `--concurrency`（每首独立） |

大批量、可能被打断的场景建议逐曲模式（多首并发 + 每首完成即落盘）：

```bash
python -m audio_style_workflow.main "D:\我的音乐" \
    --sglang --sglang-audio-base64 --per-song --concurrency 4 --state-flush-every 1
```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `root` | 音乐库根目录（递归扫描） |
| `-e / --endpoint` | 分析端点 URL（或 `SONG_ANALYZER_ENDPOINT`） |
| `--api-key` | Bearer 鉴权（或 `SONG_ANALYZER_API_KEY`） |
| `--custom-tag` | 风格标签键名，默认 `GENRE` |
| `--joiner` | 多风格连接符，默认 `"; "` |
| `--skip-tagged` | 跳过已有该标签的歌曲（依据文件内标签） |
| `--skip-scanned` | 跳过扫描记录中未变化的歌曲（文件变动自动重处理） |
| `--state-file` | 扫描记录路径，默认 `<root>/.audio_style_scan.json` |
| `--state-flush-every N` | 扫描记录每 N 条落盘一次，默认 10（0 = 只在结束时） |
| `--reset-state` | 清空扫描记录后全量重跑 |
| `--per-song` | 逐曲闭环模式 |
| `--dry-run` | 全流程演练但不写文件、不产生记录 |
| `--split-dir` / `--resplit` | TTA 分轨输出目录 / 强制重新切分 |
| `--no-musical-tags` | 不写入 BPM/Tempo/Key/Mood/评分等额外标签 |
| `--tempo-from-bpm` | 用 BPM 换算 Tempo（忽略模型自报的速度词） |
| `--mock` | 进程内 mock 端点 |
| `--local` | 本地 MusicFlamingo 推理 |
| `--sglang [URL]` | 使用 sglang 服务（读 `SGLANG_BASE`，回退 `http://localhost:30000`） |
| `--sglang-base URL` | 显式指定 sglang 地址（优先级最高） |
| `--sglang-audio-base64` | 音频以 `data:` URL 内联，无需共享文件系统 |
| `--sglang-path-map LOCAL=REMOTE` | 本地路径前缀映射为服务端路径，可重复 |
| `--sglang-audio-16k` | 发送前整首转 16 kHz 单声道（模型内部就按 16k 解码，省 80%~97% 带宽） |
| `--sglang-audio-codec {wav,flac,mp3,opus}` | `--sglang-audio-16k` 的输出格式，默认 `wav`（无损） |
| `--sglang-audio-bitrate` | mp3/opus 码率，默认 `64k` |
| `--sglang-clip SECONDS` | 只发送前/中 N 秒（仅用于降低体积，会丢信息） |
| `--sglang-thinking-budget N` | 限制思考 token（需处理器串） |
| `--sglang-custom-logit-processor` | thinking budget 的序列化串，支持 `@文件` |
| `--prompt-lang {zh,en}` | 默认 `zh`：风格名/Tempo/Key 英文，评述中文 |
| `--max-new-tokens` | 生成上限，sglang 建议 1024~2048 |
| `--temperature/--top-p/--top-k` | 采样参数，默认贪心（`temperature 0`） |
| `--concurrency` | 分析并发数，批处理与逐曲模式都生效（本地模型自动串行） |
| `--llm-model` | 非结构化响应的 LLM 兜底解析 |
| `--log-dir DIR` | 日志目录，默认 `log/` |

环境变量：`SONG_ANALYZER_ENDPOINT`、`SONG_ANALYZER_API_KEY`、`SONG_ANALYZER_FILE_FIELD`、
`SONG_ANALYZER_LLM_MODEL`、`SONG_ANALYZER_LOG_DIR`、`FFMPEG_PATH`、`SGLANG_BASE`、
`SGLANG_CUSTOM_LOGIT_PROCESSOR`。

## 写入的标签

| 属性 | FLAC（VorbisComment） | MP3 / WAV（ID3v2.4） | 示例 |
| --- | --- | --- | --- |
| 流派 | `GENRE` | 标准帧 `TCON` | `Pop; Dream Pop` |
| 氛围 | `MOOD` | 标准帧 `TMOO` | `温暖, 舒缓` |
| 评分 | `RATING`（0-100） | 标准帧 `POPM`（星级） | `80` / 4 星 |
| 评价 | `COMMENT` | 标准帧 `COMM` | `旋律抓耳，制作干净。` |
| BPM | `BPM` | 标准帧 `TBPM` | `120` |
| Tempo | `TEMPO` | `TXXX:TEMPO` | `Moderato` |
| Key | `KEY` | 标准帧 `TKEY` | `C major` |

- 支持 **mp3 / flac / wav**（WAV 用 RIFF 里的 ID3 chunk，帧名与 MP3 相同）；
- 默认键会映射到**标准帧**，Windows 资源管理器「属性 → 详细信息」直接可见；
  自定义键（如 `--custom-tag MY_STYLE`）仍写 `TXXX:<key>`；
- 提取是确定性的（无需 LLM），同时兼容缩略写法和中文标签；
- TTA 分轨另带 `TITLE / ARTIST / ALBUM / ALBUMARTIST / TRACKNUMBER`。

## 输出与日志

```
log/
├── latest.log                          # 本次运行全量日志
├── details/<歌名>__<hash>.json          # 每首歌的原始请求/响应
└── Genre description/<歌名>__<hash>.txt # 每首歌的可读评价（含「为什么是这个风格」）
```

CLI 默认写 `log/`，可用 `--log-dir` 修改；库/Notebook 需显式传
`WorkflowConfig(log_dir=Path("log"))`。`log/` 已在 `.gitignore` 中。

## 分析端点契约

`POST <endpoint>`（`multipart/form-data`，字段名 `--file-field`，默认 `file`）：

```json
{ "status": "ok", "styles": ["Pop", "R&B"], "confidence": 0.87 }
```

- 风格字段兼容 `styles/style/genres/genre/music_style(s)/music_genre(s)/tags`，
  字符串按 `, ; / | 、` 拆分，支持一层嵌套；
- `status` 为 `error/failed` 或 HTTP ≥400 按失败处理；
- 完全非结构化响应可配 `--llm-model` 用 LLM 提取。

接入自有服务只需实现接口：

```python
from audio_style_workflow.api.base import SongAnalysisClient

class MyClient(SongAnalysisClient):
    name = "my-service"
    def analyze(self, audio_path):
        return {"styles": ["Pop"], "description": "..."}
```

## 断点续跑

| | `--skip-tagged` | `--skip-scanned` |
| --- | --- | --- |
| 依据 | 文件内标签 | 状态文件指纹（大小 + mtime） |
| 换 tag key 重跑 | 失效 | 照常命中 |
| TTA 专辑级跳过 | 无 | 专辑未变时连切分都跳过 |
| 产生文件 | 无 | `<root>/.audio_style_scan.json` |

```bash
python -m audio_style_workflow.main "D:\我的音乐" --sglang --skip-scanned
```

首次全量处理并生成记录，之后只处理新增/变动的文件；`--reset-state` 强制全量。
`--dry-run` 不读写扫描记录。

## 标签导出 / 导入

把标签导出成 JSON，换台机器或换个目录结构也能灌回去：

```bash
python export_tag.py -i D:\我的音乐 -o tags.json
python import_tag.py -i tags.json -o E:\我的音乐
```

匹配优先级：**相对路径 → 文件 MD5 → 音频 MD5 → 文件名**（忽略大小写，同名多份时用 MD5 收敛）。

- **文件 MD5**：字节完全一致的副本（整库拷贝）直接命中；
- **音频 MD5**：解码后 PCM 的 md5（`ffmpeg -f md5`），重写标签、改文件名后仍稳定，
  所以「干净的原始文件 + 另一个库的标签」也能对上；
- 导出还会记录 `size`、基础标签与（TTA 分轨的）来源信息。

TTA 整轨：JSON 里记 `tta_rel_path` / `tta_md5` / `cue_md5` / `track_no`，
导入时若分轨还不存在，会**先按 CUE 切分再写标签**（`--no-split` 可关闭）。

```json
{
  "version": 1,
  "tag_keys": {"custom": "GENRE", "joiner": "; "},
  "tracks": [
    {
      "rel_path": "album/01 - Song.flac",
      "name": "01 - Song",
      "md5": "<文件 md5>",
      "audio_md5": "<解码音频 md5>",
      "tags": {"GENRE": "Pop; Dream Pop", "BPM": "120", "RATING": "9", "COMMENT": "…"},
      "album": {"tta_rel_path": "album/album.tta", "tta_md5": "…", "track_no": 1}
    }
  ]
}
```

常用选项：`--dry-run`（只匹配不写）、`--include-untagged`（连无标签文件也记录）、
`--no-audio-hash`（跳过较慢的音频 md5）、`--custom-tag` / `--joiner`（覆盖键名）、`--ffmpeg`。

## 流程细节

- **TTA 切分**：TTA 与 FLAC 均为无损，逐轨转 FLAC 无质量损失；切点取 `INDEX 01`
  （缺失回退首个 INDEX），帧时间按 75fps 换算；使用精确 seek + `.part` 临时文件。
- **CUE 配对**：优先同名 `foo.tta` + `foo.cue`，否则按 CUE 的 `FILE` 指令精确配对；
  配不上的 TTA 记警告跳过，不动原文件。
- **编码**：CUE 自动尝试 UTF-8(BOM) → GBK → Latin-1。
- **失败隔离**：网络、响应、解析、写标签失败都只标记该曲目，结束时汇总；
  存在失败时退出码非 0。

## 测试

```bash
python -m pytest tests/ -v
```

单元测试覆盖 CUE 解析、风格提取、标签读写、扫描配对、状态存储；
端到端测试用 ffmpeg 真实生成 TTA 整轨，验证切分、专辑标签、断点续跑与 dry-run。

## 更多文档

- `export_tag.py` / `import_tag.py`：标签导出与导入（见上节）
- [docs/moss-music.md](docs/moss-music.md)：MOSS-Music 部署、参数、prompt 设计与实测数据
- [docs/troubleshooting.md](docs/troubleshooting.md)：输出异常、复读、续跑、日志排查
