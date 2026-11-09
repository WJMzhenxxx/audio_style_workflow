# 排错与续跑

## 输出与歌曲无关 / 每首歌结果都一样

先确认音频是否真的送达服务端：

```bash
# 直接发一次请求，打印 prompt_tokens / finish_reason / 原始 JSON
python examples/probe_sglang.py "music/song.mp3" \
    --base-url http://10.0.0.5:43449 --base64 --api generate
```

判据：**`prompt_tokens` 只有几百就是没吃到音频**（正常一首 3 分钟的歌在 2700 左右）。

常见原因：

| 原因 | 表现 | 处置 |
| --- | --- | --- |
| 用了 `/v1/chat/completions` | `prompt_tokens` ≈105 | 改走 `/generate`（CLI 已固定） |
| 只发了裸 base64 | 服务端当路径处理，读不到 | 需要 `data:<mime>;base64,...` 前缀 |
| 跨机发了本地绝对路径 | 服务端打不开 `C:\...` | 用 `--sglang-audio-base64` 或 `--sglang-path-map` |

客户端还会在连续多首返回相同文本时打 warning，并把 `finish_reason` 记进
`details/*.json`（`length` = 被 `max_tokens` 截断，`stop` = 模型自己结束）。

## `content` 为 null

thinking 模型常把答案写在 `reasoning_content` 里（思考段没闭合就用完了预算）。
客户端会自动回退并在明细里标 `content_source: "reasoning_content"`。

```bash
--max-new-tokens 4096                                   # 让思考自然闭合
--sglang-thinking-budget 0 --sglang-custom-logit-processor @processor.json   # 直接关掉思考
```

## 输出复读 / 超长

症状：单曲耗时异常（几十秒到几分钟）、`description` 上万字符、同一句话重复几十次，
后面的字段整段缺失。这是模型的重复退化（官方 issue #1 也记录过）。

客户端会检测「12 字片段重复 ≥6 次」，打 warning、在明细里写 `degenerate` 字段、
并截断重复尾巴。彻底缓解靠 prompt（见 [moss-music.md](moss-music.md)）：把容易诱发
罗列的小节排到最后，并禁止逐段列时间戳。

## 标签不准（子流派/风格漂移）

先确认不是"没听到音频"（见上），再按以下顺序排查：

1. 是否用了采样解码？`--temperature 1.0` 会让 BPM 跳 10~35，改用默认 0；
2. 是否需要 `--tempo-from-bpm` 让 Tempo 与 BPM 一致；
3. 流派本身是模型短板（`K-Pop` 等场景词不在其词汇表内），处理方式见
   [moss-music.md](moss-music.md) 的「能力边界」。

## 中断后怎么续跑

两种跳过机制：

| | `--skip-tagged` | `--skip-scanned` |
| --- | --- | --- |
| 依据 | 音频文件内的标签 | 独立状态文件里的指纹（大小 + mtime） |
| 换 tag key 重跑 | 失效（重新全量） | 照常命中 |
| TTA 专辑级跳过 | 无 | 专辑指纹一致时连切分都跳过 |
| 产生文件 | 无 | `<root>/.audio_style_scan.json` |

中断场景：

| 跳过参数 | 能否跳过已完成 |
| --- | --- |
| `--skip-scanned` | ✅ 前提是扫描记录已落盘。每 `--state-flush-every` 条（默认 10）落盘一次，Ctrl+C / 异常时会在 `finally` 再落盘；只有被强杀（kill -9 / 断电）才丢最后几条 |
| `--skip-tagged` | ✅ 读音频里的标签，逐曲模式下每首写完即在盘上；分阶段批处理要等 `write_tags` 阶段 |

推荐：`--per-song --skip-scanned`，对时间敏感可加 `--state-flush-every 1`。

## 日志在哪

```
log/
├── latest.log              # 本次运行全量日志（每次覆盖）
├── details/<歌名>__<hash>.json        # 每首歌的原始请求/响应
└── Genre description/<歌名>__<hash>.txt  # 每首歌的可读评价
```

`details/*.json` 里可查 `request.prompt`、`response.prompt_tokens`、
`finish_reason`、`elapsed_sec`、`error`、`degenerate` 等字段。
库/Notebook 里需显式 `WorkflowConfig(log_dir=Path("log"))` 才会落盘，
CLI 默认写 `log/`，可用 `--log-dir` 改。
