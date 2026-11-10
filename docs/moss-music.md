# MOSS-Music 使用与调优

记录本项目对接 [MOSS-Music](https://github.com/OpenMOSS/MOSS-Music)（8B Instruct / Thinking）
时的部署方式、参数选择与 prompt 设计依据。以下数据均来自本机实测（9 首曲目）。

## 部署

官方明确建议用 SGLang，Transformers 路径只用于调试：

```bash
sglang serve --model-path ./weights/MOSS-Music-8B-Instruct --trust-remote-code
```

- 若用 `torch==2.9.1+cu128`，先 `pip install nvidia-cudnn-cu12==9.16.0.29`；
- **打标签选 Instruct**：在官方 8 个音乐 QA 基准上 Instruct 平均 80.38，
  Thinking 只有 74.26（思考链对听感判断帮助不大）；
- 可选 `--reasoning-parser qwen3`（思考分离）、`--enable-custom-logit-processor`
  （thinking budget 必需）。

## 请求路径：只用 `/generate`

```json
{
  "text": "<prompt>",
  "audio_data": "<路径 | data:audio/wav;base64,...>",
  "sampling_params": {"max_new_tokens": 2048, "temperature": 0.0}
}
```

**不要用 `/v1/chat/completions`**：实测该端点在部分 sglang 版本里不把 `audio_url`
转成音频输入——请求发出 19 MB，响应 `prompt_tokens` 却只有 ~105（纯文本规模），
模型只是在续写 prompt，于是不同歌曲得到几乎相同的答案。同一服务端的 `/generate`
则正常（`prompt_tokens` 数千）。客户端已固定走 `/generate`，库层面若显式要求 chat
会在检测到音频缺失后自动回落。

## 音频输入

| 项 | 值 | 说明 |
| --- | --- | --- |
| 采样率 | 16 kHz 单声道（内部重采样） | 任意输入都能喂，客户端也可预转 |
| 音频 token | **12.5 / 秒** + 每 2 秒一个时间标记 | 实测折算 ≈13.8 token/秒 |
| 整首是否被听 | **是** | 见下方实测 |

整首摄入验证（文本 prompt 约 105 token，已扣除）：

| 音频 | 时长 | prompt_tokens | 音频 token | 折算 |
| --- | --- | --- | --- | --- |
| Aphonia 前 30 秒 | 30.0s | 502 | 397 | 13.2 t/s |
| Golden.flac 整首 | 194.6s | 2772 | 2667 | 13.7 t/s |
| Aphonia.mp3 整首 | 362.7s | 5126 | 5021 | 13.8 t/s |

严格线性 → 模型完整"听"了整首歌，**不存在只听前 30 秒**。

### 发送前转 16 kHz 能省多少带宽

模型内部固定按 16 kHz 单声道解码（`MelConfig.mel_sr = 16000`），所以高采样率
无损文件的额外信息**对模型无用**，客户端先转换再 base64 可省 80%~97%：

（实测源文件：48 kHz / 24 bit / 立体声 FLAC，194.6 秒，43.3 MB）

| 发送方式 | 实际字节 | base64 后 | 节省 |
| --- | ---: | ---: | ---: |
| 原文件（默认） | 43.27 MB | 57.69 MB | — |
| `--sglang-audio-16k`（wav） | 6.23 MB | 8.30 MB | 85.6% |
| `--sglang-audio-16k --sglang-audio-codec mp3` | 1.56 MB | 2.08 MB | 96.4% |
| `--sglang-audio-16k --sglang-audio-codec opus --sglang-audio-bitrate 48k` | 1.20 MB | 1.59 MB | 97.2% |

> 库里若是 **WAV**，原始文件尤其大（44.1k 立体声约 10 MB/分钟），更要开 `--sglang-audio-16k`。

**转完更大就自动跳过**：16k/16bit 单声道 PCM 固定 256 kbps，比 128 kbps 的
MP3 还大。遇到这种源文件（实测 3.2 MB → 6.0 MB）会直接发原文件——模型内部
同样重采样到 16k，体积更小且无质量损失。判断只在**没有截取**时生效：

| 源 | 原始 | 16k wav | 决策 |
| --- | ---: | ---: | --- |
| 128 kbps mp3（30s） | 481 KB | 960 KB | 发原文件 |
| 320 kbps mp3（30s） | 1203 KB | 960 KB | 转 16k |
| 无损 flac（30s，真实音乐） | 3599 KB | 960 KB | 转 16k |

`wav` 是 16k/16bit PCM，与模型输入完全一致（无损）；`mp3/opus` 更小但有损，
对流派/BPM 判断影响很小。注意 16k 单声道 FLAC 反而比 WAV 大（密集混音压不动），
所以默认用 `wav`。

因此 `--sglang-clip N` 只用于**降低请求体积**（60 秒约 2.5MB vs 整首 57.7MB），
不要指望它提升准确率：实测截取 30 秒后同一首歌的标签完全变样（`Aphonia` 整首是
`Dream Pop/Ambient Pop/Synth-Pop`，前 30 秒却成了 `ambient/new age/film score`）。

## 采样参数

| 来源 | temperature | top_p | top_k |
| --- | --- | --- | --- |
| 官方 `infer.py` | 1.0 | 0.8 | 50 |
| 官方 `app.py` | 1.0 | 1.0 | 50 |
| 官方文档示例 | 0.0 | — | — |
| **本项目默认** | **0.0** | — | — |

官方 T=1.0 是为文案多样性服务的；对**事实字段**（BPM/Key）有害：

| 曲目 | greedy BPM | T=1.0 BPM | Δ |
| --- | ---: | ---: | ---: |
| Aphonia | 100 | 136 | **+36** |
| What a Night | 100 | 135 | **+35** |
| Pictures | 128 | 143 | +15 |
| Silver Lining | 136 | 150 | +14 |
| Now and Never | 122 | 130 | +8 |
| River of Light | 132 | 137 | +5 |
| Golden | 125 | 127 | +2 |
| 日落潮汐 | 124 | 120 | -4 |

结论：想准就用默认贪心；想描述更丰富可用 `--temperature 0.3~0.5` 折中。
`--temperature/--top-p/--top-k` 均已开放。

模型自报的 `Tempo` 词常与 BPM 打架（136 BPM 却写 `Moderato`），
可用 `--tempo-from-bpm` 按 BPM 换算成
`Largo/Adagio/Andante/Moderato/Allegro/Vivace/Presto`。

## Prompt 设计

### 依据

官方训练语料由 [MOSS-Music-Data-Pipeline](https://github.com/wx9Songs/MOSS-Music-Data-Pipeline)
生成：ALM（Qwen3-Omni / MusicFlamingo）先打底 caption，再由 Qwen3-235B 结合 MIR
特征合成最终 caption，最后反向生成"用户提问"。所以模型的**原生输出是自然语言**，
要么一整段（paragraph），要么**分节小标题**（sections）：

```
**Tempo**  **Instrument:**  **Song Structure & Dynamics:**
**Harmony & Key:**  **Vocals & Lyrics:**  **Mood & Theme:**
```

本项目采用 sections 格式，额外补了 `**Genre**` / `**Why**` / `**Score**` / `**Review**`。
解析器同时兼容 `TAGS:` 行、第一行标签、分节小标题。

### 三条实测修正

**① 不要在示例里点名具体流派。**
曾在示例写 `K-Pop, City Pop`，结果 9 首全被标成 K-Pop——示例即 few-shot，会被照抄。

**② 字段顺序要按"重要性"排。**
`**Why**` 放在最后时 9 首里有 6 首整段漏写，挪到 `**Genre**` 之后即 9/9。
当前顺序：`Genre → Why → Tempo → Harmony & Key → Mood → Score → Review →
Instrument → Vocals & Lyrics → Song Structure & Dynamics`。

**③ 易失控的小节放最后。**
`**Song Structure & Dynamics**` 会诱发复读退化：实测一首歌输出 9368 字符、
同一片段重复 22 次，把 `max_new_tokens` 烧光，后面的 `Mood/Score/Review` 全缺。
现在它排在最后，并明确要求「3-5 句话概括、不要逐段罗列时间戳」，整体限长 400 字；
客户端还会检测 12 字 shingle 重复 ≥6 次并截断、打 warning（`degenerate` 字段）。

`**Review**` 限制在 60 字以内（太长会挤掉后面的小节，也会让 `COMM` 标签过长），
`**Mood & Theme**` 要求 2-3 个短词（模型倾向写整句解释，解析时会再裁一次）。

## 能力边界

| 字段 | 可信度 | 说明 |
| --- | --- | --- |
| 和弦 / 配器 / 结构描述 | ★★★★★ | 会给出 `Abmaj7、Fm、Gm7`、`i–VI–III–VII` 这类具体分析 |
| Key | ★★★★ | 跨版本稳定 |
| BPM | ★★★ | 贪心下稳定；采样会跳 10~35 |
| Mood | ★★★ | 主观但合理 |
| Tempo 词 | ★★ | 与 BPM 常矛盾，建议 `--tempo-from-bpm` |
| **Genre** | ★★ | 见下 |
| Score | ★ | 普遍 8-9 分，区分度低 |

**流派是短板**：模型按声音特征打标，缺少地域/产业场景先验。官方 GTZAN（10 类）
93.59 说明基础流派很强，但 `K-Pop` / `City Pop` 属于词表外概念——`Golden.flac`
听完整首给出的是 `Progressive House, Trance`（或 `Electronic Pop`），同类曲目
跨轮还会漂移。可靠的做法：

1. 把输出约束到固定词表（GTZAN 10 类 + 少量扩展）；
2. 场景标签单独用判断题问（"是否属于 K-Pop 场景？是/否 + 理由"），比开放生成准；
3. 多采样投票（`--temperature 0.6` 跑 N 次取多数）压住跨次漂移；
4. 接受现状，把 GENRE 当粗略检索线索，真正分类以 BPM/Key/Mood + 人工为准。

## Thinking 控制

| 手段 | 有效性 |
| --- | --- |
| `chat_template_kwargs.enable_thinking` | ❌ 只对纯文本 chat 生效，音频请求走模板短路分支 |
| thinking budget（`custom_params.thinking_budget`） | 需服务端 `--enable-custom-logit-processor`；`0`=不允许思考，`50`/`200` 为上限 |
| `separate_reasoning` | 只需 `--reasoning-parser`，用于拿到思考过程 |

`custom_logit_processor` 必须传 `to_str()` 产出的**序列化 JSON 串**，传类名会让
服务端 `orjson.loads` 抛错并导致 scheduler 崩溃（SIGQUIT）。工作流会在发送前校验，
不合法立即报错退出，不会波及服务端。

## 参考

- 仓库：<https://github.com/OpenMOSS/MOSS-Music>
- 数据流水线：<https://github.com/wx9Songs/MOSS-Music-Data-Pipeline>
- Issue #1（Transformers 推理复读）：<https://github.com/OpenMOSS/MOSS-Music/issues/1>
