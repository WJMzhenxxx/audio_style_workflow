"""单曲探测：对 sglang 发一次分析请求，原样打印响应，用于排查「输出与歌曲无关」。

不依赖本项目其它模块，只用标准库。

用法::

    python examples/probe_sglang.py "music/Aphonia.mp3" \
        --base-url http://10.0.0.5:43449 \
        --base64 \
        --max-tokens 2048

    # 换一个极简 prompt，判断是 prompt 问题还是音频没进去
    python examples/probe_sglang.py "music/Aphonia.mp3" --prompt "用一句话描述这段音频。"

输出包含：请求体大小、HTTP 状态、finish_reason、content、reasoning_content 预览，
以及原始 JSON（截断）。若 content 为 null 且 finish_reason=length，说明思考没结束；
若两首歌文本完全一致，说明音频没被服务端消费。
"""
from __future__ import annotations

import argparse
import atexit
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_MIME = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".tta": "audio/x-tta",
}

DEFAULT_PROMPT = (
    "请从整体氛围、乐器音色、结构发展、调性色彩和情绪表达几个方面描述这段音乐，"
    "总长度控制在 400 字以内。\n"
    "请严格使用下面的小标题（保留 ** 号），每个小标题占一行：\n"
    "**Genre**: <英文流派标签，逗号分隔，1-3 个，最具体的放最前>\n"
    "**Why**: <一到两句说明为什么判定为这些风格>\n"
    "**Tempo**: <BPM 整数 + 慢/中/快>\n"
    "**Harmony & Key**: <调性，例如 G major>\n"
    "**Mood & Theme**: <必须给 2-3 个简短的中文情绪词，用逗号分隔>\n"
    "**Score**: <0-10 整数评分>\n"
    "**Review**: <用中文写一到两句总体评价，不超过 60 字>\n"
    "**Instrument**: <主要乐器与音色>\n"
    "**Vocals & Lyrics**: <人声特征；纯器乐请注明>\n"
    "**Song Structure & Dynamics**: <用 3-5 句话概括；不要逐段罗列时间戳>"
)


def _audio_field(path: Path, base64_mode: bool, base_url: str, path_map: str | None) -> str:
    if base64_mode:
        mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    resolved = str(path.resolve())
    if path_map:
        local, _, remote = path_map.partition("=")
        if resolved.startswith(local):
            rest = resolved[len(local):].lstrip("\\/").replace("\\", "/")
            return remote.rstrip("/") + "/" + rest
    return resolved


def _clip(path: Path, seconds: int, mode: str):
    """用 ffmpeg 截取一段 16k 单声道 WAV，便于对比 prompt_tokens。"""
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    out_dir = Path(tempfile.mkdtemp(prefix="probe_clip_"))
    # 进程退出时删掉临时目录，避免在仓库里留下 probe_clip_*
    atexit.register(shutil.rmtree, out_dir, True)
    out = out_dir / "clip.wav"
    start = "0"
    if mode == "middle":
        # 近似中段：用 ffprobe 拿时长
        try:
            dur = subprocess.run(
                [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error",
                 "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True,
            ).stdout.strip()
            total = float(dur)
            if total > seconds:
                start = f"{max(0.0, (total - seconds) / 2):.3f}"
        except Exception:
            start = "0"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", start, "-t", str(seconds), "-i", str(path),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        return None
    return out if out.is_file() else None


def build_payload(args, audio_field: str) -> dict:
    if args.api == "generate":
        return {
            "text": args.prompt,
            "audio_data": audio_field,
            "sampling_params": {"max_new_tokens": args.max_tokens, "temperature": 0.0},
        }
    return {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_field}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "separate_reasoning": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sglang 单曲探测")
    parser.add_argument("audio", type=Path, help="音频文件")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--api", choices=["chat", "generate"], default="chat")
    parser.add_argument("--model", default="default")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--base64", action="store_true", help="以 data URL 内联音频")
    parser.add_argument("--path-map", default=None, help="LOCAL=REMOTE，路径模式下使用")
    parser.add_argument("--clip", type=int, default=0, metavar="SECONDS",
                        help="只送这么多秒（0=整首）；用来看 prompt_tokens 是否随长度线性变化")
    parser.add_argument("--clip-mode", choices=["middle", "start"], default="middle")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--head", type=int, default=3000, help="原始 JSON 打印长度")
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"[错误] 音频不存在: {args.audio}", file=sys.stderr)
        return 2

    source = args.audio
    if args.clip > 0:
        source = _clip(args.audio, args.clip, args.clip_mode)
        if source is None:
            print("[错误] ffmpeg 截取失败", file=sys.stderr)
            return 2
        print(f"截取      : {args.clip}s ({args.clip_mode}) -> {source}")

    audio_field = _audio_field(source, args.base64, args.base_url, args.path_map)
    if audio_field.startswith("data:"):
        head, _, body = audio_field.partition(",")
        shown = f"{head},<{len(body)} base64 chars>"
    else:
        shown = audio_field

    url = args.base_url.rstrip("/") + ("/generate" if args.api == "generate" else "/v1/chat/completions")
    body = build_payload(args, audio_field)
    raw = json.dumps(body).encode("utf-8")

    print(f"文件      : {args.audio}  ({args.audio.stat().st_size/1e6:.2f} MB)")
    print(f"音频字段  : {shown}")
    print(f"请求      : POST {url}  ({len(raw)/1e6:.2f} MB)")
    print(f"max_tokens: {args.max_tokens}")
    print()

    req = urllib.request.Request(
        url, data=raw, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        text = exc.read().decode("utf-8", "replace")
    except Exception as exc:  # 网络/超时
        print(f"[请求失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"HTTP {status}，响应 {len(text)} 字节")
    try:
        data = json.loads(text)
    except ValueError:
        print("[响应不是 JSON]")
        print(text[: args.head])
        return 1

    if args.api == "generate":
        model_text = data.get("text")
        meta = data.get("meta_info") or {}
        print(f"prompt_tokens: {meta.get('prompt_tokens')}  (音频是否线性摄入看这里)")
        print(f"text      : {model_text!r}"[:600])
    else:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        print(f"finish_reason: {finish}")
        print(f"usage        : {usage}")
        print(f"content      : {content!r}"[:600])
        print(f"reasoning    : {(reasoning or '')[:400]!r}")
        print()
        if isinstance(prompt_tokens, int) and prompt_tokens < 200:
            print(">> 判断：prompt_tokens 只有 %d，音频没进入模型——服务端只是在续写 prompt 文本。" % prompt_tokens)
            print("   试试 --api generate（audio_data 走另一条解析路径），或确认服务端模型是多模态的。")
        elif content in (None, "") and finish == "length":
            print(">> 判断：思考把 max_tokens 用完了（content 未产出）。加大 --max-tokens 或启用 thinking budget。")
        elif content in (None, ""):
            print(">> 判断：content 为空但未截断，可能只写了 reasoning_content。")
        else:
            print(">> 判断：拿到了正文。若多首歌文本完全一致，说明音频没被消费。")

    print()
    print("=== 原始响应（截断）===")
    print(text[: args.head])
    return 0


if __name__ == "__main__":
    sys.exit(main())
