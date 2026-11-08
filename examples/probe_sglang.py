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
    "Listen to this track and analyze it. Reply in this exact format:\n"
    "Line 1: its musical styles/genres as a short comma-separated list.\n"
    "Line 2: BPM: <integer>\n"
    "Line 3: Tempo: <descriptor>\n"
    "Line 4: Key: <key signature>\n"
    "Line 5: Why: <why these styles fit>\n"
    "Then describe the genre, instruments and mood."
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
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--head", type=int, default=3000, help="原始 JSON 打印长度")
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"[错误] 音频不存在: {args.audio}", file=sys.stderr)
        return 2

    audio_field = _audio_field(args.audio, args.base64, args.base_url, args.path_map)
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
        print(f"text      : {model_text!r}"[:600])
    else:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        print(f"finish_reason: {finish}")
        print(f"content      : {content!r}"[:600])
        print(f"reasoning    : {(reasoning or '')[:400]!r}")
        print()
        if content in (None, "") and finish == "length":
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
