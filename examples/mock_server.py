"""演示用的 HTTP 歌曲分析端点（FastAPI + multipart 上传）。

用于本 WorkFlow 的联调测试：

    # 启动（项目根目录下）
    python -m uvicorn examples.mock_server:app --host 127.0.0.1 --port 8010

然后用工作流指向它::

    python -m audio_style_workflow.main <目录> -e http://127.0.0.1:8010/analyze

端点契约（POST /analyze）::

    请求: multipart/form-data
          file  = 音频文件（mp3/flac/tta...）
          raw   = "json"(默认) 或 "freeform"  自由选择响应形态
    响应: JSON：
          {"status": "ok", "styles": [...], "confidence": 0.xx}
          freeform 模式返回非结构化 {"caption": "..."}，用于测 LLM 兜底。
"""
from __future__ import annotations

import hashlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="Mock Song Analyzer", version="1.0")

STYLE_POOL = [
    "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop",
    "R&B", "Folk", "Metal", "Blues", "Ambient", "City Pop",
]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "mock-song-analyzer"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    raw: str = Form("json"),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")

    digest = hashlib.md5(data[:65536] + (file.filename or "").encode("utf-8", "replace")).hexdigest()
    picks: list[str] = []
    for i in range(3):
        style = STYLE_POOL[int(digest[i * 4:(i + 1) * 4], 16) % len(STYLE_POOL)]
        picks.append(style)
    styles = list(dict.fromkeys(picks))

    if raw == "freeform":
        return {
            "caption": f"听起来这是 {styles[0]} 和 {styles[-1]} 的跨界融合，略带 {STYLE_POOL[int(digest[:2], 16) % len(STYLE_POOL)]} 的影子。",
        }
    return {
        "status": "ok",
        "styles": styles,
        "confidence": 0.87 + int(digest[-2:], 16) % 12 / 100,
        "echo": {"filename": file.filename, "bytes": len(data), "content_type": file.content_type},
    }
