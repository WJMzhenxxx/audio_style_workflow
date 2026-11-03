"""歌曲分析端点 API 包。

- :mod:`.base`              —— 抽象接口定义（先看这个）
- :mod:`.http_client`       —— 基于 HTTP multipart 上传的实现
- :mod:`.mock_client`       —— 进程内演示实现（无需网络）
- :mod:`.local_model_client`—— 本地 MusicFlamingo 模型推理实现（离线）
- :mod:`.sglang_client`     —— sglang 部署的音乐理解服务实现（音频传本地路径）
"""
from .base import SongAnalysisClient, SongAnalysisError, create_client

__all__ = [
    "SongAnalysisClient",
    "SongAnalysisError",
    "create_client",
]
