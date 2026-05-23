from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[1]
    checks = {
        "数据库连接": settings.database_url,
        "Redis 连接": settings.redis_url,
        "MinIO 端点": settings.minio_endpoint,
        "对象存储提供器": settings.object_storage_provider,
        "DeepSeek API Key 已配置": bool(settings.deepseek_api_key and settings.deepseek_api_key != "replace-me"),
        "DeepSeek 模型": settings.deepseek_model,
        "向量模型": settings.embedding_model,
        "向量模型允许首次下载": not settings.embedding_local_files_only,
        "重排模型允许首次下载": not settings.reranker_local_files_only,
        "上传目录": str(settings.upload_dir.resolve()),
        "图片目录": str(settings.image_dir.resolve()),
        "模型缓存目录": str(settings.model_cache_dir.resolve()),
        "项目目录": str(project_root),
    }
    for key, value in checks.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
