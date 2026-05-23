from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[1]
    checks = {
        "database_url": bool(settings.database_url),
        "redis_url": bool(settings.redis_url),
        "deepseek_api_key": bool(settings.deepseek_api_key),
        "deepseek_model": bool(settings.deepseek_model),
        "embedding_model": settings.embedding_model,
        "upload_dir": str(settings.upload_dir.resolve()),
        "image_dir": str(settings.image_dir.resolve()),
        "model_cache_dir": str(settings.model_cache_dir.resolve()),
        "project_root": str(project_root),
        "project_on_d_drive": project_root.drive.upper() == "D:",
    }
    for key, value in checks.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
