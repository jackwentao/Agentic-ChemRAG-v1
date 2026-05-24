from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.database import build_database_engine, prepare_database


def main() -> None:
    """为部署流水线显式执行与应用启动一致的数据库准备步骤。"""
    engine = build_database_engine(get_settings().database_url)
    try:
        prepare_database(engine)
        print("数据库准备完成。")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
