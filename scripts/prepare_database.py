from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alembic import command
from alembic.config import Config


def main() -> None:
    """执行数据库版本化迁移。"""
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    command.upgrade(config, "head")
    print("数据库迁移完成。")


if __name__ == "__main__":
    main()
