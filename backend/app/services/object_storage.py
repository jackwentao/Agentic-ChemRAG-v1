from abc import ABC, abstractmethod
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from urllib.parse import urlparse

from backend.app.config import Settings


class ObjectStorage(ABC):
    @abstractmethod
    def healthcheck(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        raise NotImplementedError

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        self.put_bytes(key, path.read_bytes(), content_type)

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        target = self._path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, target.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)

    def healthcheck(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def open(self, key: str) -> BinaryIO:
        return self._path_for(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def _path_for(self, key: str) -> Path:
        path = (self.root_dir / key).resolve()
        root = self.root_dir.resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"Object key escapes storage root: {key}")
        return path


class MinioObjectStorage(ObjectStorage):
    def __init__(self, settings: Settings) -> None:
        from minio import Minio
        from minio.error import S3Error

        self._s3_error = S3Error
        self.bucket = settings.minio_bucket
        endpoint_host = urlparse(f"http://{settings.minio_endpoint}").hostname
        if endpoint_host in {"localhost", "127.0.0.1"}:
            os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
            os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def healthcheck(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            raise RuntimeError(f"Object storage bucket is unavailable: {self.bucket}")

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        from io import BytesIO

        self.client.put_object(
            self.bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        with path.open("rb") as data:
            self.client.put_object(
                self.bucket,
                key,
                data,
                length=size,
                content_type=content_type,
            )

    def open(self, key: str) -> BinaryIO:
        response = self.client.get_object(self.bucket, key)
        temp = NamedTemporaryFile(delete=True)
        try:
            for chunk in response.stream(32 * 1024):
                temp.write(chunk)
            temp.seek(0)
            return temp
        finally:
            response.close()
            response.release_conn()

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except self._s3_error as exc:
            if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchBucket"}:
                return False
            raise


def document_pdf_key(document_id: str) -> str:
    return f"documents/{document_id}.pdf"


def image_object_key(image_id: str, extension: str) -> str:
    return f"images/{image_id}.{extension}"


def build_object_storage(settings: Settings) -> ObjectStorage:
    if settings.object_storage_provider.lower() == "local":
        return LocalObjectStorage(Path("storage/objects"))
    if settings.object_storage_provider.lower() == "minio":
        return MinioObjectStorage(settings)
    raise ValueError(f"Unsupported object storage provider: {settings.object_storage_provider}")
