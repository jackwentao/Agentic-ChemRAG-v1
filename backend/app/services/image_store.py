from pathlib import Path
from uuid import uuid4

from backend.app.domain.documents import ParsedDocument, StoredImage
from backend.app.services.object_storage import ObjectStorage, image_object_key


class LocalImageStore:
    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def save_images(self, parsed: ParsedDocument) -> list[StoredImage]:
        stored: list[StoredImage] = []
        for page in parsed.pages:
            for image in page.images:
                image_id = uuid4().hex
                filename = f"{image_id}.{image.extension}"
                path = self.image_dir / filename
                path.write_bytes(image.bytes_data)
                stored.append(
                    StoredImage(
                        image_id=image_id,
                        document_id=parsed.document_id,
                        page=image.page,
                        path=str(path),
                        extension=image.extension,
                        width=image.width,
                        height=image.height,
                        label=image.label,
                        caption=image.caption,
                        nearby_text=image.nearby_text,
                        bbox=image.bbox,
                        source_type=image.source_type,
                    )
                )
        return stored


class ObjectImageStore:
    def __init__(self, object_storage: ObjectStorage) -> None:
        self.object_storage = object_storage

    def save_images(self, parsed: ParsedDocument) -> list[StoredImage]:
        stored: list[StoredImage] = []
        for page in parsed.pages:
            for image in page.images:
                image_id = uuid4().hex
                extension = image.extension or "png"
                object_key = image_object_key(image_id, extension)
                self.object_storage.put_bytes(object_key, image.bytes_data, content_type=f"image/{extension}")
                stored.append(
                    StoredImage(
                        image_id=image_id,
                        document_id=parsed.document_id,
                        page=image.page,
                        path=object_key,
                        extension=extension,
                        width=image.width,
                        height=image.height,
                        label=image.label,
                        caption=image.caption,
                        nearby_text=image.nearby_text,
                        bbox=image.bbox,
                        source_type=image.source_type,
                    )
                )
        return stored
