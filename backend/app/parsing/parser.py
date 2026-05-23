from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re

from backend.app.domain.documents import ParsedDocument, ParsedImage, ParsedPage


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, document_id: str, file_path: Path) -> ParsedDocument:
        raise NotImplementedError


@dataclass(frozen=True)
class CaptionCandidate:
    label: str
    text: str
    bbox: tuple[float, float, float, float] | None = None


class PyMuPDFParser(DocumentParser):
    def __init__(self, max_rendered_caption_crops_per_page: int = 2) -> None:
        self.max_rendered_caption_crops_per_page = max_rendered_caption_crops_per_page

    def parse(self, document_id: str, file_path: Path) -> ParsedDocument:
        import fitz

        pages: list[ParsedPage] = []
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = self._clean_text(page.get_text("text") or "")
                captions = self._extract_caption_candidates(page.get_text("dict") or {}, text)
                image_contexts = self._extract_image_contexts(text, len(page.get_images(full=True)), captions)
                images: list[ParsedImage] = []
                used_caption_indexes: set[int] = set()
                kept_image_index = 0

                for image_index, image in enumerate(page.get_images(full=True), start=1):
                    xref = image[0]
                    extracted = pdf.extract_image(xref)
                    width = extracted.get("width")
                    height = extracted.get("height")
                    if self._is_tiny_image(width, height, len(extracted["image"])):
                        continue

                    rect = self._first_image_rect(page, xref)
                    kept_image_index += 1
                    caption_index = self._nearest_caption_index(rect, captions, used_caption_indexes)
                    if caption_index is not None:
                        used_caption_indexes.add(caption_index)
                        caption = captions[caption_index]
                        context = {"label": caption.label, "caption": caption.text, "nearby_text": None}
                    else:
                        context = image_contexts[kept_image_index - 1] if kept_image_index <= len(image_contexts) else {}

                    images.append(
                        ParsedImage(
                            page=page_index,
                            index=image_index,
                            extension=extracted.get("ext", "png"),
                            bytes_data=extracted["image"],
                            width=width,
                            height=height,
                            label=context.get("label"),
                            caption=context.get("caption"),
                            nearby_text=context.get("nearby_text"),
                            bbox=rect,
                            source_type="embedded",
                        )
                    )

                images.extend(
                    self._render_unmatched_caption_figures(
                        page=page,
                        page_number=page_index,
                        captions=captions,
                        used_caption_indexes=used_caption_indexes,
                        next_index=len(images) + 1,
                        fallback_text=text,
                    )
                )
                pages.append(ParsedPage(page=page_index, text=text, images=images))
        return ParsedDocument(document_id=document_id, filename=file_path.name, pages=pages)

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x02", " ").replace("\x03", " ")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        return re.sub(r"[ \t]+", " ", text)

    def _is_tiny_image(self, width: int | None, height: int | None, byte_size: int) -> bool:
        if width is None or height is None:
            return byte_size < 5_000
        return width <= 32 or height <= 32 or byte_size < 5_000

    def _extract_image_contexts(
        self,
        text: str,
        image_count: int,
        captions: list[CaptionCandidate] | None = None,
    ) -> list[dict[str, str | None]]:
        if image_count <= 0:
            return []
        caption_candidates = captions or [
            CaptionCandidate(label=self._caption_label(caption) or "", text=caption)
            for caption in self._extract_captions(text)
        ]
        nearby_text = self._nearby_text(text)
        contexts: list[dict[str, str | None]] = []
        for index in range(image_count):
            caption = caption_candidates[index] if index < len(caption_candidates) else None
            contexts.append(
                {
                    "label": caption.label if caption else None,
                    "caption": caption.text if caption else None,
                    "nearby_text": None if caption else nearby_text,
                }
            )
        return contexts

    def _extract_captions(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines()]
        captions: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line or not self._is_caption_start(line):
                index += 1
                continue

            caption_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index].strip()
                if not next_line or self._is_caption_start(next_line) or self._looks_like_section_heading(next_line):
                    break
                caption_lines.append(next_line)
                if self._caption_is_long_enough(caption_lines):
                    break
                index += 1
            captions.append(self._compact_text(" ".join(caption_lines))[:800])
        return captions

    def _is_caption_start(self, line: str) -> bool:
        return self._caption_label(line) is not None

    def _caption_label(self, line: str) -> str | None:
        match = re.match(
            r"^\s*((?:fig(?:ure)?|scheme|table)\.?\s*[\dIVXLC]+[A-Za-z]?(?:\.\d+)?|[图表]\s*[\d一二三四五六七八九十百]+)\s*[:：.\-]?",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return self._compact_text(match.group(1))

    def _looks_like_section_heading(self, line: str) -> bool:
        if len(line) > 80 or line.endswith("."):
            return False
        alpha = [char for char in line if char.isalpha()]
        if not alpha:
            return False
        uppercase_ratio = sum(char.isupper() for char in alpha) / len(alpha)
        return uppercase_ratio > 0.7 or line.istitle()

    def _caption_is_long_enough(self, caption_lines: list[str]) -> bool:
        text = " ".join(caption_lines)
        return len(text) >= 220 or text.count(".") >= 2 or text.count("。") >= 2

    def _nearby_text(self, text: str) -> str | None:
        compacted = self._compact_text(text)
        return compacted[:800] if compacted else None

    def _compact_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _extract_caption_candidates(self, text_dict: dict, fallback_text: str) -> list[CaptionCandidate]:
        lines = self._dict_lines(text_dict)
        if not lines:
            return [
                CaptionCandidate(label=self._caption_label(caption) or "", text=caption)
                for caption in self._extract_captions(fallback_text)
            ]

        captions: list[CaptionCandidate] = []
        index = 0
        while index < len(lines):
            line_text, line_bbox = lines[index]
            label = self._caption_label(line_text)
            if label is None:
                index += 1
                continue

            caption_lines = [line_text]
            caption_bbox = line_bbox
            index += 1
            while index < len(lines):
                next_text, next_bbox = lines[index]
                if not next_text or self._caption_label(next_text) is not None or self._looks_like_section_heading(next_text):
                    break
                caption_lines.append(next_text)
                caption_bbox = self._union_bbox(caption_bbox, next_bbox)
                if self._caption_is_long_enough(caption_lines):
                    break
                index += 1
            captions.append(
                CaptionCandidate(
                    label=label,
                    text=self._compact_text(" ".join(caption_lines))[:800],
                    bbox=caption_bbox,
                )
            )
        return captions

    def _dict_lines(self, text_dict: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
        lines: list[tuple[str, tuple[float, float, float, float]]] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = self._compact_text("".join(span.get("text", "") for span in line.get("spans", [])))
                if not text:
                    continue
                raw_bbox = line.get("bbox", block.get("bbox", (0, 0, 0, 0)))
                bbox = tuple(float(value) for value in raw_bbox)
                lines.append((text, bbox))  # type: ignore[arg-type]
        lines.sort(key=lambda item: (item[1][1], item[1][0]))
        return lines

    def _first_image_rect(self, page, xref: int) -> tuple[float, float, float, float] | None:
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            return None
        if not rects:
            return None
        rect = rects[0]
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

    def _nearest_caption_index(
        self,
        image_bbox: tuple[float, float, float, float] | None,
        captions: list[CaptionCandidate],
        used_caption_indexes: set[int],
    ) -> int | None:
        if not captions:
            return None
        if image_bbox is None:
            for index, caption in enumerate(captions):
                if index not in used_caption_indexes and caption.bbox is not None:
                    return index
            return None

        scored: list[tuple[float, int]] = []
        image_center_x = (image_bbox[0] + image_bbox[2]) / 2
        image_bottom = image_bbox[3]
        for index, caption in enumerate(captions):
            if index in used_caption_indexes or caption.bbox is None:
                continue
            caption_center_x = (caption.bbox[0] + caption.bbox[2]) / 2
            vertical_gap = max(0.0, caption.bbox[1] - image_bottom)
            if caption.bbox[1] < image_bbox[1] - 80:
                vertical_gap += 300
            horizontal_gap = abs(caption_center_x - image_center_x) * 0.15
            scored.append((vertical_gap + horizontal_gap, index))
        if not scored:
            return None
        score, index = min(scored)
        return index if score < 450 else None

    def _render_unmatched_caption_figures(
        self,
        page,
        page_number: int,
        captions: list[CaptionCandidate],
        used_caption_indexes: set[int],
        next_index: int,
        fallback_text: str,
    ) -> list[ParsedImage]:
        if not captions:
            return []
        rendered: list[ParsedImage] = []
        for caption_index, caption in enumerate(captions):
            if len(rendered) >= self.max_rendered_caption_crops_per_page:
                break
            if caption_index in used_caption_indexes or caption.bbox is None:
                continue
            clip = self._caption_crop(page, caption.bbox)
            if clip is None:
                continue
            try:
                pixmap = page.get_pixmap(clip=clip, matrix=self._render_matrix(), alpha=False)
                bytes_data = pixmap.tobytes("png")
            except Exception:
                continue
            if self._is_tiny_image(pixmap.width, pixmap.height, len(bytes_data)):
                continue
            rendered.append(
                ParsedImage(
                    page=page_number,
                    index=next_index + len(rendered),
                    extension="png",
                    bytes_data=bytes_data,
                    width=pixmap.width,
                    height=pixmap.height,
                    label=caption.label,
                    caption=caption.text,
                    nearby_text=None if caption.text else self._nearby_text(fallback_text),
                    bbox=(float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)),
                    source_type="page_crop",
                )
            )
        return rendered

    def _caption_crop(self, page, caption_bbox: tuple[float, float, float, float]):
        import fitz

        page_rect = page.rect
        top = max(float(page_rect.y0), caption_bbox[1] - 260)
        bottom = max(top + 20, caption_bbox[1] - 4)
        if bottom <= top:
            return None
        return fitz.Rect(float(page_rect.x0), top, float(page_rect.x1), bottom)

    def _render_matrix(self):
        import fitz

        return fitz.Matrix(2, 2)

    def _union_bbox(
        self,
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )
