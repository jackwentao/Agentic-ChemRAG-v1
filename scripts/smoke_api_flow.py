import argparse
from pathlib import Path
import time
from typing import Callable

import httpx


def wait_for_indexing(
    client: httpx.Client,
    document_id: str,
    task_id: str | None,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if task_id:
            task_response = client.get(f"/api/tasks/{task_id}")
            task_response.raise_for_status()
            task = task_response.json()
            if task["status"] in {"failure", "failed"}:
                raise RuntimeError(f"索引任务失败：{task.get('error') or '未知错误'}")

        documents_response = client.get("/api/documents")
        documents_response.raise_for_status()
        document = next(
            (item for item in documents_response.json() if item["document_id"] == document_id),
            None,
        )
        if document and (document["is_indexed"] or document["indexing_status"] == "succeeded"):
            return document
        if document and document["indexing_status"] == "failed":
            raise RuntimeError(f"文档索引失败：{document.get('indexing_error') or '未知错误'}")
        sleep(poll_interval_seconds)
    raise TimeoutError(f"等待文档 {document_id} 索引完成超时。")


def run_smoke_flow(
    client: httpx.Client,
    pdf_path: Path,
    question: str,
    timeout_seconds: float = 1800,
    poll_interval_seconds: float = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    with pdf_path.open("rb") as pdf_file:
        upload_response = client.post(
            "/api/documents",
            files={"file": (pdf_path.name, pdf_file, "application/pdf")},
        )
    upload_response.raise_for_status()
    upload = upload_response.json()
    document_id = upload["document_id"]
    document = wait_for_indexing(
        client,
        document_id,
        upload.get("task_id"),
        timeout_seconds,
        poll_interval_seconds,
        sleep,
    )
    chat_response = client.post(
        "/api/chat",
        json={
            "question": question,
            "scope": "current_document",
            "document_ids": [document_id],
        },
    )
    chat_response.raise_for_status()
    chat = chat_response.json()
    if not chat.get("answer"):
        raise RuntimeError("问答接口没有返回答案。")
    if not chat.get("citations"):
        raise RuntimeError("问答接口没有返回引用证据。")
    return {"upload": upload, "document": document, "chat": chat}


def main() -> None:
    parser = argparse.ArgumentParser(description="通过真实 API 验证 PDF 上传、异步索引与问答链路。")
    parser.add_argument("pdf", type=Path, help="用于验证的本地 PDF 文件路径。")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="API 服务地址。")
    parser.add_argument("--question", default="这篇论文的主要结论是什么？", help="用于验证的提问。")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="等待首次模型加载和索引的最长秒数。")
    args = parser.parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    with httpx.Client(base_url=args.api_base, timeout=120) as client:
        result = run_smoke_flow(client, args.pdf, args.question, timeout_seconds=args.timeout_seconds)
    upload = result["upload"]
    document = result["document"]
    chat = result["chat"]
    print(f"上传成功：document_id={upload['document_id']}，status={upload['status']}")
    print(f"索引完成：chunks={document['chunk_count']}")
    print("回答：")
    print(chat["answer"])
    print(f"引用数量：{len(chat['citations'])}")


if __name__ == "__main__":
    main()
