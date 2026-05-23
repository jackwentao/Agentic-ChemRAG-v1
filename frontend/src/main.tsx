import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileUp, Send } from "lucide-react";
import "./styles.css";

const API_BASE = "http://localhost:8000";

type Citation = {
  document_id: string;
  chunk_id: string;
  page_start: number;
  page_end: number;
  source_url: string;
  section_title: string;
  quote: string;
  score: number;
  image_ids: string[];
  images: CitationImage[];
};

type CitationImage = {
  image_id: string;
  url: string;
  source_url: string;
  page: number;
  width: number | null;
  height: number | null;
  label: string | null;
  caption: string | null;
  reason: string | null;
};

type Task = {
  task_id: string | null;
  status: string;
  result?: number;
  error?: string;
};

type BatchUploadItem = {
  filename: string;
  document_id: string | null;
  task_id: string | null;
  status: string;
  reused: boolean;
  error: string | null;
};

type BatchUploadResponse = {
  results: BatchUploadItem[];
};

type DocumentSummary = {
  document_id: string;
  filename: string;
  created_at: string;
  chunk_count: number;
  is_indexed: boolean;
  indexing_status: "pending" | "running" | "failed" | "succeeded" | string;
  indexing_error: string | null;
};

type ChatScope = "current_document" | "all_documents";

function App() {
  const [taskId, setTaskId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [scope, setScope] = useState<ChatScope>("current_document");
  const [task, setTask] = useState<Task | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const selectedDocument = useMemo(
    () => documents.find((document) => document.document_id === documentId),
    [documents, documentId]
  );
  const selectedDocumentStatus = selectedDocument ? getDocumentStatus(selectedDocument) : "";
  const selectedCitationImages = useMemo(() => collectCitationImages(citations), [citations]);

  useEffect(() => {
    void refreshDocuments();
  }, []);

  async function refreshDocuments(preferredDocumentId?: string) {
    const response = await fetch(`${API_BASE}/api/documents`);
    const data: DocumentSummary[] = await response.json();
    setDocuments(data);

    const preferred = preferredDocumentId
      ? data.find((document) => document.document_id === preferredDocumentId)
      : undefined;
    const current = documentId ? data.find((document) => document.document_id === documentId) : undefined;
    const firstIndexed = data.find((document) => document.is_indexed);
    const next = preferred ?? current ?? firstIndexed ?? data[0];
    if (next) setDocumentId(next.document_id);
  }

  async function upload(files: FileList) {
    const selectedFiles = Array.from(files);
    if (selectedFiles.length === 0) return;
    setNotice("");
    const form = new FormData();
    selectedFiles.forEach((file) => form.append("files", file));
    const response = await fetch(`${API_BASE}/api/documents/batch`, { method: "POST", body: form });
    const data: BatchUploadResponse = await response.json();
    if (!response.ok) {
      setNotice((data as { detail?: string }).detail ?? "Upload failed.");
      return;
    }

    const firstDocument = data.results.find((result) => result.document_id);
    const firstTask = data.results.find((result) => result.task_id);
    setTaskId(firstTask?.task_id ?? "");
    setTask(firstTask ? { task_id: firstTask.task_id, status: firstTask.status } : null);
    if (firstDocument?.document_id) {
      setDocumentId(firstDocument.document_id);
      void refreshDocuments(firstDocument.document_id);
    }
    data.results.forEach((result) => {
      if (result.task_id) void pollTask(result.task_id, result.document_id ?? undefined);
    });
    setNotice(summarizeUpload(data.results));
  }

  async function pollTask(id: string, indexedDocumentId?: string) {
    try {
      const response = await fetch(`${API_BASE}/api/tasks/${id}`);
      const data = await response.json();
      if (!response.ok) {
        setNotice(data.detail ?? "Task status is unavailable. Refreshing document states.");
        void refreshDocuments(indexedDocumentId);
        return;
      }
      setTask(data);
      if (["pending", "started", "running"].includes(data.status)) {
        window.setTimeout(() => void pollTask(id, indexedDocumentId), 1500);
      }
      if (["success", "failure", "failed"].includes(data.status)) {
        void refreshDocuments(indexedDocumentId);
      }
    } catch {
      setNotice("Task status is unavailable. Refreshing document states.");
      void refreshDocuments(indexedDocumentId);
    }
  }

  async function ask() {
    if (!question.trim()) return;
    if (scope === "current_document" && !selectedDocument) {
      setNotice("Select a document first.");
      return;
    }
    if (scope === "current_document" && selectedDocument && !selectedDocument.is_indexed) {
      if (selectedDocument.indexing_status === "failed") {
        setNotice(selectedDocument.indexing_error ?? "Document indexing failed.");
      } else {
        setNotice("This document is still indexing.");
      }
      return;
    }
    if (scope === "all_documents" && !documents.some((document) => document.is_indexed)) {
      setNotice("No indexed documents are available yet.");
      return;
    }

    setBusy(true);
    setNotice("正在检索文献并生成回答...");
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question,
          scope,
          document_ids: scope === "current_document" && documentId ? [documentId] : undefined
        })
      });
      const data = await response.json();
      if (!response.ok) {
        setNotice(data.detail ?? "Question failed.");
        return;
      }
      setAnswer(data.answer);
      setCitations(data.citations ?? []);
      setNotice("");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setNotice("This question took too long. Try a narrower scope or ask about a specific figure, method, or result.");
      } else {
        setNotice("Question failed. Please try again.");
      }
    } finally {
      window.clearTimeout(timeoutId);
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="toolbar">
        <label className="upload-button">
          <FileUp size={18} />
          <span>Upload PDFs</span>
          <input type="file" accept="application/pdf" multiple onChange={(event) => {
            const files = event.target.files;
            if (files) void upload(files);
            event.currentTarget.value = "";
          }} />
        </label>
        {taskId && <span className="task">Task {taskId.slice(0, 8)} · {task?.status ?? "pending"}</span>}
        {documentId && <span className="task">Doc {documentId.slice(0, 8)}</span>}
      </section>

      <section className="workspace">
        <aside className="library">
          <header>
            <h2>My documents</h2>
            <button onClick={() => void refreshDocuments()}>Refresh</button>
          </header>
          {documents.length === 0 ? (
            <p className="empty">Upload a PDF to build your library.</p>
          ) : (
            <div className="document-list">
              {documents.map((document) => (
                <button
                  className={document.document_id === documentId ? "document active" : "document"}
                  key={document.document_id}
                  onClick={() => setDocumentId(document.document_id)}
                >
                  <span>{document.filename}</span>
                  <small className={`document-status ${document.indexing_status}`}>
                    {getDocumentStatus(document)}
                  </small>
                </button>
              ))}
            </div>
          )}
        </aside>

        <section className="chat">
          <div className="scope-tabs">
            <button className={scope === "current_document" ? "active" : ""} onClick={() => setScope("current_document")}>Current</button>
            <button className={scope === "all_documents" ? "active" : ""} onClick={() => setScope("all_documents")}>All docs</button>
          </div>
          {selectedDocumentStatus && (
            <div className={`selected-document ${selectedDocument?.indexing_status ?? ""}`}>
              <strong>{selectedDocument?.filename}</strong>
              <span>{selectedDocumentStatus}</span>
            </div>
          )}
          {notice && <div className="notice">{notice}</div>}
          <div className="messages">
            {answer ? <article className="answer">{answer}</article> : <article className="placeholder">Ask a question after indexing a paper.</article>}
            {selectedCitationImages.length > 0 && (
              <div className="answer-images">
                {selectedCitationImages.map((image) => (
                  <a
                    className="citation-image"
                    href={`${API_BASE}${image.source_url}`}
                    key={image.image_id}
                    rel="noreferrer"
                    target="_blank"
                    title={imageTitle(image)}
                  >
                    <img alt={`Related figure from page ${image.page}`} src={`${API_BASE}${image.url}`} />
                    <span>p. {image.page}</span>
                  </a>
                ))}
              </div>
            )}
            {citations.map((citation) => (
              <article className="citation" key={citation.chunk_id}>
                <header>
                  <a className="citation-source" href={`${API_BASE}${citation.source_url}`} rel="noreferrer" target="_blank">
                    <strong>{citation.section_title}</strong>
                    <span>p. {citation.page_start}-{citation.page_end}</span>
                  </a>
                </header>
                <p>{citation.quote}</p>
                {(citation.images ?? []).length > 0 && (
                  <div className="citation-images">
                    {(citation.images ?? []).map((image) => (
                      <a
                        className="citation-image"
                        href={`${API_BASE}${image.source_url}`}
                        key={image.image_id}
                        rel="noreferrer"
                        target="_blank"
                        title={imageTitle(image)}
                      >
                        <img alt={`Related figure from page ${image.page}`} src={`${API_BASE}${image.url}`} />
                        <span>p. {image.page}</span>
                      </a>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
          <div className="composer">
            <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about mechanisms, methods, or results..." />
            <button onClick={ask} disabled={busy}>
              <Send size={18} />
            </button>
          </div>
        </section>
      </section>
    </main>
  );
}

function getDocumentStatus(document: DocumentSummary) {
  if (document.is_indexed || document.indexing_status === "succeeded") {
    return `${document.chunk_count} chunks`;
  }
  if (document.indexing_status === "failed") {
    return "failed";
  }
  return "indexing";
}

function summarizeUpload(results: BatchUploadItem[]) {
  const started = results.filter((result) => result.task_id).length;
  const reused = results.filter((result) => result.reused && !result.task_id).length;
  const failed = results.filter((result) => result.error).length;
  const parts = [`${results.length} file${results.length > 1 ? "s" : ""}`];
  if (started) parts.push(`${started} indexing`);
  if (reused) parts.push(`${reused} reused`);
  if (failed) parts.push(`${failed} failed`);
  return `Upload batch: ${parts.join(", ")}.`;
}

function collectCitationImages(citations: Citation[]) {
  const seen = new Set<string>();
  const images: CitationImage[] = [];
  for (const citation of citations) {
    for (const image of citation.images ?? []) {
      if (seen.has(image.image_id)) continue;
      seen.add(image.image_id);
      images.push(image);
    }
  }
  return images;
}

function imageTitle(image: CitationImage) {
  const caption = [image.label, image.caption].filter(Boolean).join(" ");
  return caption || image.reason || `Image from page ${image.page}`;
}

createRoot(document.getElementById("root")!).render(<App />);
