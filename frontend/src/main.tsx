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

type ChatSession = {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

type ChatMessage = {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  scope: string | null;
  document_ids: string[];
  citations: Citation[];
  images: CitationImage[];
  limitations: string[];
  diagnostics_summary?: Record<string, unknown>;
  created_at: string | null;
  status?: "sending" | "failed";
};

type ChatResponse = {
  session_id: string;
  answer: string;
  citations: Citation[];
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  limitations: string[];
  warnings: string[];
};

function App() {
  const [taskId, setTaskId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [scope, setScope] = useState<ChatScope>("current_document");
  const [task, setTask] = useState<Task | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const selectedDocument = useMemo(
    () => documents.find((document) => document.document_id === documentId),
    [documents, documentId]
  );
  const selectedDocumentStatus = selectedDocument ? getDocumentStatus(selectedDocument) : "";

  useEffect(() => {
    void refreshDocuments();
    void refreshSessions();
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

  async function refreshSessions(preferredSessionId?: string, options: { loadMessages?: boolean } = {}) {
    const response = await fetch(`${API_BASE}/api/chat/sessions`);
    const data: { sessions: ChatSession[] } = await response.json();
    setSessions(data.sessions);
    const nextSessionId = preferredSessionId ?? sessionId ?? data.sessions[0]?.session_id ?? "";
    if (nextSessionId) {
      setSessionId(nextSessionId);
      if (options.loadMessages !== false) {
        await loadMessages(nextSessionId);
      }
    }
  }

  async function loadMessages(nextSessionId: string) {
    const response = await fetch(`${API_BASE}/api/chat/sessions/${nextSessionId}/messages`);
    if (!response.ok) {
      setMessages([]);
      return;
    }
    const data: { messages: ChatMessage[] } = await response.json();
    setMessages(data.messages);
  }

  function startNewChat() {
    setSessionId("");
    setMessages([]);
    setNotice("");
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

    const submittedQuestion = question;
    const tempUserMessageId = localMessageId("user");
    const requestDocumentIds = scope === "current_document" && documentId ? [documentId] : [];
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        message_id: tempUserMessageId,
        session_id: sessionId || `local-session-${tempUserMessageId}`,
        role: "user",
        content: submittedQuestion,
        scope,
        document_ids: requestDocumentIds,
        citations: [],
        images: [],
        limitations: [],
        diagnostics_summary: {},
        created_at: new Date().toISOString(),
        status: "sending"
      }
    ]);
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
          question: submittedQuestion,
          session_id: sessionId || undefined,
          scope,
          document_ids: scope === "current_document" && documentId ? [documentId] : undefined
        })
      });
      const data: ChatResponse & { detail?: string } = await response.json();
      if (!response.ok) {
        markMessageFailed(tempUserMessageId);
        setNotice(data.detail ?? "Question failed.");
        return;
      }
      setSessionId(data.session_id);
      setQuestion("");
      setMessages((currentMessages) => [
        ...currentMessages.map((message) =>
          message.message_id === tempUserMessageId ? data.user_message : message
        ),
        data.assistant_message
      ]);
      await refreshSessions(data.session_id, { loadMessages: false });
      setNotice("");
    } catch (error) {
      markMessageFailed(tempUserMessageId);
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

  function markMessageFailed(messageId: string) {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.message_id === messageId ? { ...message, status: "failed" } : message
      )
    );
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

          <section className="sessions">
            <header>
              <h2>Chats</h2>
              <button onClick={startNewChat}>New</button>
            </header>
            {sessions.length === 0 ? (
              <p className="empty">Ask a question to start a chat.</p>
            ) : (
              <div className="session-list">
                {sessions.map((session) => (
                  <button
                    className={session.session_id === sessionId ? "session active" : "session"}
                    key={session.session_id}
                    onClick={() => {
                      setSessionId(session.session_id);
                      void loadMessages(session.session_id);
                    }}
                  >
                    <span>{session.title}</span>
                    <small>{formatSessionTime(session.updated_at)}</small>
                  </button>
                ))}
              </div>
            )}
          </section>
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
            {messages.length === 0 ? (
              <article className="placeholder">Ask a question after indexing a paper.</article>
            ) : (
              messages.map((message) => <ChatMessageView key={message.message_id} message={message} />)
            )}
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

function ChatMessageView({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <article className="message user-message">{message.content}</article>;
  }
  const selectedCitationImages = collectCitationImages(message.citations ?? []);
  return (
    <article className="message assistant-message">
      <div className="answer">{message.content}</div>
      {message.limitations.length > 0 && (
        <div className="limitations">
          <strong>资料限制</strong>
          <ul>
            {message.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </div>
      )}
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
      {message.citations.map((citation) => (
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
    </article>
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

function formatSessionTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function localMessageId(role: ChatMessage["role"]) {
  return `local-${role}-${crypto.randomUUID()}`;
}

createRoot(document.getElementById("root")!).render(<App />);
