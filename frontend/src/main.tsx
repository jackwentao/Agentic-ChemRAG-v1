import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileUp, Pin, PinOff, Send, Trash2 } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type Citation = {
  document_id: string;
  chunk_id: string;
  page_start: number;
  page_end: number;
  source_url: string | null;
  section_title: string;
  quote: string;
  score: number;
  evidence_type: string;
  source_scope: string;
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
  pinned: boolean;
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

type ServerSentEvent = {
  event: string;
  data: Record<string, unknown>;
};

type ChatStage = {
  stage: string;
  label: string;
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
  const [chatStage, setChatStage] = useState<ChatStage | null>(null);

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
    const preferred = preferredSessionId
      ? data.sessions.find((session) => session.session_id === preferredSessionId)
      : undefined;
    const current = sessionId ? data.sessions.find((session) => session.session_id === sessionId) : undefined;
    const nextSessionId = preferred?.session_id ?? current?.session_id ?? data.sessions[0]?.session_id ?? "";
    if (nextSessionId) {
      setSessionId(nextSessionId);
      if (options.loadMessages !== false) {
        await loadMessages(nextSessionId);
      }
    } else {
      setSessionId("");
      setMessages([]);
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
    setChatStage(null);
  }

  async function toggleSessionPinned(session: ChatSession) {
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions/${session.session_id}/pin`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned: !session.pinned })
      });
      const data = await readResponsePayload(response);
      if (!response.ok) {
        setNotice(apiErrorMessage(data, "Could not update this chat."));
        return;
      }
      await refreshSessions(sessionId || undefined, { loadMessages: false });
    } catch {
      setNotice("Could not update this chat. Please try again.");
    }
  }

  async function deleteSession(targetSessionId: string) {
    if (!window.confirm("Delete this chat history?")) return;
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions/${targetSessionId}`, { method: "DELETE" });
      if (!response.ok) {
        const data = await readResponsePayload(response);
        setNotice(apiErrorMessage(data, "Could not delete this chat."));
        return;
      }
      if (targetSessionId === sessionId) {
        setSessionId("");
        setMessages([]);
        setChatStage(null);
      }
      await refreshSessions(undefined, { loadMessages: targetSessionId !== sessionId });
    } catch {
      setNotice("Could not delete this chat. Please try again.");
    }
  }

  async function upload(files: FileList) {
    const selectedFiles = Array.from(files);
    if (selectedFiles.length === 0) return;
    setNotice("");
    const form = new FormData();
    selectedFiles.forEach((file) => form.append("files", file));
    let response: Response;
    let data: BatchUploadResponse | unknown;
    try {
      response = await fetch(`${API_BASE}/api/documents/batch`, { method: "POST", body: form });
      data = await readResponsePayload(response);
      if (!response.ok) {
        setNotice(apiErrorMessage(data, uploadHttpFallback(response)));
        return;
      }
    } catch {
      setNotice("Upload failed. Please check that the API is running and try again.");
      return;
    }
    if (!isBatchUploadResponse(data)) {
      setNotice("Upload failed. The server returned an unexpected response.");
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
      const data = await readResponsePayload(response);
      if (!response.ok) {
        setNotice(apiErrorMessage(data, "Task status is unavailable. Refreshing document states."));
        void refreshDocuments(indexedDocumentId);
        return;
      }
      const taskData = data as Task;
      setTask(taskData);
      if (["pending", "started", "running"].includes(taskData.status)) {
        window.setTimeout(() => void pollTask(id, indexedDocumentId), 1500);
      }
      if (["success", "failure", "failed"].includes(taskData.status)) {
        void refreshDocuments(indexedDocumentId);
      }
    } catch {
      setNotice("Task status is unavailable. Refreshing document states.");
      void refreshDocuments(indexedDocumentId);
    }
  }

  async function ask() {
    const submittedQuestion = question.trim();
    if (!submittedQuestion) return;
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

    const tempUserMessageId = localMessageId("user");
    const tempAssistantMessageId = localMessageId("assistant");
    const requestDocumentIds = scope === "current_document" && documentId ? [documentId] : [];
    setQuestion("");
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
      },
      {
        message_id: tempAssistantMessageId,
        session_id: sessionId || `local-session-${tempAssistantMessageId}`,
        role: "assistant",
        content: "",
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
    setNotice("");
    setChatStage({ stage: "planning", label: "正在规划检索路径" });
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 180000);
    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
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
      if (!response.ok || !response.body) {
        const data = (await readResponsePayload(response)) as ChatResponse & { detail?: string };
        markMessageFailed(tempUserMessageId);
        markMessageFailed(tempAssistantMessageId);
        setNotice(apiErrorMessage(data, "Question failed."));
        return;
      }
      let finalResponse: ChatResponse | null = null;
      for await (const event of readServerSentEvents(response.body)) {
        if (event.event === "stage") {
          const nextStage = parseChatStage(event.data);
          if (nextStage) setChatStage(nextStage);
        }
        if (event.event === "answer_delta") {
          const delta = typeof event.data.delta === "string" ? event.data.delta : "";
          if (!delta) continue;
          setChatStage((currentStage) =>
            currentStage?.stage === "answering" ? currentStage : { stage: "answering", label: "正在回答" }
          );
          setMessages((currentMessages) =>
            currentMessages.map((message) =>
              message.message_id === tempAssistantMessageId
                ? { ...message, content: `${message.content}${delta}` }
                : message
            )
          );
        }
        if (event.event === "error") {
          markMessageFailed(tempUserMessageId);
          markMessageFailed(tempAssistantMessageId);
          setChatStage(null);
          setNotice(apiErrorMessage(event.data, "Question failed."));
          return;
        }
        if (event.event === "final") {
          finalResponse = event.data as ChatResponse;
          setSessionId(finalResponse.session_id);
          setMessages((currentMessages) =>
            currentMessages.map((message) => {
              if (message.message_id === tempUserMessageId) return finalResponse!.user_message;
              if (message.message_id === tempAssistantMessageId) return finalResponse!.assistant_message;
              return message;
            })
          );
          void refreshSessions(finalResponse.session_id, { loadMessages: false });
          setChatStage(null);
          setNotice("");
        }
      }
      if (!finalResponse) {
        markMessageFailed(tempUserMessageId);
        markMessageFailed(tempAssistantMessageId);
        setChatStage(null);
        setNotice("Question failed. The server stream ended before sending a final answer.");
      }
    } catch (error) {
      markMessageFailed(tempUserMessageId);
      markMessageFailed(tempAssistantMessageId);
      setChatStage(null);
      if (error instanceof DOMException && error.name === "AbortError") {
        setNotice("This question took too long. Try a narrower scope or ask about a specific figure, method, or result.");
      } else {
        setNotice("Question failed. Please try again.");
      }
    } finally {
      window.clearTimeout(timeoutId);
      setBusy(false);
      setChatStage(null);
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
        </aside>

        <aside className="sessions">
          <header>
            <h2>Chats</h2>
            <button onClick={startNewChat}>New</button>
          </header>
          {sessions.length === 0 ? (
            <p className="empty">Ask a question to start a chat.</p>
          ) : (
            <div className="session-list">
              {sessions.map((session) => (
                <article className={session.session_id === sessionId ? "session active" : "session"} key={session.session_id}>
                  <button
                    className="session-main"
                    onClick={() => {
                      setSessionId(session.session_id);
                      void loadMessages(session.session_id);
                    }}
                  >
                    <span>{session.title}</span>
                    <small>{formatSessionTime(session.updated_at)}</small>
                  </button>
                  <div className="session-actions">
                    <button
                      aria-label={session.pinned ? "Unpin chat" : "Pin chat"}
                      onClick={() => void toggleSessionPinned(session)}
                      title={session.pinned ? "Unpin chat" : "Pin chat"}
                    >
                      {session.pinned ? <PinOff size={16} /> : <Pin size={16} />}
                    </button>
                    <button
                      aria-label="Delete chat"
                      onClick={() => void deleteSession(session.session_id)}
                      title="Delete chat"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </article>
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
          {busy && chatStage && (
            <div className={`stage-hint ${chatStage.stage}`} aria-live="polite">
              {chatStage.label}
            </div>
          )}
          <div className="messages">
            {messages.length === 0 ? (
              <article className="placeholder">Ask a question after indexing a paper.</article>
            ) : (
              messages.map((message) => <ChatMessageView key={message.message_id} message={message} />)
            )}
          </div>
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              if (!busy) void ask();
            }}
          >
            <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about mechanisms, methods, or results..." />
            <button type="submit" disabled={busy}>
              <Send size={18} />
            </button>
          </form>
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
      <div className="answer">{message.content || (message.status === "sending" ? "正在生成..." : "")}</div>
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
            {citation.source_url ? (
              <a className="citation-source" href={`${API_BASE}${citation.source_url}`} rel="noreferrer" target="_blank">
                <strong>{citation.section_title}</strong>
                <span>p. {citation.page_start}-{citation.page_end}</span>
              </a>
            ) : (
              <div className="citation-source">
                <strong>{citation.section_title}</strong>
                <span>{citation.source_scope === "document_level" ? "文档画像" : citation.evidence_type}</span>
              </div>
            )}
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

function apiErrorMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === "object") {
    const data = payload as { detail?: unknown; error?: { message?: unknown } };
    if (typeof data.error?.message === "string") return data.error.message;
    if (typeof data.detail === "string") return data.detail;
  }
  if (typeof payload === "string" && payload.trim()) return payload;
  return fallback;
}

async function readResponsePayload(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const text = await response.text();
  return text.trim();
}

async function* readServerSentEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<ServerSentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const event = parseServerSentEvent(frame);
        if (event) yield event;
      }
    }
    buffer += decoder.decode();
    const tail = parseServerSentEvent(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function parseServerSentEvent(frame: string): ServerSentEvent | null {
  const lines = frame.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith("event:"))?.slice("event:".length).trim() || "message";
  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart());
  if (dataLines.length === 0) return null;
  try {
    const data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    return { event, data };
  } catch {
    return null;
  }
}

function parseChatStage(payload: Record<string, unknown>): ChatStage | null {
  const stage = typeof payload.stage === "string" ? payload.stage : "";
  const label = typeof payload.label === "string" ? payload.label : "";
  if (!stage || !label) return null;
  return { stage, label };
}

function uploadHttpFallback(response: Response) {
  if (response.status === 413) {
    return "Upload failed. The PDF is larger than the server upload limit.";
  }
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    return "Upload failed. The API service is not available.";
  }
  return "Upload failed.";
}

function isBatchUploadResponse(payload: unknown): payload is BatchUploadResponse {
  return Boolean(
    payload &&
      typeof payload === "object" &&
      Array.isArray((payload as BatchUploadResponse).results)
  );
}

createRoot(document.getElementById("root")!).render(<App />);
