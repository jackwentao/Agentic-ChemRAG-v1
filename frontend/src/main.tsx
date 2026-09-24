import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  FileText,
  FileUp,
  FlaskConical,
  LayoutDashboard,
  Library,
  MessageSquare,
  MoreHorizontal,
  Pin,
  PinOff,
  RefreshCw,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
  XCircle
} from "lucide-react";
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
  const [activeView, setActiveView] = useState<"overview" | "workbench" | "library" | "insights">("workbench");
  const [mobilePanel, setMobilePanel] = useState<"documents" | "sessions" | null>(null);
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
  const [apiOnline, setApiOnline] = useState(true);

  const selectedDocument = useMemo(
    () => documents.find((document) => document.document_id === documentId),
    [documents, documentId]
  );
  const selectedDocumentStatus = selectedDocument ? getDocumentStatus(selectedDocument) : "";
  const indexedDocuments = documents.filter((document) => document.is_indexed);
  const indexingDocuments = documents.filter((document) => !document.is_indexed && document.indexing_status !== "failed");
  const failedDocuments = documents.filter((document) => document.indexing_status === "failed");

  useEffect(() => {
    void refreshDocuments();
    void refreshSessions();
  }, []);

  async function refreshDocuments(preferredDocumentId?: string) {
    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      const data: DocumentSummary[] = await response.json();
      setApiOnline(true);
      setDocuments(data);

      const preferred = preferredDocumentId
        ? data.find((document) => document.document_id === preferredDocumentId)
        : undefined;
      const current = documentId ? data.find((document) => document.document_id === documentId) : undefined;
      const firstIndexed = data.find((document) => document.is_indexed);
      const next = preferred ?? current ?? firstIndexed ?? data[0];
      if (next) setDocumentId(next.document_id);
    } catch {
      setApiOnline(false);
      setNotice("Local API is unavailable. You can still explore the workspace and connect it when the server is ready.");
    }
  }

  async function refreshSessions(preferredSessionId?: string, options: { loadMessages?: boolean } = {}) {
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions`);
      const data: { sessions: ChatSession[] } = await response.json();
      setApiOnline(true);
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
    } catch {
      setApiOnline(false);
      setSessions([]);
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

  function renderDocumentRail() {
    return (
      <aside className={`panel-rail documents-rail ${mobilePanel === "documents" ? "mobile-open" : ""}`}>
        <div className="rail-heading">
          <div>
            <span className="eyebrow">Knowledge base</span>
            <h2>Documents</h2>
          </div>
          <button className="icon-button" onClick={() => void refreshDocuments()} aria-label="Refresh documents" title="Refresh documents"><RefreshCw size={16} /></button>
        </div>
        <div className="rail-search"><Search size={15} /><input aria-label="Filter documents" placeholder="Filter documents" /></div>
        <div className="rail-summary"><span>{documents.length} sources</span><span className="summary-dot"><CheckCircle2 size={13} /> {indexedDocuments.length} ready</span></div>
        <div className="document-list">
          {documents.length === 0 ? <div className="empty-state compact"><FileText size={24} /><strong>Your library is empty</strong><span>Upload a paper to start asking questions.</span></div> : documents.map((document) => (
            <button className={document.document_id === documentId ? "document active" : "document"} key={document.document_id} onClick={() => { setDocumentId(document.document_id); setMobilePanel(null); }}>
              <span className="document-icon"><FileText size={16} /></span>
              <span className="document-copy"><strong>{document.filename}</strong><small>{document.is_indexed ? `${document.chunk_count} chunks` : getDocumentStatus(document)}</small></span>
              <span className={`status-dot ${document.indexing_status}`} />
            </button>
          ))}
        </div>
        <label className="upload-dropzone">
          <UploadCloud size={18} /><span>Upload papers</span><small>PDF · multiple files supported</small>
          <input type="file" accept="application/pdf" multiple onChange={(event) => { const files = event.target.files; if (files) void upload(files); event.currentTarget.value = ""; }} />
        </label>
      </aside>
    );
  }

  function renderSessionRail() {
    return (
      <aside className={`panel-rail sessions-rail ${mobilePanel === "sessions" ? "mobile-open" : ""}`}>
        <div className="rail-heading"><div><span className="eyebrow">Workspace</span><h2>Conversations</h2></div><button className="icon-button" onClick={startNewChat} aria-label="New conversation" title="New conversation"><Sparkles size={16} /></button></div>
        <button className="new-chat-button" onClick={startNewChat}><MessageSquare size={16} /> New conversation <span>⌘ N</span></button>
        <div className="session-list">
          {sessions.length === 0 ? <div className="empty-state compact"><MessageSquare size={24} /><strong>No conversations yet</strong><span>Ask your first question from the workspace.</span></div> : sessions.map((session) => (
            <article className={session.session_id === sessionId ? "session active" : "session"} key={session.session_id}>
              <button className="session-main" onClick={() => { setSessionId(session.session_id); void loadMessages(session.session_id); setMobilePanel(null); }}><span>{session.title}</span><small>{formatSessionTime(session.updated_at)}</small></button>
              <div className="session-actions"><button aria-label={session.pinned ? "Unpin chat" : "Pin chat"} onClick={() => void toggleSessionPinned(session)} title={session.pinned ? "Unpin chat" : "Pin chat"}>{session.pinned ? <PinOff size={14} /> : <Pin size={14} />}</button><button aria-label="Delete chat" onClick={() => void deleteSession(session.session_id)} title="Delete chat"><Trash2 size={14} /></button></div>
            </article>
          ))}
        </div>
        <div className="rail-footer"><div className="profile-avatar">AC</div><div><strong>Research workspace</strong><span>Local project</span></div><MoreHorizontal size={16} /></div>
      </aside>
    );
  }

  function renderWorkbench() {
    return (
      <section className="workbench-view">
        <div className="workbench-topline"><div><span className="eyebrow">Research assistant</span><h1>Ask your papers anything.</h1><p>Evidence-first answers from your indexed chemistry literature.</p></div><div className="topline-actions"><button className="secondary-button" onClick={() => void refreshDocuments()}><RefreshCw size={15} /> Sync library</button><label className="primary-button"><FileUp size={16} /> Upload PDFs<input type="file" accept="application/pdf" multiple onChange={(event) => { const files = event.target.files; if (files) void upload(files); event.currentTarget.value = ""; }} /></label></div></div>
        <div className="mobile-rail-actions"><button onClick={() => setMobilePanel("documents")}><Library size={16} /> Documents <span>{documents.length}</span></button><button onClick={() => setMobilePanel("sessions")}><MessageSquare size={16} /> Chats <span>{sessions.length}</span></button></div>
        <div className="workbench-grid"><div className="desktop-rail">{renderDocumentRail()}</div><div className="desktop-rail">{renderSessionRail()}</div><section className="chat-surface"><div className="chat-header"><div className="active-source"><span className="source-mark"><FlaskConical size={17} /></span><div><strong>{selectedDocument?.filename ?? "Select a document"}</strong><span>{selectedDocument ? `${selectedDocumentStatus} · ${scope === "current_document" ? "Focused source" : "All sources"}` : "Upload a PDF to begin"}</span></div></div><div className="scope-tabs"><button className={scope === "current_document" ? "active" : ""} onClick={() => setScope("current_document")}><FileText size={14} /> Current</button><button className={scope === "all_documents" ? "active" : ""} onClick={() => setScope("all_documents")}><Library size={14} /> All sources</button></div></div>{notice && <div className="notice"><CircleDashed size={15} /> {notice}</div>}{busy && chatStage && <div className={`stage-hint ${chatStage.stage}`} aria-live="polite"><Activity size={14} /> {chatStage.label}</div>}<div className="messages">{messages.length === 0 ? <div className="welcome-panel"><div className="welcome-icon"><Sparkles size={22} /></div><h2>Start with a research question</h2><p>Ask for mechanisms, methods, results, or a comparison. Every answer includes traceable evidence.</p><div className="prompt-grid"><button onClick={() => setQuestion("Summarize the main findings of this paper")}>Summarize the main findings <ArrowUpRight size={14} /></button><button onClick={() => setQuestion("Which experimental methods are used?")}>Find experimental methods <ArrowUpRight size={14} /></button><button onClick={() => setQuestion("What limitations do the authors report?")}>Surface limitations <ArrowUpRight size={14} /></button></div></div> : messages.map((message) => <ChatMessageView key={message.message_id} message={message} />)}</div><form className="composer" onSubmit={(event) => { event.preventDefault(); if (!busy) void ask(); }}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about mechanisms, methods, or results..." /><button type="submit" disabled={busy || !question.trim()} aria-label="Send question" title="Send question"><Send size={18} /></button></form></section></div>
      </section>
    );
  }

  function renderOverview() {
    return <section className="overview-view"><div className="section-heading"><div><span className="eyebrow">Workspace overview</span><h1>Good morning, researcher.</h1><p>Your evidence workspace is ready for the next question.</p></div><button className="primary-button" onClick={() => setActiveView("workbench")}><MessageSquare size={16} /> Open workbench</button></div><div className="metric-grid"><MetricCard icon={<FileText size={18} />} label="Documents" value={documents.length} detail={`${indexedDocuments.length} indexed`} tone="blue" /><MetricCard icon={<MessageSquare size={18} />} label="Conversations" value={sessions.length} detail="Saved research threads" tone="violet" /><MetricCard icon={<Activity size={18} />} label="Indexing queue" value={indexingDocuments.length} detail={failedDocuments.length ? `${failedDocuments.length} needs attention` : "All systems clear"} tone="amber" /><MetricCard icon={<ShieldCheck size={18} />} label="Evidence mode" value="ON" detail="Citations on every answer" tone="green" /></div><div className="overview-columns"><div className="overview-panel"><div className="panel-title"><div><span className="eyebrow">Recent sources</span><h2>Library health</h2></div><button className="text-button" onClick={() => setActiveView("library")}>View library <ArrowUpRight size={14} /></button></div>{documents.slice(0, 5).map((document) => <div className="health-row" key={document.document_id}><span className="document-icon"><FileText size={15} /></span><div><strong>{document.filename}</strong><span>{document.is_indexed ? `${document.chunk_count} indexed chunks` : getDocumentStatus(document)}</span></div><span className={`health-badge ${document.indexing_status}`}>{document.is_indexed ? "Ready" : document.indexing_status}</span></div>)}{documents.length === 0 && <div className="empty-state"><BookOpen size={26} /><span>Upload your first paper to populate the workspace.</span></div>}</div><div className="overview-panel accent-panel"><div className="panel-title"><div><span className="eyebrow">How it works</span><h2>Evidence pipeline</h2></div><Settings2 size={17} /></div><div className="pipeline"><PipelineStep index="01" label="Ingest" detail="Parse PDF and images" done={documents.length > 0} /><PipelineStep index="02" label="Retrieve" detail="Hybrid evidence search" done={indexedDocuments.length > 0} /><PipelineStep index="03" label="Answer" detail="Cite source pages" done={messages.some((message) => message.role === "assistant" && Boolean(message.content))} /></div></div></div></section>;
  }

  function renderLibraryView() {
    return <section className="library-view"><div className="section-heading"><div><span className="eyebrow">Knowledge base</span><h1>Document library</h1><p>Manage indexed papers and keep your evidence sources organized.</p></div><label className="primary-button"><UploadCloud size={16} /> Upload papers<input type="file" accept="application/pdf" multiple onChange={(event) => { const files = event.target.files; if (files) void upload(files); event.currentTarget.value = ""; }} /></label></div><div className="library-toolbar"><div className="rail-search"><Search size={15} /><input aria-label="Search library" placeholder="Search by filename" /></div><span>{documents.length} sources · {indexedDocuments.length} ready</span></div><div className="library-table"><div className="table-row table-header"><span>Document</span><span>Indexed content</span><span>Status</span><span>Added</span><span></span></div>{documents.map((document) => <div className="table-row" key={document.document_id}><div className="table-document"><span className="document-icon"><FileText size={17} /></span><button onClick={() => { setDocumentId(document.document_id); setActiveView("workbench"); }}>{document.filename}</button></div><span>{document.is_indexed ? `${document.chunk_count} chunks` : "—"}</span><span className={`health-badge ${document.indexing_status}`}>{document.is_indexed ? "Ready" : document.indexing_status}</span><span>{formatSessionTime(document.created_at)}</span><button className="icon-button" onClick={() => { setDocumentId(document.document_id); setActiveView("workbench"); }} aria-label="Open document"><ArrowUpRight size={16} /></button></div>)}{documents.length === 0 && <div className="empty-state"><BookOpen size={28} /><strong>No documents uploaded</strong><span>Upload PDF research papers to build your searchable knowledge base.</span></div>}</div></section>;
  }

  function renderInsightsView() {
    return <section className="insights-view"><div className="section-heading"><div><span className="eyebrow">Quality & operations</span><h1>Evidence insights</h1><p>Keep an eye on indexing health and answer traceability.</p></div><button className="secondary-button" onClick={() => void refreshDocuments()}><RefreshCw size={15} /> Refresh status</button></div><div className="insight-grid"><div className="insight-card"><div className="insight-card-head"><span className="insight-icon green"><ShieldCheck size={18} /></span><span className="health-badge succeeded">Healthy</span></div><h2>{indexedDocuments.length} indexed sources</h2><p>Ready for evidence-grounded retrieval.</p><div className="progress-track"><span style={{ width: documents.length ? `${(indexedDocuments.length / documents.length) * 100}%` : "0%" }} /></div></div><div className="insight-card"><div className="insight-card-head"><span className="insight-icon blue"><MessageSquare size={18} /></span><span className="health-badge pending">Live</span></div><h2>{messages.filter((message) => message.role === "assistant").length} answers in view</h2><p>Current conversation context and citations.</p><div className="insight-line"><span>Scope</span><strong>{scope === "current_document" ? "Focused source" : "All sources"}</strong></div></div><div className="insight-card"><div className="insight-card-head"><span className="insight-icon amber"><Activity size={18} /></span><span className="health-badge running">Monitor</span></div><h2>{indexingDocuments.length ? `${indexingDocuments.length} indexing` : "Queue clear"}</h2><p>{failedDocuments.length ? `${failedDocuments.length} source${failedDocuments.length > 1 ? "s" : ""} need attention.` : "No indexing failures detected."}</p><div className="insight-line"><span>Latest task</span><strong>{task?.status ?? "Idle"}</strong></div></div></div><div className="insight-note"><Sparkles size={18} /><div><strong>Designed for auditable research</strong><span>Answers surface source pages, quoted evidence, related figures, and explicit limitations so the next decision stays reviewable.</span></div></div></section>;
  }

  return <main className="app-shell"><aside className="app-sidebar"><div className="brand-lockup"><span className="brand-mark"><FlaskConical size={19} /></span><div><strong>Agentic</strong><span>ChemRAG</span></div></div><nav className="primary-nav"><span className="nav-label">Workspace</span><NavItem icon={<LayoutDashboard size={17} />} label="Overview" active={activeView === "overview"} onClick={() => setActiveView("overview")} /><NavItem icon={<MessageSquare size={17} />} label="Workbench" active={activeView === "workbench"} onClick={() => setActiveView("workbench")} /><NavItem icon={<Library size={17} />} label="Document library" active={activeView === "library"} onClick={() => setActiveView("library")} /><span className="nav-label secondary">System</span><NavItem icon={<Activity size={17} />} label="Evidence insights" active={activeView === "insights"} onClick={() => setActiveView("insights")} /><NavItem icon={<Settings2 size={17} />} label="Settings" disabled /></nav><div className="sidebar-bottom"><div className="system-status"><span className={`live-dot ${apiOnline ? "" : "offline"}`} /><div><strong>Local API</strong><span>{apiOnline ? "Ready for research" : "Waiting for server"}</span></div></div><div className="sidebar-version">v0.1 · Evidence mode</div></div></aside><div className="app-content"><header className="topbar"><div className="topbar-context"><span className="mobile-brand-mark"><FlaskConical size={17} /></span><span>{activeView === "overview" ? "Overview" : activeView === "workbench" ? "Research workbench" : activeView === "library" ? "Document library" : "Evidence insights"}</span><ChevronDown size={14} /></div><div className="topbar-actions"><span className="api-chip"><span className={`live-dot ${apiOnline ? "" : "offline"}`} /> {apiOnline ? "API connected" : "API offline"}</span><button className="icon-button" aria-label="Settings" title="Settings"><Settings2 size={17} /></button><div className="profile-avatar small">AC</div></div></header><div className="page-content">{activeView === "overview" ? renderOverview() : activeView === "library" ? renderLibraryView() : activeView === "insights" ? renderInsightsView() : renderWorkbench()}</div></div>{mobilePanel && <div className="mobile-overlay" onClick={() => setMobilePanel(null)}><div onClick={(event) => event.stopPropagation()}>{mobilePanel === "documents" ? renderDocumentRail() : renderSessionRail()}<button className="mobile-close" onClick={() => setMobilePanel(null)}><XCircle size={18} /> Close</button></div></div>}</main>;
}

function NavItem({ icon, label, active, disabled, onClick }: { icon: React.ReactNode; label: string; active?: boolean; disabled?: boolean; onClick?: () => void }) {
  return <button className={`nav-item ${active ? "active" : ""} ${disabled ? "disabled" : ""}`} onClick={onClick} disabled={disabled} aria-label={label}>{icon}<span>{label}</span>{active && <span className="nav-active-mark" />}</button>;
}

function MetricCard({ icon, label, value, detail, tone }: { icon: React.ReactNode; label: string; value: string | number; detail: string; tone: string }) {
  return <article className="metric-card"><span className={`metric-icon ${tone}`}>{icon}</span><div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></article>;
}

function PipelineStep({ index, label, detail, done }: { index: string; label: string; detail: string; done: boolean }) {
  return <div className="pipeline-step"><span className={`pipeline-index ${done ? "done" : ""}`}>{done ? <CheckCircle2 size={14} /> : index}</span><div><strong>{label}</strong><span>{detail}</span></div></div>;
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
