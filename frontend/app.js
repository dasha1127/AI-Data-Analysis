// ── Config ────────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  dataset: null,       // currently selected dataset name
  groqKey: "",         // Groq API key (from localStorage)
  messages: [],        // chat history
  schema: null,        // current dataset schema
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const els = {
  datasetBtns:  document.querySelectorAll(".dataset-btn"),
  uploadZone:   $("upload-zone"),
  uploadInput:  $("upload-input"),
  schemaList:   $("schema-list"),
  metrics:      $("metrics"),
  chatMessages: $("chat-messages"),
  chatInput:    $("chat-input"),
  sendBtn:      $("send-btn"),
  groqKey:      $("groq-key"),
  pillStatus:   $("pill-status"),
  tabBtns:      document.querySelectorAll(".tab-btn"),
  panels:       document.querySelectorAll(".panel"),
};

// ── Plotly config ─────────────────────────────────────────────────────────────
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: false,
};
const PLOTLY_LAYOUT_PATCH = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { color: "#e0eaff", family: "Inter, system-ui, sans-serif" },
  margin: { t: 40, b: 40, l: 40, r: 20 },
};

// ── API helpers ───────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ── Chart rendering ───────────────────────────────────────────────────────────

function renderChart(container, figData) {
  if (typeof Plotly === "undefined") {
    container.innerHTML = `<p style="color:var(--muted);padding:20px">Plotly is still loading, please wait a moment and refresh.</p>`;
    return;
  }
  const layout = { ...figData.layout, ...PLOTLY_LAYOUT_PATCH };
  Plotly.react(container, figData.data, layout, PLOTLY_CONFIG);
}

function makeChartDiv(id) {
  const div = document.createElement("div");
  div.id = id;
  div.style.width = "100%";
  div.style.minHeight = "320px";
  return div;
}

// ── Tab switching ─────────────────────────────────────────────────────────────

els.tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    els.tabBtns.forEach(b => b.classList.remove("active"));
    els.panels.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${btn.dataset.tab}`).classList.add("active");

    // Load tab content lazily
    if (btn.dataset.tab === "insights" && state.dataset) loadInsights();
    if (btn.dataset.tab === "trends"   && state.dataset === "happiness") loadTrends();
    if (btn.dataset.tab === "data"     && state.dataset) loadRawData();
  });
});

// ── Dataset selection ─────────────────────────────────────────────────────────

els.datasetBtns.forEach(btn => {
  btn.addEventListener("click", () => selectDataset(btn.dataset.name));
});

async function selectDataset(name) {
  state.dataset = name;

  // Highlight active button
  els.datasetBtns.forEach(b => b.classList.toggle("active", b.dataset.name === name));

  // Load schema → update metrics + sidebar
  try {
    state.schema = await apiFetch(`/schema/${name}`);
    renderSchema(state.schema);
    renderMetrics(state.schema);
  } catch (e) {
    showError(e.message);
  }

  // Reload whichever tab is currently active
  const activeTab = document.querySelector(".tab-btn.active")?.dataset.tab;
  if (activeTab === "insights") loadInsights();
  if (activeTab === "trends"  ) loadTrends();
  if (activeTab === "data"    ) loadRawData();

  // Clear chat when switching datasets
  state.messages = [];
  els.chatMessages.innerHTML = "";
}

// ── Schema + metrics ──────────────────────────────────────────────────────────

function renderSchema(schema) {
  els.schemaList.innerHTML = schema.columns
    .map(c => `
      <li>
        <span>${c.name}</span>
        <span class="badge">${c.dtype}</span>
      </li>`)
    .join("");
}

function renderMetrics(schema) {
  const numCols = schema.columns.filter(c => c.dtype.includes("int") || c.dtype.includes("float")).length;
  const catCols = schema.columns.length - numCols;
  els.metrics.innerHTML = `
    <div class="metric-card"><div class="metric-num">${schema.rows.toLocaleString()}</div><div class="metric-label">Rows</div></div>
    <div class="metric-card"><div class="metric-num">${schema.cols}</div><div class="metric-label">Columns</div></div>
    <div class="metric-card"><div class="metric-num">${numCols}</div><div class="metric-label">Numeric</div></div>
    <div class="metric-card"><div class="metric-num">${catCols}</div><div class="metric-label">Categorical</div></div>
    <div class="metric-card"><div class="metric-num">${state.dataset === "happiness" ? "8" : "—"}</div><div class="metric-label">Years</div></div>
  `;
}

// ── Auto-Insights tab ─────────────────────────────────────────────────────────

async function loadInsights() {
  if (!state.dataset) return;
  const panel = $("panel-insights");
  panel.innerHTML = `<div class="thinking"><span class="spinner"></span> crunching the numbers...</div>`;

  try {
    const insights = await apiFetch(`/insights/${state.dataset}`);
    const grid = document.createElement("div");
    grid.className = "chart-grid";

    insights.forEach((insight, i) => {
      const card = document.createElement("div");
      card.className = "chart-card" + (i === 0 ? " full" : "");
      const div = makeChartDiv(`insight-${i}`);
      card.appendChild(div);
      grid.appendChild(card);
    });

    panel.innerHTML = "";
    panel.appendChild(grid);

    // Render charts after DOM is ready
    insights.forEach((insight, i) => {
      renderChart($(`insight-${i}`), insight.fig);
    });
  } catch (e) {
    panel.innerHTML = `<div class="empty-state"><span class="empty-icon">⚠️</span>${e.message}</div>`;
  }
}

// ── Happiness Trends tab ──────────────────────────────────────────────────────

async function loadTrends() {
  const panel = $("panel-trends");
  if (state.dataset !== "happiness") {
    panel.innerHTML = `<div class="empty-state"><span class="empty-icon">🌍</span>Load the World Happiness dataset to see trends.</div>`;
    return;
  }

  panel.innerHTML = `<div class="thinking"><span class="spinner"></span> pulling up the charts...</div>`;

  try {
    const trends = await apiFetch("/happiness/trends");
    panel.innerHTML = `
      <div class="chart-grid">
        <div class="chart-card full" id="t-global"></div>
        <div class="chart-card" id="t-top10"></div>
        <div class="chart-card" id="t-region"></div>
        <div class="chart-card full" id="t-gdp"></div>
        <div class="chart-card full" id="t-improved"></div>
      </div>`;

    renderChart($("t-global"),   trends.global_trend);
    renderChart($("t-top10"),    trends.top10);
    renderChart($("t-region"),   trends.by_region);
    renderChart($("t-gdp"),      trends.gdp_scatter);
    renderChart($("t-improved"), trends.most_improved);
  } catch (e) {
    panel.innerHTML = `<div class="empty-state"><span class="empty-icon">⚠️</span>${e.message}</div>`;
  }
}

// ── Raw Data tab ──────────────────────────────────────────────────────────────

async function loadRawData() {
  if (!state.dataset || !state.schema) return;
  const panel = $("panel-data");
  panel.innerHTML = `<div class="thinking"><span class="spinner"></span> loading...</div>`;

  // We only have schema — show columns as a placeholder table header
  const cols = state.schema.columns.map(c => c.name);
  panel.innerHTML = `
    <div class="data-table-wrap">
      <table>
        <thead>
          <tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>
        </thead>
        <tbody>
          <tr><td colspan="${cols.length}" style="text-align:center;padding:32px;color:var(--muted)">
            Use the Chat tab to query specific rows, or load data via the API.
          </td></tr>
        </tbody>
      </table>
    </div>`;
}

// ── Chat ──────────────────────────────────────────────────────────────────────

els.sendBtn.addEventListener("click", sendMessage);
els.chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const question = els.chatInput.value.trim();
  if (!question || !state.dataset) return;

  els.chatInput.value = "";
  els.sendBtn.disabled = true;

  // Show user bubble
  appendBubble("user", question);

  // Show thinking indicator
  const thinking = appendThinking();

  try {
    const result = await apiFetch("/analyze", {
      method: "POST",
      body: JSON.stringify({
        question,
        dataset: state.dataset,
        groq_key: state.groqKey || null,
      }),
    });

    thinking.remove();
    appendAgentResponse(result);
  } catch (e) {
    thinking.remove();
    appendBubble("agent", `⚠️ ${e.message}`);
  } finally {
    els.sendBtn.disabled = false;
    els.chatInput.focus();
  }
}

function appendBubble(role, text) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `
    <div class="bubble ${role}">
      <div class="bubble-meta">${role === "user" ? "You" : "DataMind"}</div>
      ${escapeHtml(text)}
    </div>`;
  els.chatMessages.appendChild(wrap);
  scrollChat();
  return wrap;
}

function appendThinking() {
  const el = document.createElement("div");
  el.className = "thinking";
  el.innerHTML = `<span class="spinner"></span> thinking...`;
  els.chatMessages.appendChild(el);
  scrollChat();
  return el;
}

function appendAgentResponse(result) {
  const wrap = document.createElement("div");

  let html = "";
  if (result.text) {
    html += `
      <div class="bubble agent">
        <div class="bubble-meta">DataMind</div>
        ${escapeHtml(result.text)}
      </div>`;
  }
  wrap.innerHTML = html;
  els.chatMessages.appendChild(wrap);

  if (result.fig) {
    const chartWrap = document.createElement("div");
    chartWrap.className = "chart-bubble";
    const chartId = `chat-chart-${Date.now()}`;
    const div = makeChartDiv(chartId);
    chartWrap.appendChild(div);
    els.chatMessages.appendChild(chartWrap);
    setTimeout(() => renderChart(div, result.fig), 50);
  }

  scrollChat();
}

function scrollChat() {
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

// ── File upload ───────────────────────────────────────────────────────────────

els.uploadZone.addEventListener("click", () => els.uploadInput.click());
els.uploadInput.addEventListener("change", e => handleUpload(e.target.files[0]));

els.uploadZone.addEventListener("dragover", e => { e.preventDefault(); els.uploadZone.classList.add("drag"); });
els.uploadZone.addEventListener("dragleave", () => els.uploadZone.classList.remove("drag"));
els.uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  els.uploadZone.classList.remove("drag");
  handleUpload(e.dataTransfer.files[0]);
});

async function handleUpload(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);

  els.uploadZone.innerHTML = `<span class="icon"><span class="spinner"></span></span> uploading...`;

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    const data = await res.json();

    // Add button for the uploaded dataset
    const btn = document.createElement("button");
    btn.className = "dataset-btn";
    btn.dataset.name = data.dataset_id;
    btn.textContent = `📄 ${data.dataset_id} (${data.rows} rows)`;
    btn.addEventListener("click", () => selectDataset(data.dataset_id));
    document.querySelector(".sidebar-section:first-of-type").appendChild(btn);

    els.uploadZone.innerHTML = `<span class="icon">✅</span> got it! click above to explore.`;
    selectDataset(data.dataset_id);
  } catch (e) {
    els.uploadZone.innerHTML = `<span class="icon">😬</span> something went wrong, try again.`;
  }
}

// ── Groq key ──────────────────────────────────────────────────────────────────

state.groqKey = localStorage.getItem("groq_key") || "";
if (state.groqKey) {
  els.groqKey.value = state.groqKey;
  updatePill(true);
}

els.groqKey.addEventListener("input", e => {
  state.groqKey = e.target.value.trim();
  localStorage.setItem("groq_key", state.groqKey);
  updatePill(!!state.groqKey);
});

function updatePill(on) {
  els.pillStatus.className = `pill ${on ? "on" : "off"}`;
  els.pillStatus.innerHTML = on
    ? `<span class="dot"></span> LLM ON — Llama 3.3 70B`
    : `<span class="dot"></span> Rule-based mode`;
}

function showError(msg) {
  console.error(msg);
}

// ── Init — wait for full page load so Plotly is definitely ready ──────────────
window.addEventListener("load", () => selectDataset("happiness"));
