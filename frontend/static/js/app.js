const API_BASE = "";

let state = {
  models: {},
  activeModel: null,
  machines: [],
  overview: null,
  trainingRunning: false,
};

async function api(path, opts = {}) {
  const url = `${API_BASE}/api${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function showToast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function showLoading(show = true) {
  document.getElementById("loading-overlay").classList.toggle("show", show);
}

function navigate(sectionId) {
  document.querySelectorAll(".page-section").forEach((s) => s.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  document.getElementById(sectionId).classList.add("active");
  document.querySelector(`.nav-item[data-section="${sectionId}"]`).classList.add("active");
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    const section = item.dataset.section;
    if (section) navigate(section);
  });
});

// ============ DASHBOARD ============
async function loadDashboard() {
  try {
    showLoading(true);
    const [overview, modelsData] = await Promise.all([api("/data/overview"), api("/models")]);
    state.overview = overview;
    state.models = modelsData.models;
    state.activeModel = modelsData.default;

    document.getElementById("stat-machines").textContent = overview.static_machines || 0;
    document.getElementById("stat-records").textContent = overview.dynamic_records || 0;
    document.getElementById("stat-at-risk").textContent = overview.at_risk || 0;
    document.getElementById("stat-fault-rate").textContent = (overview.fault_rate || 0) + "%";
    document.getElementById("stat-total-faults").textContent = overview.total_faults || 0;
    document.getElementById("stat-models").textContent = Object.keys(modelsData.models).length;

    const topRiskBody = document.getElementById("top-risk-body");
    topRiskBody.innerHTML = "";
    if (overview.top_risk && overview.top_risk.length > 0) {
      overview.top_risk.forEach((m) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${m.TERMINAL_CODE}</strong></td>
          <td>${(m.fault_probability * 100).toFixed(1)}%</td>
          <td>${m.predicted_rul_days.toFixed(1)} days</td>
          <td><span class="badge ${m.is_fault_risk ? "badge-risk" : "badge-safe"}">${m.is_fault_risk ? "At Risk" : "Safe"}</span></td>
        `;
        topRiskBody.appendChild(row);
      });
    } else {
      topRiskBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">No predictions available. Run predictions first.</td></tr>';
    }

    if (overview.failure_modes) {
      renderFailureModeChart(overview.failure_modes);
    } else {
      document.getElementById("fm-chart").innerHTML = '<div class="empty-state"><div class="empty-icon">📊</div><p>No failure mode data</p></div>';
    }

    document.getElementById("last-refresh").textContent = new Date().toLocaleTimeString();
  } catch (err) {
    showToast("Dashboard: " + err.message, "error");
  } finally {
    showLoading(false);
  }
}

function renderFailureModeChart(modes) {
  const container = document.getElementById("fm-chart");
  container.innerHTML = '<canvas id="fmChartCanvas"></canvas>';
  const canvas = document.getElementById("fmChartCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const labels = Object.keys(modes);
  const values = Object.values(modes);
  const colors = ["#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4", "#10b981", "#f97316"];

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Count",
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" } },
        y: { ticks: { color: "#8899b4", precision: 0 }, grid: { color: "#1a2332" }, beginAtZero: true },
      },
    },
  });
}

// ============ MACHINES ============
let machineSearchTimer;
async function loadMachines(search) {
  try {
    showLoading(true);
    let url = "/data/machines";
    if (search) url += `?search=${encodeURIComponent(search)}`;
    const data = await api(url);
    state.machines = data.machines;

    document.getElementById("machines-count").textContent = `${data.total} machines`;
    const tbody = document.getElementById("machines-body");
    tbody.innerHTML = "";

    if (data.machines.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:40px;color:var(--text-muted);">No machines found</td></tr>';
      return;
    }

    data.machines.forEach((m) => {
      const risk = m.is_fault_risk;
      const prob = m.fault_probability != null ? (m.fault_probability * 100).toFixed(1) + "%" : "—";
      const rul = m.predicted_rul_days != null ? m.predicted_rul_days.toFixed(1) + " days" : "—";
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${m.TERMINAL_CODE}</strong></td>
        <td>${m.TERMINAL_TYPE_ID}</td>
        <td>${m.PLATFORM || "—"}</td>
        <td>${m.ASA_VERSION || "—"}</td>
        <td>${prob}</td>
        <td>${rul}</td>
        <td>
          ${risk === true ? '<span class="badge badge-risk">&#9888; At Risk</span>'
            : risk === false ? '<span class="badge badge-safe">&#10003; Safe</span>'
            : '<span class="badge badge-neutral">No Data</span>'}
        </td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    showToast("Machines: " + err.message, "error");
  } finally {
    showLoading(false);
  }
}

document.getElementById("machine-search").addEventListener("input", (e) => {
  clearTimeout(machineSearchTimer);
  machineSearchTimer = setTimeout(() => loadMachines(e.target.value), 300);
});

// ============ PREDICTIONS ============
async function loadPredictions() {
  try {
    const modelsData = await api("/models");
    const sel = document.getElementById("pred-model");
    sel.innerHTML = "";
    Object.keys(modelsData.models).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });

    const overview = await api("/data/overview");
    const predCount = overview.predictions_count || 0;
    const atRisk = overview.at_risk || 0;
    document.getElementById("pred-info").innerHTML = predCount > 0
      ? `<div class="alert alert-info">&#128203; ${predCount} predictions &middot; ${atRisk} machines at risk &middot; ${overview.avg_fault_prob ? (overview.avg_fault_prob * 100).toFixed(1) + "% avg probability" : ""}</div>`
      : '<div class="alert alert-info">&#128203; No predictions yet. Click "Run Predictions" to start.</div>';

    const tbody = document.getElementById("predictions-body");
    tbody.innerHTML = "";
    if (overview.top_risk && overview.top_risk.length > 0) {
      overview.top_risk.forEach((m) => {
        const pct = (m.fault_probability * 100).toFixed(1);
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${m.TERMINAL_CODE}</strong></td>
          <td>${pct}%</td>
          <td>
            <div style="background:var(--bg-input);border-radius:4px;height:6px;width:100px;overflow:hidden;">
              <div style="height:100%;width:${pct}%;background:${m.fault_probability > 0.5 ? "var(--accent-red)" : "var(--accent-green)"};border-radius:4px;"></div>
            </div>
          </td>
          <td>${m.predicted_rul_days.toFixed(1)} days</td>
          <td><span class="badge ${m.is_fault_risk ? "badge-risk" : "badge-safe"}">${m.is_fault_risk ? "&#9888; At Risk" : "&#10003; Safe"}</span></td>
        `;
        tbody.appendChild(row);
      });
    } else {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:40px;color:var(--text-muted);">Run predictions to see results</td></tr>';
    }
  } catch (err) {
    // silent
  }
}

document.getElementById("run-prediction").addEventListener("click", async () => {
  const btn = document.getElementById("run-prediction");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';
  const modelName = document.getElementById("pred-model").value || state.activeModel;
  const seqLen = document.getElementById("pred-seqlen").value;
  try {
    const result = await api(`/predict?model_name=${modelName}&seq_len=${seqLen}`, { method: "POST" });
    showToast(`Done: ${result.predictions_count} predictions, ${result.at_risk} at risk (${result.at_risk_pct}%)`, "success");
    await Promise.all([loadPredictions(), loadDashboard()]);
  } catch (err) {
    showToast("Prediction failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = "&#9654; Run Predictions";
  }
});

// ============ TRAINING (WebSocket Live) ============
let trainingHistory = [];
let trainChartInstance = null;

document.getElementById("train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.trainingRunning) return;

  const form = e.target;
  const config = {
    num_machines: parseInt(form.num_machines.value),
    days: parseInt(form.days.value),
    epochs: parseInt(form.epochs.value),
    seq_len: parseInt(form.seq_len.value),
    stride: parseInt(form.stride.value),
    d_model: parseInt(form.d_model.value),
    nhead: parseInt(form.nhead.value),
    num_layers: parseInt(form.num_layers.value),
  };

  state.trainingRunning = true;
  trainingHistory = [];
  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Training...';

  const prog = document.getElementById("train-progress");
  prog.style.display = "block";
  document.getElementById("train-status").textContent = "Initializing...";
  document.getElementById("train-epoch").textContent = "0";
  document.getElementById("train-total").textContent = config.epochs;
  document.getElementById("train-bar").style.width = "0%";
  document.getElementById("train-metrics").innerHTML = "";
  document.getElementById("train-chart").innerHTML = '<canvas id="trainChartLive"></canvas>';

  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProto}//${location.host}/ws/train`;
  let ws;
  try {
    ws = new WebSocket(wsUrl);
  } catch (err) {
    // fallback to HTTP
    try {
      const result = await api("/train", { method: "POST", body: JSON.stringify(config) });
      document.getElementById("train-status").textContent = "Training complete!";
      document.getElementById("train-epoch").textContent = config.epochs;
      document.getElementById("train-bar").style.width = "100%";
      if (result.history) updateTrainingUI(result.history);
      showToast("Training completed!", "success");
      await loadDashboard();
    } catch (err2) {
      document.getElementById("train-status").textContent = "Training failed: " + err2.message;
      showToast("Training failed: " + err2.message, "error");
    } finally {
      state.trainingRunning = false;
      btn.disabled = false;
      btn.innerHTML = "&#128640; Start Training";
    }
    return;
  }

  ws.onopen = () => {
    ws.send(JSON.stringify(config));
    document.getElementById("train-status").textContent = "Generating data...";
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "log") {
      document.getElementById("train-status").textContent = msg.message;
    }

    if (msg.type === "epoch") {
      const d = msg.data;
      trainingHistory.push(d);
      document.getElementById("train-epoch").textContent = d.epoch;
      document.getElementById("train-bar").style.width = (d.epoch / config.epochs * 100) + "%";
      document.getElementById("train-status").textContent = `Epoch ${d.epoch}/${config.epochs} complete`;
      document.getElementById("train-metrics").innerHTML = `
        <div class="metrics-grid">
          <div class="metric-item">
            <div class="metric-value" style="color:var(--accent-blue)">${d.train_loss.toFixed(4)}</div>
            <div class="metric-label">Train Loss</div>
          </div>
          <div class="metric-item">
            <div class="metric-value" style="color:var(--accent-green)">${(d.val_fault_accuracy * 100).toFixed(1)}%</div>
            <div class="metric-label">Accuracy</div>
          </div>
          <div class="metric-item">
            <div class="metric-value" style="color:var(--accent-cyan)">${d.val_fault_f1.toFixed(4)}</div>
            <div class="metric-label">F1 Score</div>
          </div>
          <div class="metric-item">
            <div class="metric-value" style="color:var(--accent-yellow)">${d.val_rul_mae.toFixed(2)}d</div>
            <div class="metric-label">RUL MAE</div>
          </div>
        </div>
      `;
      renderLiveTrainingChart(trainingHistory);
    }

    if (msg.type === "complete") {
      document.getElementById("train-status").textContent = "Training complete!";
      document.getElementById("train-epoch").textContent = config.epochs;
      document.getElementById("train-bar").style.width = "100%";
      updateTrainingUI(trainingHistory);
      showToast("Training completed successfully!", "success");
      ws.close();
      state.trainingRunning = false;
      btn.disabled = false;
      btn.innerHTML = "&#128640; Start Training";
      loadDashboard();
    }

    if (msg.type === "error") {
      document.getElementById("train-status").textContent = "Error: " + msg.message;
      showToast("Training error: " + msg.message, "error");
      ws.close();
      state.trainingRunning = false;
      btn.disabled = false;
      btn.innerHTML = "&#128640; Start Training";
    }
  };

  ws.onerror = () => {
    document.getElementById("train-status").textContent = "WebSocket connection failed";
    showToast("WebSocket connection failed", "error");
    state.trainingRunning = false;
    btn.disabled = false;
    btn.innerHTML = "&#128640; Start Training";
  };

  ws.onclose = () => {
    if (state.trainingRunning) {
      state.trainingRunning = false;
      btn.disabled = false;
      btn.innerHTML = "&#128640; Start Training";
    }
  };
});

function updateTrainingUI(history) {
  if (history.length === 0) return;
  const last = history[history.length - 1];
  document.getElementById("train-metrics").innerHTML = `
    <div class="metrics-grid">
      <div class="metric-item">
        <div class="metric-value" style="color:var(--accent-blue)">${(last.train_loss || 0).toFixed(4)}</div>
        <div class="metric-label">Train Loss</div>
      </div>
      <div class="metric-item">
        <div class="metric-value" style="color:var(--accent-green)">${((last.val_fault_accuracy || 0) * 100).toFixed(1)}%</div>
        <div class="metric-label">Accuracy</div>
      </div>
      <div class="metric-item">
        <div class="metric-value" style="color:var(--accent-cyan)">${(last.val_fault_f1 || 0).toFixed(4)}</div>
        <div class="metric-label">F1 Score</div>
      </div>
      <div class="metric-item">
        <div class="metric-value" style="color:var(--accent-yellow)">${(last.val_rul_mae || 0).toFixed(2)}d</div>
        <div class="metric-label">RUL MAE</div>
      </div>
    </div>
  `;
  renderLiveTrainingChart(history);
}

function renderLiveTrainingChart(history) {
  const canvas = document.getElementById("trainChartLive");
  if (!canvas || history.length === 0) return;
  const ctx = canvas.getContext("2d");
  const epochs = history.map((h) => h.epoch);
  const losses = history.map((h) => h.train_loss);
  const f1s = history.map((h) => h.val_fault_f1);

  if (trainChartInstance) trainChartInstance.destroy();

  trainChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train Loss",
          data: losses,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
        },
        {
          label: "Val F1 Score",
          data: f1s,
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { intersect: false, mode: "index" },
      plugins: { legend: { labels: { color: "#8899b4" } } },
      scales: {
        x: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" }, title: { display: true, text: "Epoch", color: "#5a6d8a" } },
        y: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" }, title: { display: true, text: "Loss", color: "#5a6d8a" } },
        y1: { position: "right", ticks: { color: "#8899b4" }, grid: { drawOnChartArea: false }, title: { display: true, text: "F1", color: "#5a6d8a" }, min: 0, max: 1 },
      },
    },
  });
}

// ============ ANALYTICS ============
async function loadAnalytics() {
  try {
    showLoading(true);
    const modelsData = await api("/models");
    const sel = document.getElementById("analytics-model");
    sel.innerHTML = '<option value="">Select model...</option>';
    Object.keys(modelsData.models).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === modelsData.default) opt.selected = true;
      sel.appendChild(opt);
    });
    if (modelsData.default) await loadModelHistory(modelsData.default);
  } catch (err) {
    showToast("Analytics: " + err.message, "error");
  } finally {
    showLoading(false);
  }
}

document.getElementById("analytics-model").addEventListener("change", async (e) => {
  if (e.target.value) await loadModelHistory(e.target.value);
});

async function loadModelHistory(modelName) {
  try {
    const data = await api(`/model/${modelName}/history`);
    const history = data.history;

    document.getElementById("analytics-metrics").innerHTML = "";
    document.getElementById("analytics-charts").innerHTML = "";

    if (history && history.length > 0) {
      const last = history[history.length - 1];
      document.getElementById("analytics-metrics").innerHTML = `
        <div class="metrics-grid">
          ${Object.entries(last).filter(([k]) => k !== "epoch").map(([k, v]) => `
            <div class="metric-item">
              <div class="metric-value" style="color:var(--accent-cyan)">${typeof v === "number" ? (v < 1 ? v.toFixed(4) : v.toFixed(2)) : v}</div>
              <div class="metric-label">${k.replace(/_/g, " ")}</div>
            </div>
          `).join("")}
        </div>
      `;
      renderAnalyticsCharts(history);
    } else {
      document.getElementById("analytics-charts").innerHTML = '<div class="empty-state"><div class="empty-icon">&#128202;</div><h3>No training history</h3><p>Train a model first</p></div>';
    }
  } catch (err) {
    document.getElementById("analytics-charts").innerHTML = `<div class="empty-state"><div class="empty-icon">&#9888;</div><p>${err.message}</p></div>`;
  }
}

function renderAnalyticsCharts(history) {
  const container = document.getElementById("analytics-charts");
  container.innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><h3>Loss & Accuracy</h3></div>
        <div class="card-body"><div class="chart-container"><canvas id="chart-loss-acc"></canvas></div></div>
      </div>
      <div class="card">
        <div class="card-header"><h3>F1, Precision, Recall</h3></div>
        <div class="card-body"><div class="chart-container"><canvas id="chart-f1"></canvas></div></div>
      </div>
    </div>
    <div class="card" style="margin-top:20px;">
      <div class="card-header"><h3>RUL MAE Over Epochs</h3></div>
      <div class="card-body"><div class="chart-container" style="min-height:250px;"><canvas id="chart-rul"></canvas></div></div>
    </div>
  `;

  const epochs = history.map((h) => h.epoch);
  const losses = history.map((h) => h.train_loss);
  const accs = history.map((h) => h.val_fault_accuracy);

  new Chart(document.getElementById("chart-loss-acc"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "Train Loss", data: losses, borderColor: "#3b82f6", tension: 0.3, pointRadius: 3 },
        { label: "Accuracy", data: accs, borderColor: "#10b981", tension: 0.3, pointRadius: 3, yAxisID: "y1" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8899b4" } } },
      scales: {
        x: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" } },
        y: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" } },
        y1: { position: "right", ticks: { color: "#8899b4" }, grid: { drawOnChartArea: false }, min: 0, max: 1 },
      },
    },
  });

  const f1s = history.map((h) => h.val_fault_f1);
  const precs = history.map((h) => h.val_fault_precision);
  const recs = history.map((h) => h.val_fault_recall);

  new Chart(document.getElementById("chart-f1"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "F1 Score", data: f1s, borderColor: "#8b5cf6", tension: 0.3, pointRadius: 3 },
        { label: "Precision", data: precs, borderColor: "#06b6d4", tension: 0.3, pointRadius: 3 },
        { label: "Recall", data: recs, borderColor: "#f59e0b", tension: 0.3, pointRadius: 3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8899b4" } } },
      scales: {
        x: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" } },
        y: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" }, min: 0, max: 1 },
      },
    },
  });

  const rul = history.map((h) => h.val_rul_mae || 0);
  new Chart(document.getElementById("chart-rul"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [{ label: "RUL MAE (days)", data: rul, borderColor: "#f97316", tension: 0.3, pointRadius: 3, fill: true, backgroundColor: "rgba(249,115,22,0.1)" }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8899b4" } } },
      scales: {
        x: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" } },
        y: { ticks: { color: "#8899b4" }, grid: { color: "#1a2332" }, beginAtZero: true },
      },
    },
  });
}

// ============ INIT ============
document.addEventListener("DOMContentLoaded", async () => {
  showLoading(true);
  try {
    await Promise.all([loadDashboard(), loadPredictions(), loadAnalytics()]);
  } catch (err) {
    showToast("Init: " + err.message, "error");
  } finally {
    showLoading(false);
  }
});
