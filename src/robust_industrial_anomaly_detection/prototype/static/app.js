"use strict";

const state = { config: null, files: [], results: [], selected: 0 };
const $ = (id) => document.getElementById(id);
const category = $("category");
const fileInput = $("files");
const inspectButton = $("inspect");
const fileDrop = $("file-drop");
const themeButton = $("theme-toggle");

function applyTheme(theme) {
  const isLight = theme === "light";
  document.body.dataset.theme = isLight ? "light" : "dark";
  themeButton.setAttribute("aria-pressed", String(isLight));
  themeButton.setAttribute("aria-label", `Switch to ${isLight ? "dark" : "light"} mode`);
  $("theme-label").textContent = isLight ? "LIGHT" : "DARK";
}

themeButton.addEventListener("click", () => {
  applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
});

function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("hidden");
}

function clearError() { $("error").classList.add("hidden"); }
function formatNumber(value, digits = 4) { return Number(value).toFixed(digits); }
function formatBytes(bytes) { return `${(bytes / 1024 / 1024).toFixed(0)} MB maximum per image`; }

async function initialize() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("The local model did not become ready.");
    state.config = await response.json();
    category.innerHTML = state.config.categories.map((item) => `<option value="${item}">${item}</option>`).join("");
    category.disabled = false;
    fileInput.disabled = false;
    $("model-name").textContent = `${state.config.model} · ${state.config.device}`;
    $("model-spec").textContent = state.config.model.replaceAll("_", " ").toUpperCase();
    $("system-status").textContent = "READY / LOCAL";
    $("status-dot").classList.add("ready");
    $("file-summary").textContent = `Up to ${state.config.max_batch_files} files · ${formatBytes(state.config.max_upload_bytes)}`;
    updateButton();
  } catch (error) {
    $("system-status").textContent = "Model unavailable";
    $("status-dot").classList.add("error");
    showError(error.message);
  }
}

function acceptFiles(files) {
  clearError();
  const selected = Array.from(files || []);
  if (!state.config) return;
  if (selected.length > state.config.max_batch_files) {
    showError(`Choose no more than ${state.config.max_batch_files} files per batch.`);
    return;
  }
  const oversized = selected.find((file) => file.size > state.config.max_upload_bytes);
  if (oversized) {
    showError(`${oversized.name} exceeds the per-image size limit.`);
    return;
  }
  state.files = selected;
  $("file-summary").textContent = selected.length ? `${selected.length} image${selected.length === 1 ? "" : "s"} selected` : "No images selected";
  updateButton();
}

function updateButton() { inspectButton.disabled = !state.config || state.files.length === 0; }
fileInput.addEventListener("change", () => acceptFiles(fileInput.files));
fileDrop.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") fileInput.click(); });
["dragenter", "dragover"].forEach((name) => fileDrop.addEventListener(name, (event) => { event.preventDefault(); fileDrop.classList.add("drag"); }));
["dragleave", "drop"].forEach((name) => fileDrop.addEventListener(name, (event) => { event.preventDefault(); fileDrop.classList.remove("drag"); }));
fileDrop.addEventListener("drop", (event) => acceptFiles(event.dataTransfer.files));

async function inspectFile(file) {
  const response = await fetch(`/api/inspect?category=${encodeURIComponent(category.value)}`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream", "X-Filename": file.name },
    body: file,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Inspection failed for ${file.name}.`);
  return payload;
}

inspectButton.addEventListener("click", async () => {
  clearError();
  inspectButton.disabled = true;
  category.disabled = true;
  fileInput.disabled = true;
  state.results = [];
  $("progress-wrap").classList.remove("hidden");
  try {
    for (let index = 0; index < state.files.length; index += 1) {
      $("progress-label").textContent = `Inspecting ${index + 1} of ${state.files.length} · ${state.files[index].name}`;
      $("progress-bar").style.width = `${(index / state.files.length) * 100}%`;
      state.results.push(await inspectFile(state.files[index]));
    }
    $("progress-bar").style.width = "100%";
    $("progress-label").textContent = `${state.results.length} inspection${state.results.length === 1 ? "" : "s"} complete`;
    renderResults();
  } catch (error) {
    showError(error.message);
  } finally {
    inspectButton.disabled = false;
    category.disabled = false;
    fileInput.disabled = false;
  }
});

function renderResults() {
  if (!state.results.length) return;
  $("empty-state").classList.add("hidden");
  $("results").classList.remove("hidden");
  const counts = { normal: 0, review: 0, anomalous: 0 };
  state.results.forEach((result) => { counts[result.decision] += 1; });
  $("batch-summary").textContent = `${counts.normal} normal · ${counts.review} review · ${counts.anomalous} anomalous`;
  $("result-rows").innerHTML = state.results.map((result, index) => `
    <tr data-index="${index}">
      <td>${escapeHtml(result.filename)}</td><td>${escapeHtml(result.category)}</td>
      <td><span class="table-decision ${result.decision}">${result.decision}</span></td>
      <td>${formatNumber(result.anomaly_score)}</td><td>${formatNumber(result.conformal_p)}</td>
      <td>${formatNumber(result.timing_ms.total, 1)} ms</td>
    </tr>`).join("");
  document.querySelectorAll("#result-rows tr").forEach((row) => row.addEventListener("click", () => selectResult(Number(row.dataset.index))));
  selectResult(0);
  window.scrollTo({ top: $("results").offsetTop - 90, behavior: "smooth" });
}

function selectResult(index) {
  state.selected = index;
  const result = state.results[index];
  $("result-name").textContent = result.filename;
  const pill = $("decision-pill");
  pill.textContent = result.decision;
  pill.className = `decision-pill ${result.decision}`;
  $("score").textContent = formatNumber(result.anomaly_score);
  $("score-threshold").textContent = `95% normal score reference ${formatNumber(result.empirical_score_threshold)}`;
  $("p-value").textContent = formatNumber(result.conformal_p);
  $("area").textContent = `${(result.anomalous_area_fraction * 100).toFixed(1)}%`;
  $("runtime").textContent = `${formatNumber(result.timing_ms.total, 1)} ms`;
  $("runtime-detail").textContent = `Features ${formatNumber(result.timing_ms.feature_extraction, 1)} ms · score ${formatNumber(result.timing_ms.memory_scoring, 1)} ms`;
  $("preview").src = result.preview_data_url;
  $("heatmap").src = result.heatmap_data_url;
  $("limitations").innerHTML = result.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  document.querySelectorAll("#result-rows tr").forEach((row) => row.classList.toggle("active", Number(row.dataset.index) === index));
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value);
  return node.innerHTML;
}

$("export").addEventListener("click", async () => {
  clearError();
  try {
    const response = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ results: state.results }),
    });
    if (!response.ok) { const payload = await response.json(); throw new Error(payload.error || "Report export failed."); }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `inspection-report-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) { showError(error.message); }
});

initialize();
