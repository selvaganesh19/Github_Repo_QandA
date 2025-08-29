import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

let client = null;
let connecting = false;

async function ensureClient() {
  if (client || connecting) return client;
  connecting = true;
  try {
    client = await Client.connect("PraneshJs/GithubprojectQandA");
    return client;
  } finally {
    connecting = false;
  }
}

function isValidRepoUrl(url) {
  const pattern = /^https?:\/\/(www\.)?github\.com\/[^\/\s]+\/[^\/\s#?]+\/?$/i;
  return pattern.test(url);
}

function setRunningState(running, message = "") {
  const runBtn = document.getElementById("runBtn");
  const clearBtn = document.getElementById("clearBtn");
  const copyBtn = document.getElementById("copyBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const status = document.getElementById("statusText");
  const spin = document.getElementById("spinner");

  runBtn.disabled = running;
  clearBtn.disabled = running;
  copyBtn.disabled = running;
  downloadBtn.disabled = running;

  if (running) {
    spin.style.display = "inline-block";
    status.textContent = message || "Processing…";
  } else {
    spin.style.display = "none";
    status.textContent = message || "";
  }
}

function saveState() {
  try {
    const url = document.getElementById("repoUrl").value.trim();
    // const topic = document.getElementById("topic").value.trim(); // topic removed
    const n = document.getElementById("numQ").value;
    localStorage.setItem("ghqa:url", url);
    // localStorage.setItem("ghqa:topic", topic);
    localStorage.setItem("ghqa:n", n);
  } catch {}
}

function restoreState() {
  try {
    const url = localStorage.getItem("ghqa:url");
    // const topic = localStorage.getItem("ghqa:topic");
    const n = localStorage.getItem("ghqa:n");

    if (url) document.getElementById("repoUrl").value = url;
    // if (topic) document.getElementById("topic").value = topic;
    if (n) {
      document.getElementById("numQ").value = n;
      document.getElementById("qval").textContent = n;
    }
  } catch {}
}

// Render only Q&A section, color Q and A differently
function renderQAColored(qaText) {
  let qaSection = qaText;
  // Remove everything before Q&A: (case-insensitive)
  const idx = qaSection.search(/Q&A:/i);
  if (idx !== -1) qaSection = qaSection.slice(idx + 4);
  qaSection = qaSection.trim();
  // Color Q and A lines
  return qaSection.split('\n').map(line => {
    if (/^Q\d*:/.test(line.trim())) {
      return `<span class="qa-q">${line}</span>`;
    } else if (/^A\d*:/.test(line.trim())) {
      return `<span class="qa-a">${line}</span>`;
    } else {
      return line;
    }
  }).join('<br>');
}

// Get plain text Q&A for download
function getQAText(qaText) {
  let qaSection = qaText;
  const idx = qaSection.search(/Q&A:/i);
  if (idx !== -1) qaSection = qaSection.slice(idx + 4);
  return qaSection.trim();
}

async function runAnalysis() {
  const urlEl = document.getElementById("repoUrl");
  // const topicEl = document.getElementById("topic"); // topic removed
  const nEl = document.getElementById("numQ");
  const outEl = document.getElementById("output");
  const status = document.getElementById("statusText");
  const errEl = document.getElementById("errorMsg");

  const url = urlEl.value.trim();
  // const topic = topicEl.value.trim(); // topic removed
  const n = Number(nEl.value);

  errEl.textContent = "";
  outEl.textContent = "";

  if (!url) {
    errEl.textContent = "Please enter a GitHub repository URL (required).";
    urlEl.focus();
    return;
  }
  if (!isValidRepoUrl(url)) {
    errEl.textContent = "Please enter a valid GitHub repo URL like https://github.com/owner/repo";
    urlEl.focus();
    return;
  }

  saveState();
  setRunningState(true, "Connecting…");
  outEl.setAttribute("aria-busy", "true");

  try {
    const cli = await ensureClient();
    setRunningState(true, "Analyzing repository…");
    await cli.predict("/on_analyze", { url });

    setRunningState(true, "Generating Q&A…");
    const qa = await cli.predict("/on_generate", { n });

    // Only show Q&A section, colored
    const qaContent = (qa?.data?.[0] ?? "");
    outEl.innerHTML = renderQAColored(qaContent);
    outEl.setAttribute("data-raw", qaContent); // Save for download
    status.textContent = "Done";

  } catch (err) {
    document.getElementById("errorMsg").textContent = "Error: " + (err?.message || err);
  } finally {
    setRunningState(false);
    outEl.setAttribute("aria-busy", "false");
  }
}

function clearAll() {
  document.getElementById("output").textContent = "";
  document.getElementById("output").removeAttribute("data-raw");
  document.getElementById("errorMsg").textContent = "";
  document.getElementById("statusText").textContent = "";
  document.getElementById("repoUrl").focus();
}

async function copyOutput() {
  const outEl = document.getElementById("output");
  const text = outEl.textContent.trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const status = document.getElementById("statusText");
    status.textContent = "Copied to clipboard";
    setTimeout(() => {
      if (status.textContent === "Copied to clipboard") status.textContent = "";
    }, 1200);
  } catch {}
}

function downloadOutput() {
  const outEl = document.getElementById("output");
  const raw = outEl.getAttribute("data-raw") || "";
  const text = getQAText(raw);
  if (!text) return;
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "repo-qa.txt";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 200);
}

window.addEventListener("DOMContentLoaded", () => {
  restoreState();
  // Range value bubble
  const range = document.getElementById("numQ");
  const qval = document.getElementById("qval");
  if (range && qval) {
    range.addEventListener("input", () => { qval.textContent = range.value; });
  }

  // Only add event listeners if elements exist
  const runBtn = document.getElementById("runBtn");
  if (runBtn) runBtn.addEventListener("click", runAnalysis);

  const clearBtn = document.getElementById("clearBtn");
  if (clearBtn) clearBtn.addEventListener("click", clearAll);

  const copyBtn = document.getElementById("copyBtn");
  if (copyBtn) copyBtn.addEventListener("click", copyOutput);

  const downloadBtn = document.getElementById("downloadBtn");
  if (downloadBtn) downloadBtn.addEventListener("click", downloadOutput);

  // Enter submits from text inputs (only if present)
  const repoUrl = document.getElementById("repoUrl");
  if (repoUrl) {
    repoUrl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runAnalysis();
      }
    });
  }
  // topic field removed, so skip
});