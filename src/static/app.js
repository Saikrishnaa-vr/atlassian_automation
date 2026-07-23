const els = {
  apiBase: document.getElementById("api-base"),
  apiDot: document.getElementById("api-dot"),
  apiStatus: document.getElementById("api-status"),
  requirementForm: document.getElementById("requirement-form"),
  submitBtn: document.getElementById("submit-btn"),
  projectKey: document.getElementById("project-key"),
  requirement: document.getElementById("requirement"),
  workflowId: document.getElementById("workflow-id"),
  refreshBtn: document.getElementById("refresh-btn"),
  approvalButtons: Array.from(document.querySelectorAll("[data-approval]")),
  workflowSummary: document.getElementById("workflow-summary"),
  output: document.getElementById("output"),
  copyJson: document.getElementById("copy-json"),
};

let lastPayload = null;

function resolveBaseUrl() {
  const raw = els.apiBase.value.trim();
  if (!raw) {
    return window.location.origin;
  }
  return raw.replace(/\/$/, "");
}

function setBusy(button, busy) {
  button.disabled = busy;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Working...";
  } else if (button.dataset.originalText) {
    button.textContent = button.dataset.originalText;
  }
}

function setOutput(payload) {
  lastPayload = payload;
  els.output.textContent = JSON.stringify(payload, null, 2);
}

function renderSummary(state) {
  const stories = state?.generated_stories?.stories || [];
  const jiraResults = state?.jira_results || [];

  const links = jiraResults
    .map((issue) => {
      const key = issue?.key || "(unknown)";
      const url = issue?.url;
      return url
        ? `<li><a href="${url}" target="_blank" rel="noreferrer">${key}</a></li>`
        : `<li>${key}</li>`;
    })
    .join("");

  els.workflowSummary.innerHTML = `
    <p><strong>Step:</strong> ${state?.current_step || "n/a"}</p>
    <p><strong>Approval:</strong> ${state?.approval_status || "n/a"}</p>
    <p><strong>Stories:</strong> ${stories.length}</p>
    <p><strong>Jira Issues:</strong> ${jiraResults.length}</p>
    ${links ? `<ul>${links}</ul>` : ""}
  `;
}

async function apiCall(path, options = {}) {
  const base = resolveBaseUrl();
  const url = `${base}${path}`;
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = { detail: "No JSON response body" };
  }

  if (!response.ok) {
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }

  return body;
}

async function checkApiHealth() {
  try {
    await apiCall("/health", { method: "GET" });
    els.apiDot.classList.remove("fail");
    els.apiDot.classList.add("ok");
    els.apiStatus.textContent = `API status: connected (${resolveBaseUrl()})`;
  } catch (error) {
    els.apiDot.classList.remove("ok");
    els.apiDot.classList.add("fail");
    els.apiStatus.textContent = `API status: ${error.message}`;
  }
}

els.requirementForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(els.submitBtn, true);

  try {
    const payload = {
      requirement: els.requirement.value.trim(),
      project_key: els.projectKey.value.trim(),
    };

    const data = await apiCall("/requirements", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    els.workflowId.value = data.workflow_id;
    setOutput(data);
    renderSummary(data);
    await checkApiHealth();
  } catch (error) {
    setOutput({ error: error.message });
  } finally {
    setBusy(els.submitBtn, false);
  }
});

els.refreshBtn.addEventListener("click", async () => {
  const workflowId = els.workflowId.value.trim();
  if (!workflowId) {
    setOutput({ error: "Enter a workflow ID first." });
    return;
  }

  setBusy(els.refreshBtn, true);
  try {
    const data = await apiCall(`/requirements/${workflowId}`, { method: "GET" });
    setOutput(data);
    renderSummary(data);
  } catch (error) {
    setOutput({ error: error.message });
  } finally {
    setBusy(els.refreshBtn, false);
  }
});

els.approvalButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const workflowId = els.workflowId.value.trim();
    if (!workflowId) {
      setOutput({ error: "Enter a workflow ID first." });
      return;
    }

    const approval = button.dataset.approval;
    setBusy(button, true);

    try {
      const data = await apiCall(`/requirements/${workflowId}/approval`, {
        method: "POST",
        body: JSON.stringify({ approval }),
      });

      setOutput(data);
      renderSummary(data);
    } catch (error) {
      setOutput({ error: error.message });
    } finally {
      setBusy(button, false);
    }
  });
});

els.copyJson.addEventListener("click", async () => {
  if (!lastPayload) {
    return;
  }

  try {
    await navigator.clipboard.writeText(JSON.stringify(lastPayload, null, 2));
    const original = els.copyJson.textContent;
    els.copyJson.textContent = "Copied";
    setTimeout(() => {
      els.copyJson.textContent = original;
    }, 800);
  } catch (error) {
    setOutput({ error: `Could not copy JSON: ${error.message}` });
  }
});

els.apiBase.addEventListener("change", checkApiHealth);
checkApiHealth();
