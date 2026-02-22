(() => {
  const KEY_STORAGE = "tollama.dashboard.api_key";
  const THEME_STORAGE = "tollama.dashboard.theme";
  const state = {
    apiKey: sessionStorage.getItem(KEY_STORAGE) || "",
    eventAbort: null,
    eventCount: 0,
    installedModelNames: [],
    installPromptEvent: null,
  };

  function detailToText(detail) {
    if (typeof detail === "string") {
      return detail.trim();
    }
    if (typeof detail === "number" || typeof detail === "boolean") {
      return String(detail);
    }
    if (detail == null) {
      return "";
    }
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }

  async function buildApiError(response) {
    let detail = "";
    let hint = "";
    const body = await response.text();
    if (body) {
      try {
        const payload = JSON.parse(body);
        if (payload && typeof payload === "object") {
          detail = detailToText(payload.detail);
          hint = detailToText(payload.hint);
        } else {
          detail = detailToText(payload);
        }
      } catch {
        detail = body.trim();
      }
    }
    if (!detail) {
      detail = `HTTP ${response.status}`;
    }
    const error = new Error(detail);
    error.status = response.status;
    error.detail = detail;
    if (hint) {
      error.hint = hint;
    }
    return error;
  }

  function formatApiError(error, prefix) {
    const detail = error?.detail || error?.message || String(error || "unknown error");
    const hint = error?.hint ? ` Hint: ${error.hint}` : "";
    return `${prefix}: ${detail}${hint}`;
  }

  function isLicenseRequiredError(error) {
    if (!error || error.status !== 409) {
      return false;
    }
    const text = `${error.detail || ""} ${error.hint || ""} ${error.message || ""}`.toLowerCase();
    return text.includes("license") || text.includes("accept_license") || text.includes("accept-license");
  }

  function authHeaders(extra = {}) {
    const headers = { ...extra };
    if (state.apiKey) {
      headers.Authorization = `Bearer ${state.apiKey}`;
    }
    return headers;
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: authHeaders({
        Accept: "application/json",
        ...(options.headers || {}),
      }),
    });
    if (!response.ok) {
      throw await buildApiError(response);
    }
    return response.json();
  }

  async function fetchText(url) {
    const response = await fetch(url, { headers: authHeaders({ Accept: "text/html" }) });
    if (!response.ok) {
      throw await buildApiError(response);
    }
    return response.text();
  }

  function setConnection(ok) {
    const dot = document.getElementById("connection-dot");
    if (!dot) {
      return;
    }
    dot.classList.toggle("status-connected", ok);
    dot.classList.toggle("status-disconnected", !ok);
    dot.title = ok ? "Connected" : "Disconnected";
  }

  function setDaemonMeta(text) {
    const node = document.getElementById("daemon-meta");
    if (node) {
      node.textContent = text;
    }
  }

  async function loadPartials() {
    const targets = {
      "models-table-panel": "models-table.html",
      "loaded-models-panel": "loaded-models.html",
      "event-log-panel": "event-log.html",
      "usage-stats-panel": "usage-stats.html",
      "forecast-form-panel": "forecast-form.html",
      "forecast-result-panel": "forecast-result.html",
      "model-detail-panel": "model-detail.html",
      "model-actions-panel": "model-actions.html",
    };
    await Promise.all(
      Object.entries(targets).map(async ([id, partial]) => {
        const root = document.getElementById(id);
        if (!root) {
          return;
        }
        try {
          root.innerHTML = await fetchText(`/dashboard/static/partials/${partial}`);
        } catch {
          root.innerHTML = `<p>Unable to load ${partial}</p>`;
        }
      }),
    );
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem(THEME_STORAGE, next);
  }

  function applyTheme() {
    const stored = localStorage.getItem(THEME_STORAGE);
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
  }

  function initTabs() {
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const panels = Array.from(document.querySelectorAll(".panel"));
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.getAttribute("data-tab");
        tabs.forEach((item) => item.classList.toggle("active", item === tab));
        panels.forEach((panel) => panel.classList.toggle("active", panel.getAttribute("data-panel") === key));
      });
    });
  }

  function normalizeModelNames(models) {
    const names = [];
    const seen = new Set();
    models.forEach((item) => {
      const name = String(item?.name || item?.model || "").trim();
      if (!name || seen.has(name)) {
        return;
      }
      seen.add(name);
      names.push(name);
    });
    names.sort((left, right) => left.localeCompare(right));
    return names;
  }

  function ensureModelSelectorWiring(selectId, customInputId) {
    const select = document.getElementById(selectId);
    const customInput = document.getElementById(customInputId);
    if (!(select instanceof HTMLSelectElement) || !(customInput instanceof HTMLInputElement)) {
      return;
    }
    if (select.dataset.modelWired === "1") {
      return;
    }
    const syncVisibility = () => {
      const isCustom = select.value === "__custom__";
      customInput.classList.toggle("hidden", !isCustom);
      if (isCustom) {
        customInput.focus();
      }
    };
    select.addEventListener("change", syncVisibility);
    syncVisibility();
    select.dataset.modelWired = "1";
  }

  function populateModelSelectorOptions(selectId, customInputId, modelNames, preferredName = "") {
    const select = document.getElementById(selectId);
    const customInput = document.getElementById(customInputId);
    if (!(select instanceof HTMLSelectElement)) {
      return;
    }

    const previous = select.value;
    const customValue = customInput instanceof HTMLInputElement ? customInput.value.trim() : "";
    const preferCustom = previous === "__custom__" || customValue.length > 0;

    select.replaceChildren();
    const appendOption = (value, label, selected = false) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = selected;
      select.append(option);
    };

    if (modelNames.length === 0) {
      appendOption("", "No installed models", true);
    } else {
      modelNames.forEach((name) => {
        appendOption(name, name);
      });
    }
    appendOption("__custom__", "Custom...");

    if (preferCustom) {
      select.value = "__custom__";
    } else if (modelNames.includes(previous)) {
      select.value = previous;
    } else if (preferredName && modelNames.includes(preferredName)) {
      select.value = preferredName;
    } else if (modelNames.length > 0) {
      select.value = modelNames[0];
    } else {
      select.value = "";
    }

    ensureModelSelectorWiring(selectId, customInputId);
    if (select.value === "__custom__" && customInput instanceof HTMLInputElement) {
      customInput.classList.remove("hidden");
    }
  }

  function syncModelSelectors(models) {
    const modelNames = normalizeModelNames(models);
    state.installedModelNames = modelNames;
    const selectors = [
      ["forecast-model-select", "forecast-model-custom", "mock"],
      ["model-detail-name-select", "model-detail-name-custom", "mock"],
      ["model-action-name-select", "model-action-name-custom", "mock"],
    ];
    selectors.forEach(([selectId, customInputId, preferredName]) => {
      populateModelSelectorOptions(selectId, customInputId, modelNames, preferredName);
    });
  }

  function resolveSelectedModel(selectId, customInputId) {
    const select = document.getElementById(selectId);
    const customInput = document.getElementById(customInputId);
    if (select instanceof HTMLSelectElement) {
      if (select.value && select.value !== "__custom__") {
        return select.value.trim();
      }
    }
    if (customInput instanceof HTMLInputElement) {
      return customInput.value.trim();
    }
    return "";
  }

  async function refreshState() {
    try {
      const payload = await fetchJson("/api/dashboard/state");
      setConnection(true);
      const daemon = payload.info?.daemon || {};
      setDaemonMeta(`v${daemon.version || "?"} | uptime ${daemon.uptime_seconds || 0}s`);
      const models = payload.ps?.models || [];
      const badge = document.getElementById("badge-models");
      if (badge) {
        badge.textContent = String(models.length);
      }

      const modelsTable = document.getElementById("models-table-body");
      let tagsModels = [];
      try {
        const tags = await fetchJson("/api/tags");
        tagsModels = Array.isArray(tags.models) ? tags.models : [];
      } catch (error) {
        tagsModels = [];
        const warningNode = document.getElementById("dashboard-warnings");
        if (warningNode) {
          warningNode.textContent = formatApiError(error, "Model list refresh failed");
        }
      }
      syncModelSelectors(tagsModels);
      if (modelsTable) {
        modelsTable.innerHTML = tagsModels
          .map((item) => `<tr><td>${item.name || ""}</td><td>${item.details?.family || ""}</td><td>${item.size || 0}</td></tr>`)
          .join("");
      }

      const loadedTable = document.getElementById("loaded-models-body");
      if (loadedTable) {
        loadedTable.innerHTML = models
          .map((item) => `<tr><td>${item.model || item.name || ""}</td><td>${item.details?.family || ""}</td><td>${item.expires_at || "-"}</td></tr>`)
          .join("");
      }

      const usageTarget = document.getElementById("usage-summary");
      if (usageTarget) {
        const usage = payload.usage?.summary || {};
        usageTarget.textContent = JSON.stringify(usage, null, 2);
      }

      const warningNode = document.getElementById("dashboard-warnings");
      if (warningNode) {
        const lines = (payload.warnings || []).map((item) => `[${item.source}] ${item.status_code}: ${item.detail}`);
        warningNode.textContent = lines.length ? lines.join("\n") : "No warnings.";
      }
    } catch (error) {
      setConnection(false);
      if (error && error.status === 401) {
        promptForApiKey();
        return;
      }
      setDaemonMeta(formatApiError(error, "Dashboard refresh failed"));
    }
  }

  async function streamEvents() {
    if (state.eventAbort) {
      state.eventAbort.abort();
    }
    state.eventAbort = new AbortController();
    const log = document.getElementById("event-log");
    if (!log) {
      return;
    }

    try {
      const response = await fetch("/api/events?heartbeat=15", {
        method: "GET",
        headers: authHeaders({ Accept: "text/event-stream" }),
        signal: state.eventAbort.signal,
      });
      if (!response.ok || !response.body) {
        throw await buildApiError(response);
      }
      const decoder = new TextDecoder();
      const reader = response.body.getReader();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        for (const chunk of chunks) {
          const lines = chunk.split("\n");
          let eventName = "message";
          let data = "";
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            }
            if (line.startsWith("data:")) {
              data += line.slice(5).trim();
            }
          }
          if (!data) {
            continue;
          }
          const row = document.createElement("div");
          row.textContent = `${new Date().toLocaleTimeString()} ${eventName}: ${data}`;
          log.prepend(row);
          state.eventCount += 1;
          const badge = document.getElementById("badge-events");
          if (badge) {
            badge.textContent = String(state.eventCount);
          }
          while (log.children.length > 200) {
            log.removeChild(log.lastChild);
          }
        }
      }
    } catch (error) {
      setConnection(false);
      if (error && error.status === 401) {
        promptForApiKey();
      } else if (log) {
        const row = document.createElement("div");
        row.textContent = formatApiError(error, "event stream failed");
        log.prepend(row);
      }
      window.setTimeout(streamEvents, 3000);
    }
  }

  function promptForApiKey() {
    const dialog = document.getElementById("auth-dialog");
    const form = document.getElementById("auth-form");
    const input = document.getElementById("api-key-input");
    if (!(dialog instanceof HTMLDialogElement) || !form || !(input instanceof HTMLInputElement)) {
      return;
    }
    if (!dialog.open) {
      dialog.showModal();
    }

    form.addEventListener(
      "submit",
      (event) => {
        event.preventDefault();
        const value = input.value.trim();
        if (!value) {
          return;
        }
        state.apiKey = value;
        sessionStorage.setItem(KEY_STORAGE, value);
        dialog.close();
        refreshState();
        streamEvents();
      },
      { once: true },
    );
  }

  function initInstallBanner() {
    const banner = document.getElementById("install-banner");
    const action = document.getElementById("install-action");
    const dismiss = document.getElementById("install-dismiss");
    if (!banner || !action || !dismiss) {
      return;
    }

    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      state.installPromptEvent = event;
      banner.classList.remove("hidden");
    });

    action.addEventListener("click", async () => {
      if (!state.installPromptEvent) {
        return;
      }
      await state.installPromptEvent.prompt();
      banner.classList.add("hidden");
      state.installPromptEvent = null;
    });

    dismiss.addEventListener("click", () => {
      banner.classList.add("hidden");
    });
  }

  function initShortcuts() {
    document.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "f" || key === "m" || key === "?") {
        event.preventDefault();
      }
      if (key === "f") {
        document.querySelector('.tab[data-tab="forecast"]')?.click();
      }
      if (key === "m") {
        document.querySelector('.tab[data-tab="models"]')?.click();
      }
      if (key === "?") {
        document.querySelector('.tab[data-tab="help"]')?.click();
      }
    });
  }

  function initTopbarActions() {
    document.getElementById("theme-toggle")?.addEventListener("click", toggleTheme);
    document.getElementById("api-docs-link")?.addEventListener("click", () => {
      window.open("/docs", "_blank", "noopener,noreferrer");
    });
  }

  function initModelActions(fetchJson) {
    const modelSelect = document.getElementById("model-detail-name-select");
    const loadButton = document.getElementById("model-detail-load");
    const modelOutput = document.getElementById("model-detail-output");
    const actionSelect = document.getElementById("model-action-name-select");
    const pullButton = document.getElementById("model-action-pull");
    const deleteButton = document.getElementById("model-action-delete");
    const acceptLicense = document.getElementById("model-action-accept-license");
    const actionOutput = document.getElementById("model-actions-output");

    if (loadButton && modelOutput && modelSelect) {
      loadButton.addEventListener("click", async () => {
        const model = resolveSelectedModel("model-detail-name-select", "model-detail-name-custom");
        if (!model) {
          modelOutput.textContent = "Provide a model name.";
          return;
        }
        try {
          const payload = await fetchJson("/api/show", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model }),
          });
          modelOutput.textContent = JSON.stringify(payload, null, 2);
        } catch (error) {
          modelOutput.textContent = formatApiError(error, "Show failed");
        }
      });
    }

    async function streamPull(model, acceptLicenseForPull) {
      if (!actionOutput) {
        return;
      }
      actionOutput.textContent = "";
      const response = await fetch("/api/pull", {
        method: "POST",
        headers: authHeaders({
          "Content-Type": "application/json",
          Accept: "application/x-ndjson",
        }),
        body: JSON.stringify({
          model,
          stream: true,
          accept_license: Boolean(acceptLicenseForPull),
        }),
      });
      if (!response.ok || !response.body) {
        throw await buildApiError(response);
      }
      const decoder = new TextDecoder();
      const reader = response.body.getReader();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) {
            continue;
          }
          actionOutput.textContent = `${actionOutput.textContent}${trimmed}\n`;
        }
      }
    }

    if (pullButton && actionSelect && actionOutput) {
      pullButton.addEventListener("click", async () => {
        const model = resolveSelectedModel("model-action-name-select", "model-action-name-custom");
        if (!model) {
          actionOutput.textContent = "Provide a model name.";
          return;
        }
        try {
          const acceptSelected = Boolean(acceptLicense instanceof HTMLInputElement && acceptLicense.checked);
          await streamPull(model, acceptSelected);
        } catch (error) {
          const acceptSelected = Boolean(acceptLicense instanceof HTMLInputElement && acceptLicense.checked);
          if (!acceptSelected && isLicenseRequiredError(error)) {
            const confirmed = window.confirm(
              "This model requires license acceptance. Retry with license acceptance enabled?",
            );
            if (confirmed) {
              if (acceptLicense instanceof HTMLInputElement) {
                acceptLicense.checked = true;
              }
              try {
                await streamPull(model, true);
                return;
              } catch (retryError) {
                actionOutput.textContent = formatApiError(retryError, "Pull failed");
                return;
              }
            }
          }
          actionOutput.textContent = formatApiError(error, "Pull failed");
        }
      });
    }

    if (deleteButton && actionSelect && actionOutput) {
      deleteButton.addEventListener("click", async () => {
        const model = resolveSelectedModel("model-action-name-select", "model-action-name-custom");
        if (!model) {
          actionOutput.textContent = "Provide a model name.";
          return;
        }
        if (!window.confirm(`Delete installed model '${model}'?`)) {
          return;
        }
        try {
          const payload = await fetchJson("/api/delete", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model }),
          });
          actionOutput.textContent = JSON.stringify(payload, null, 2);
        } catch (error) {
          actionOutput.textContent = formatApiError(error, "Delete failed");
        }
      });
    }
  }

  async function bootstrap() {
    applyTheme();
    initTabs();
    initShortcuts();
    initTopbarActions();
    initInstallBanner();
    await loadPartials();
    syncModelSelectors(state.installedModelNames);
    if (window.TollamaForecastPlayground && typeof window.TollamaForecastPlayground.init === "function") {
      window.TollamaForecastPlayground.init({ fetchJson, formatApiError });
    }
    if (window.TollamaComparison && typeof window.TollamaComparison.init === "function") {
      window.TollamaComparison.init({ fetchJson, formatApiError });
    }
    initModelActions(fetchJson);
    await refreshState();
    await streamEvents();
    window.setInterval(refreshState, 10000);

    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/dashboard/static/sw.js").catch(() => {
        /* ignore registration failures */
      });
    }
  }

  bootstrap();
})();
