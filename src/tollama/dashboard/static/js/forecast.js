(() => {
  let lastResponse = null;
  let forecastCount = 0;

  function setForecastBadge() {
    const badge = document.getElementById("badge-forecasts");
    if (badge) {
      badge.textContent = String(forecastCount);
    }
  }

  function renderFallbackChart(target, values) {
    target.textContent = values.join(", ");
  }

  function resolveSelectedModel() {
    const select = document.getElementById("forecast-model-select");
    const customInput = document.getElementById("forecast-model-custom");
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

  function downloadBlob(filename, mimeType, content) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  function responseToCsv(payload) {
    const forecast = payload?.forecasts?.[0] || {};
    const mean = Array.isArray(forecast.mean) ? forecast.mean : [];
    const timestamps = Array.isArray(forecast.timestamps) ? forecast.timestamps : [];
    const rows = ["step,timestamp,mean"];
    mean.forEach((value, index) => {
      const timestamp = timestamps[index] || "";
      rows.push(`${index + 1},${timestamp},${value}`);
    });
    return rows.join("\n");
  }

  function bindExportButtons() {
    const csvButton = document.getElementById("forecast-export-csv");
    const jsonButton = document.getElementById("forecast-export-json");
    const pngButton = document.getElementById("forecast-export-png");
    const copyButton = document.getElementById("forecast-copy-json");

    csvButton?.addEventListener("click", () => {
      if (!lastResponse) {
        return;
      }
      downloadBlob("forecast.csv", "text/csv;charset=utf-8", responseToCsv(lastResponse));
    });

    jsonButton?.addEventListener("click", () => {
      if (!lastResponse) {
        return;
      }
      downloadBlob(
        "forecast.json",
        "application/json;charset=utf-8",
        JSON.stringify(lastResponse, null, 2),
      );
    });

    pngButton?.addEventListener("click", () => {
      const canvas = document.getElementById("forecast-chart");
      if (!(canvas instanceof HTMLCanvasElement)) {
        return;
      }
      const url = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = url;
      link.download = "forecast.png";
      link.click();
    });

    copyButton?.addEventListener("click", async () => {
      if (!lastResponse || !navigator.clipboard) {
        return;
      }
      await navigator.clipboard.writeText(JSON.stringify(lastResponse, null, 2));
    });
  }

  async function submitForecast(fetchJson) {
    const model = resolveSelectedModel();
    const horizon = Number(document.getElementById("forecast-horizon")?.value || "0");
    const raw = document.getElementById("forecast-series")?.value || "[]";
    const output = document.getElementById("forecast-result-json");
    if (!model || horizon <= 0 || !output) {
      return;
    }

    let series;
    try {
      series = JSON.parse(raw);
    } catch {
      output.textContent = "Invalid JSON in series input.";
      return;
    }

    try {
      const payload = {
        model,
        horizon,
        series,
        options: {},
        stream: false,
      };
      const response = await fetchJson("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      lastResponse = response;
      forecastCount += 1;
      setForecastBadge();
      output.textContent = JSON.stringify(response, null, 2);

      const first = response.forecasts?.[0]?.mean || [];
      const chartHost = document.getElementById("forecast-chart");
      if (chartHost) {
        if (window.Chart && chartHost instanceof HTMLCanvasElement) {
          const labels = first.map((_, index) => String(index + 1));
          if (window.__forecastChart && typeof window.__forecastChart.destroy === "function") {
            window.__forecastChart.destroy();
          }
          window.__forecastChart = new window.Chart(chartHost.getContext("2d"), {
            type: "line",
            data: {
              labels,
              datasets: [
                {
                  label: "Forecast mean",
                  data: first,
                  borderColor: "#2fc5a2",
                  fill: false,
                },
              ],
            },
            options: { responsive: true, maintainAspectRatio: false },
          });
        } else {
          renderFallbackChart(output, first);
        }
      }
    } catch (error) {
      const raw = String(error?.message || error || "unknown error");
      let friendly = `Forecast failed: ${raw}`;
      if (
        raw.includes("input series length is shorter than model context_length") ||
        raw.includes("context_length")
      ) {
        friendly = `${friendly}\n\nTip: the selected model needs more history points. ` +
          `Use model 'mock' for quick demos, or provide a longer target series (>= required context length).`;
      }
      output.textContent = friendly;
    }
  }

  function init({ fetchJson }) {
    const form = document.getElementById("forecast-form");
    if (!form) {
      return;
    }
    bindExportButtons();
    setForecastBadge();
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitForecast(fetchJson);
    });
  }

  window.TollamaForecastPlayground = { init };
})();
