(() => {
  async function handleCompare(fetchJson) {
    const form = document.getElementById("compare-form");
    const output = document.getElementById("compare-output");
    const canvas = document.getElementById("compare-chart");
    if (!form || !output) {
      return;
    }

    const modelsRaw = document.getElementById("compare-models")?.value || "";
    const horizon = Number(document.getElementById("compare-horizon")?.value || "0");
    const seriesRaw = document.getElementById("compare-series")?.value || "[]";

    const models = modelsRaw
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    let series;
    try {
      series = JSON.parse(seriesRaw);
    } catch {
      output.textContent = "Series JSON is invalid.";
      return;
    }

    try {
      const payload = {
        models,
        horizon,
        series,
        options: {},
      };
      const response = await fetchJson("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      output.textContent = JSON.stringify(response, null, 2);

      if (window.Chart && canvas instanceof HTMLCanvasElement) {
        const successful = (response.results || []).filter((item) => item.ok && item.response);
        const labels = successful[0]?.response?.forecasts?.[0]?.mean?.map((_, index) => String(index + 1)) || [];
        const datasets = successful.map((item, idx) => ({
          label: item.model,
          data: item.response.forecasts[0]?.mean || [],
          borderColor: ["#2fc5a2", "#3d8eff", "#f6c667", "#f66b6b"][idx % 4],
          fill: false,
        }));

        if (window.__compareChart && typeof window.__compareChart.destroy === "function") {
          window.__compareChart.destroy();
        }
        window.__compareChart = new window.Chart(canvas.getContext("2d"), {
          type: "line",
          data: { labels, datasets },
          options: { responsive: true, maintainAspectRatio: false },
        });
      }
    } catch (error) {
      output.textContent = `Comparison failed: ${error.message || error}`;
    }
  }

  function init({ fetchJson }) {
    const form = document.getElementById("compare-form");
    if (!form) {
      return;
    }
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      handleCompare(fetchJson);
    });
  }

  window.TollamaComparison = { init };
})();
