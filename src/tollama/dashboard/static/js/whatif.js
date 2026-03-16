/* What-If scenario explorer for tollama dashboard. */
(function () {
  "use strict";

  let baselineData = null;
  let scenarioData = null;
  let whatifChart = null;

  function getBaseUrl() {
    return window.location.origin;
  }

  async function runForecast(model, series, horizon) {
    const resp = await fetch(getBaseUrl() + "/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, series, horizon }),
    });
    if (!resp.ok) throw new Error("Forecast failed: " + resp.statusText);
    return resp.json();
  }

  async function runWhatIf(model, series, horizon, scenario) {
    const resp = await fetch(getBaseUrl() + "/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, series, horizon, scenario }),
    });
    if (!resp.ok) throw new Error("What-if failed: " + resp.statusText);
    return resp.json();
  }

  function renderChart(baseline, scenario) {
    const ctx = document.getElementById("whatif-chart");
    if (!ctx) return;

    if (whatifChart) whatifChart.destroy();

    const labels = baseline.forecasts[0].mean.map((_, i) => "t+" + (i + 1));
    const datasets = [
      {
        label: "Baseline",
        data: baseline.forecasts[0].mean,
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59,130,246,0.1)",
        fill: true,
        tension: 0.3,
      },
    ];

    if (scenario) {
      const scenarioForecasts = scenario.scenario_forecast || scenario;
      if (scenarioForecasts.forecasts && scenarioForecasts.forecasts[0]) {
        datasets.push({
          label: "Scenario",
          data: scenarioForecasts.forecasts[0].mean,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245,158,11,0.1)",
          fill: true,
          borderDash: [5, 5],
          tension: 0.3,
        });
      }
    }

    if (typeof Chart !== "undefined") {
      whatifChart = new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true,
          plugins: { legend: { position: "top" } },
          scales: {
            y: { title: { display: true, text: "Value" } },
            x: { title: { display: true, text: "Horizon Step" } },
          },
        },
      });
    }
  }

  function init() {
    const form = document.getElementById("whatif-form");
    const scenarioBtn = document.getElementById("whatif-scenario-btn");
    const output = document.getElementById("whatif-output");

    if (!form) return;

    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      const model = form.model.value;
      const series = JSON.parse(form.series.value);
      const horizon = parseInt(form.horizon.value, 10);

      try {
        output.textContent = "Running baseline forecast...";
        baselineData = await runForecast(model, series, horizon);
        output.textContent = JSON.stringify(baselineData, null, 2);
        renderChart(baselineData, null);
      } catch (err) {
        output.textContent = "Error: " + err.message;
      }
    });

    if (scenarioBtn) {
      scenarioBtn.addEventListener("click", async function () {
        const model = form.model.value;
        const series = JSON.parse(form.series.value);
        const horizon = parseInt(form.horizon.value, 10);
        const operation = form.operation.value;
        const factor = parseFloat(form.factor.value);

        const scenario = {
          name: "dashboard_whatif",
          target: "target",
          operation: operation,
          value: factor,
          start_index: 0,
        };

        try {
          output.textContent = "Running scenario...";
          scenarioData = await runWhatIf(model, series, horizon, scenario);
          output.textContent = JSON.stringify(scenarioData, null, 2);
          renderChart(baselineData || scenarioData, scenarioData);
        } catch (err) {
          output.textContent = "Error: " + err.message;
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
