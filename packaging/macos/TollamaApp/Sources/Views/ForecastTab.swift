import SwiftUI

struct ForecastTab: View {
    @EnvironmentObject private var workspace: ForecastWorkspace
    let client: TollamaHTTPClient

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Forecast")
                    .font(.largeTitle.bold())
                Spacer()
                Button("Install more…") {
                    workspace.selectedTab = .models
                }
            }

            form
            Divider()
            resultPane
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .task {
            if workspace.models.isEmpty {
                await workspace.refreshModels(client: client)
            }
        }
    }

    private var form: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .bottom, spacing: 16) {
                VStack(alignment: .leading) {
                    Text("Model")
                        .font(.headline)
                    Picker("Model", selection: selectedModelBinding) {
                        if workspace.installedModels.isEmpty {
                            Text("No installed models").tag("")
                        }
                        ForEach(workspace.installedModels) { model in
                            Text(model.name).tag(model.name)
                        }
                    }
                    .frame(width: 280)
                }

                VStack(alignment: .leading) {
                    Text("Horizon")
                        .font(.headline)
                    Stepper(
                        String(workspace.horizon),
                        value: $workspace.horizon,
                        in: 1...(workspace.selectedModel?.maxHorizon ?? 10_000)
                    )
                    .frame(width: 140)
                }

                VStack(alignment: .leading) {
                    Text("Quantiles")
                        .font(.headline)
                    HStack {
                        quantileToggle(0.1, label: "q10")
                        quantileToggle(0.5, label: "q50")
                        quantileToggle(0.9, label: "q90")
                    }
                }

                Spacer()

                Button {
                    Task { await workspace.runForecast(client: client) }
                } label: {
                    if workspace.isRunningForecast {
                        ProgressView()
                    } else {
                        Text("Run Forecast")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(workspace.isRunningForecast || workspace.selectedCSV == nil || workspace.selectedModelName == nil)
            }

            if let selectedCSV = workspace.selectedCSV {
                Text("CSV: \(selectedCSV.name)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Choose a CSV in the Data tab before running a forecast.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            if let maxHorizon = workspace.selectedModel?.maxHorizon {
                Text("Selected model max horizon: \(maxHorizon)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var resultPane: some View {
        if let response = workspace.lastResponse, let forecast = response.forecasts.first {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Text("Result: \(response.model)")
                            .font(.title2.bold())
                        Spacer()
                        if !workspace.recentForecasts.isEmpty {
                            Menu("Recent") {
                                ForEach(workspace.recentForecasts) { item in
                                    Button("\(item.model) · \(item.forecasts.first?.id ?? "series")") {
                                        workspace.selectRecentForecast(item)
                                    }
                                }
                            }
                        }
                    }

                    if let warnings = response.warnings, !warnings.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Warnings")
                                .font(.headline)
                            ForEach(warnings, id: \.self) { warning in
                                Text(warning)
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                        }
                    }

                    ForecastChart(history: workspace.historyPoints, forecast: forecast)
                    ForecastTable(forecast: forecast)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.bottom, 24)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                Text("No forecast yet")
                    .font(.title2.bold())
                Text("Select an installed model and CSV, then run a forecast.")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
    }

    private var selectedModelBinding: Binding<String> {
        Binding(
            get: { workspace.selectedModelName ?? "" },
            set: { value in
                workspace.selectedModelName = value.isEmpty ? nil : value
                workspace.clampHorizonToSelectedModel()
            }
        )
    }

    private func quantileToggle(_ value: Double, label: String) -> some View {
        Toggle(
            label,
            isOn: Binding(
                get: { workspace.selectedQuantiles.contains(value) },
                set: { enabled in
                    if enabled {
                        workspace.selectedQuantiles.insert(value)
                    } else {
                        workspace.selectedQuantiles.remove(value)
                    }
                }
            )
        )
        .toggleStyle(.checkbox)
    }
}
