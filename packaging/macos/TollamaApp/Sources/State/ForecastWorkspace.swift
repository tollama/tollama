import Foundation

enum WorkspaceTab: String, Hashable {
    case models
    case data
    case forecast
    case logs
}

enum MissingValuePreprocessingMode: String, CaseIterable, Identifiable {
    case off
    case auto
    case bspline
    case linear
    case seasonal

    var id: String { rawValue }

    var label: String {
        switch self {
        case .off:
            return "Off"
        case .auto:
            return "Auto"
        case .bspline:
            return "B-spline"
        case .linear:
            return "Linear"
        case .seasonal:
            return "Seasonal"
        }
    }

    var uploadMethod: String? {
        self == .off ? nil : rawValue
    }
}

@MainActor
final class ForecastWorkspace: ObservableObject {
    @Published var selectedTab: WorkspaceTab = .models
    @Published private(set) var models: [RegistryModel] = []
    @Published private(set) var csvFiles: [CSVFileItem] = []
    @Published private(set) var csvPreview: CSVPreview?
    @Published private(set) var csvPreviewError: String?
    @Published private(set) var selectedFolder: URL?
    @Published var selectedCSV: CSVFileItem?
    @Published var selectedModelName: String?
    @Published var horizon: Int = 10
    @Published var selectedQuantiles: Set<Double> = [0.1, 0.9]
    @Published var timestampColumnOverride = ""
    @Published var seriesIDColumnOverride = ""
    @Published var targetColumnOverride = ""
    @Published var frequencyOverride = ""
    @Published var freqColumnOverride = ""
    @Published var missingValueMode: MissingValuePreprocessingMode = .off
    @Published private(set) var lastResponse: ForecastResponseDTO?
    @Published private(set) var recentForecasts: [ForecastResponseDTO] = []
    @Published private(set) var isLoadingModels = false
    @Published private(set) var isScanningFolder = false
    @Published private(set) var isLoadingCSVPreview = false
    @Published private(set) var isRunningForecast = false
    @Published var banner: ActionBanner?

    var installedModels: [RegistryModel] {
        models
            .filter(\.installed)
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    var installedForecastModels: [RegistryModel] {
        installedModels.filter(\.forecastReady)
    }

    var selectedModel: RegistryModel? {
        guard let selectedModelName else {
            return nil
        }
        return models.first { $0.name == selectedModelName }
    }

    var selectedQuantilesSorted: [Double] {
        selectedQuantiles.sorted()
    }

    var historyPoints: [ForecastHistoryPoint] {
        csvPreview?.history ?? []
    }

    func refreshModels(client: TollamaHTTPClient) async {
        isLoadingModels = true
        defer {
            isLoadingModels = false
        }
        do {
            let response = try await client.listModels()
            models = response.available
            if selectedModelName == nil
                || !installedForecastModels.contains(where: { $0.name == selectedModelName })
            {
                selectedModelName = installedForecastModels.first?.name
            }
            clampHorizonToSelectedModel()
        } catch {
            banner = ActionBanner(title: "Unable to load models", detail: error.localizedDescription)
        }
    }

    func scanFolder(_ folder: URL) async {
        selectedFolder = folder
        isScanningFolder = true
        defer {
            isScanningFolder = false
        }
        do {
            let files = try await Task.detached(priority: .userInitiated) {
                try CSVSniffer.scanCSVFiles(in: folder)
            }.value
            csvFiles = files
            if let first = files.first {
                await selectCSV(first)
            } else {
                selectedCSV = nil
                csvPreview = nil
                csvPreviewError = nil
            }
        } catch {
            banner = ActionBanner(title: "Unable to scan folder", detail: error.localizedDescription)
        }
    }

    func selectCSV(_ file: CSVFileItem) async {
        selectedCSV = file
        csvPreview = nil
        csvPreviewError = nil
        isLoadingCSVPreview = true
        defer {
            isLoadingCSVPreview = false
        }
        do {
            let preview = try await Task.detached(priority: .userInitiated) {
                try CSVSniffer.preview(url: file.url)
            }.value
            csvPreview = preview
            timestampColumnOverride = preview.timestampColumn ?? ""
            seriesIDColumnOverride = preview.seriesIDColumn ?? ""
            targetColumnOverride = preview.targetColumn ?? ""
            frequencyOverride = preview.freqColumn == nil ? preview.inferredFrequency ?? "" : ""
            freqColumnOverride = preview.freqColumn ?? ""
            missingValueMode = .off
        } catch {
            csvPreview = nil
            csvPreviewError = error.localizedDescription
            banner = ActionBanner(title: "Unable to preview CSV", detail: error.localizedDescription)
        }
    }

    func clampHorizonToSelectedModel() {
        guard let maxHorizon = selectedModel?.maxHorizon else {
            return
        }
        horizon = min(max(1, horizon), maxHorizon)
    }

    func selectRecentForecast(_ response: ForecastResponseDTO) {
        lastResponse = response
    }

    func runForecast(client: TollamaHTTPClient) async {
        guard let selectedCSV else {
            banner = ActionBanner(title: "No CSV selected", detail: "Choose a CSV file in the Data tab.")
            selectedTab = .data
            return
        }
        guard let selectedModelName, !selectedModelName.isEmpty else {
            banner = ActionBanner(title: "No model selected", detail: "Install and select a model first.")
            selectedTab = .models
            return
        }
        guard selectedModel?.forecastReady != false else {
            banner = ActionBanner(
                title: "Model is manifest-only",
                detail: "Select a forecast-ready model such as timer-base or sundial-base-128m."
            )
            selectedTab = .models
            return
        }

        isRunningForecast = true
        defer {
            isRunningForecast = false
        }

        do {
            clampHorizonToSelectedModel()
            let response = try await client.forecastLocalCSV(
                fileURL: selectedCSV.url,
                model: selectedModelName,
                horizon: horizon,
                quantiles: selectedQuantilesSorted,
                timestampColumn: optionalOverride(timestampColumnOverride),
                seriesIDColumn: optionalOverride(seriesIDColumnOverride),
                targetColumn: optionalOverride(targetColumnOverride),
                freq: optionalFrequencyOverride(frequencyOverride),
                freqColumn: optionalOverride(freqColumnOverride),
                preprocessMissing: missingValueMode != .off,
                missingMethod: missingValueMode.uploadMethod
            )
            lastResponse = response
            recentForecasts.insert(response, at: 0)
            recentForecasts = Array(recentForecasts.prefix(5))
        } catch {
            banner = ActionBanner(title: "Forecast failed", detail: error.localizedDescription)
        }
    }

    private func optionalOverride(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func optionalFrequencyOverride(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.lowercased() == "auto" {
            return nil
        }
        return trimmed
    }
}
