import SwiftUI

struct ModelsTab: View {
    @EnvironmentObject private var workspace: ForecastWorkspace
    let client: TollamaHTTPClient

    @State private var installingModel: String?
    @State private var deletingModel: String?
    @State private var pullProgress: [String: String] = [:]
    @State private var licensePrompt: RegistryModel?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Models")
                    .font(.largeTitle.bold())
                Spacer()
                Button("Refresh") {
                    Task { await workspace.refreshModels(client: client) }
                }
                .disabled(workspace.isLoadingModels)
            }

            if workspace.isLoadingModels && workspace.models.isEmpty {
                ProgressView("Loading model registry")
            } else {
                List {
                    ForEach(groupedFamilies, id: \.self) { family in
                        Section(family) {
                            ForEach(models(for: family)) { model in
                                modelRow(model)
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .padding(24)
        .task {
            if workspace.models.isEmpty {
                await workspace.refreshModels(client: client)
            }
        }
        .alert(item: $licensePrompt) { model in
            Alert(
                title: Text("Accept model license?"),
                message: Text(model.license.notice ?? "This model requires license acceptance before download."),
                primaryButton: .default(Text("Accept and Install")) {
                    Task { await install(model, acceptLicense: true) }
                },
                secondaryButton: .cancel()
            )
        }
    }

    private var groupedFamilies: [String] {
        Array(Set(workspace.models.map(\.family))).sorted()
    }

    private func models(for family: String) -> [RegistryModel] {
        workspace.models
            .filter { $0.family == family }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    @ViewBuilder
    private func modelRow(_ model: RegistryModel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(model.name)
                            .font(.headline)
                        if model.isDemoModel {
                            Text("Demo/local")
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.16))
                                .clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        if model.installed {
                            Text("Installed")
                                .font(.caption)
                                .foregroundStyle(.green)
                        }
                    }
                    Text(model.source?.repoID ?? "local source")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text(modelDetailLine(model))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                actionButton(for: model)
            }

            if let progress = pullProgress[model.name], installingModel == model.name {
                ProgressView(progress)
                    .font(.caption)
            }

            if !model.installed && model.family != "mock" {
                Text("Runtime dependencies install lazily on first forecast when auto-bootstrap is enabled.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func actionButton(for model: RegistryModel) -> some View {
        if model.installed {
            Button(deletingModel == model.name ? "Removing..." : "Remove") {
                Task { await delete(model) }
            }
            .disabled(deletingModel != nil || installingModel != nil)
        } else {
            Button(installingModel == model.name ? "Installing..." : "Install") {
                if model.license.needsAcceptance && model.license.accepted != true {
                    licensePrompt = model
                } else {
                    Task { await install(model, acceptLicense: false) }
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(deletingModel != nil || installingModel != nil)
        }
    }

    private func modelDetailLine(_ model: RegistryModel) -> String {
        var parts = ["family \(model.family)"]
        if let maxHorizon = model.maxHorizon {
            parts.append("max horizon \(maxHorizon)")
        }
        if let capabilities = model.capabilities {
            parts.append(capabilities.summary)
        }
        if let implementation = model.implementation {
            parts.append(implementation)
        }
        return parts.joined(separator: " · ")
    }

    private func install(_ model: RegistryModel, acceptLicense: Bool) async {
        installingModel = model.name
        pullProgress[model.name] = "Starting pull..."
        defer {
            installingModel = nil
        }
        do {
            try await client.pullModel(name: model.name, acceptLicense: acceptLicense) { line in
                await MainActor.run {
                    pullProgress[model.name] = line
                }
            }
            pullProgress[model.name] = "Installed."
            await workspace.refreshModels(client: client)
            if workspace.selectedModelName == nil {
                workspace.selectedModelName = model.name
            }
        } catch {
            workspace.banner = ActionBanner(title: "Install failed", detail: error.localizedDescription)
        }
    }

    private func delete(_ model: RegistryModel) async {
        deletingModel = model.name
        defer {
            deletingModel = nil
        }
        do {
            try await client.deleteModel(name: model.name)
            pullProgress[model.name] = nil
            await workspace.refreshModels(client: client)
        } catch {
            workspace.banner = ActionBanner(title: "Remove failed", detail: error.localizedDescription)
        }
    }
}
