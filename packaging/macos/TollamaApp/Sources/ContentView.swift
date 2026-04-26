import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppViewModel
    @EnvironmentObject private var workspace: ForecastWorkspace
    @State private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    statusCard
                    actionButtons
                    logCard
                }
                .padding(20)
            }
            .navigationTitle("Tollama")
            .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 360)
        } detail: {
            Group {
                if model.dashboardReady {
                    NativeWorkspaceTabs(model: model)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(alignment: .leading, spacing: 16) {
                        Text(model.statusTitle)
                            .font(.largeTitle.bold())
                        Text(model.statusDetail)
                            .foregroundStyle(.secondary)
                        if let busyLabel = model.busyLabel {
                            ProgressView(busyLabel)
                                .controlSize(.large)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                    .padding(32)
                }
            }
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(item: activeBanner) { banner in
            ActionBannerSheet(banner: banner) {
                model.banner = nil
                workspace.banner = nil
            }
        }
    }

    private var activeBanner: Binding<ActionBanner?> {
        Binding(
            get: { model.banner ?? workspace.banner },
            set: { _ in
                model.banner = nil
                workspace.banner = nil
            }
        )
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(model.statusTitle)
                .font(.title2.bold())
            Text(model.modeLabel)
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(model.statusDetail)
                .foregroundStyle(.secondary)
            if let busyLabel = model.busyLabel {
                ProgressView(busyLabel)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var actionButtons: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
            Button("Try Demo Forecast") {
                Task { await model.tryDemoForecast() }
            }
            .buttonStyle(.borderedProminent)

            Button("Install Starter Model") {
                Task { await model.installStarterModel() }
            }
            .buttonStyle(.bordered)

            Button("Open Logs") {
                model.openLogs()
            }
            .buttonStyle(.bordered)

            Button("Repair Runtime") {
                Task { await model.repairRuntime() }
            }
            .buttonStyle(.bordered)
            .disabled(model.maintenanceActionsDisabled)

            Button("Reset Local Data") {
                Task { await model.resetLocalData() }
            }
            .buttonStyle(.bordered)
            .disabled(model.maintenanceActionsDisabled)

            if model.usesExternalDaemon {
                Text("Repair and reset are disabled while the app is attached to an already-running daemon.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var logCard: some View {
        LogsTailView(logTail: model.logTail, minHeight: 220)
    }
}

struct ActionBannerSheet: View {
    let banner: ActionBanner
    let dismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(banner.title)
                .font(.title2.bold())

            ScrollView {
                Text(banner.detail)
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.trailing, 8)
            }
            .frame(minHeight: 120, maxHeight: 360)

            HStack {
                Spacer()
                Button("OK") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(minWidth: 420, idealWidth: 560, maxWidth: 720)
    }
}

struct NativeWorkspaceTabs: View {
    @ObservedObject var model: AppViewModel
    @EnvironmentObject private var workspace: ForecastWorkspace

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("Workspace", selection: $workspace.selectedTab) {
                    Text("Models").tag(WorkspaceTab.models)
                    Text("Data").tag(WorkspaceTab.data)
                    Text("Forecast").tag(WorkspaceTab.forecast)
                    Text("Logs").tag(WorkspaceTab.logs)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 520)

                Spacer()
            }
            .padding(.horizontal, 24)
            .padding(.top, 18)
            .padding(.bottom, 12)

            Divider()

            Group {
                switch workspace.selectedTab {
                case .models:
                    ModelsTab(client: model.httpClient)
                case .data:
                    DataTab()
                case .forecast:
                    ForecastTab(client: model.httpClient)
                case .logs:
                    LogsTab(model: model)
                }
            }
            .id(workspace.selectedTab)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
