import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppViewModel
    @EnvironmentObject private var workspace: ForecastWorkspace

    var body: some View {
        NavigationSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    statusCard
                    actionButtons
                    logCard
                }
                .padding(20)
            }
            .navigationTitle("Tollama")
        } detail: {
            Group {
                if model.dashboardReady {
                    NativeWorkspaceTabs(model: model)
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
        .alert(item: activeBanner) { banner in
            Alert(
                title: Text(banner.title),
                message: Text(banner.detail),
                dismissButton: .default(Text("OK"))
            )
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

struct NativeWorkspaceTabs: View {
    @ObservedObject var model: AppViewModel
    @EnvironmentObject private var workspace: ForecastWorkspace

    var body: some View {
        TabView(selection: $workspace.selectedTab) {
            ModelsTab(client: model.httpClient)
                .tabItem { Text("Models") }
                .tag(WorkspaceTab.models)

            DataTab()
                .tabItem { Text("Data") }
                .tag(WorkspaceTab.data)

            ForecastTab(client: model.httpClient)
                .tabItem { Text("Forecast") }
                .tag(WorkspaceTab.forecast)

            LogsTab(model: model)
                .tabItem { Text("Logs") }
                .tag(WorkspaceTab.logs)
        }
    }
}
