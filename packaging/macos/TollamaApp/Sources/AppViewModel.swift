import AppKit
import Foundation

private struct ListeningProcess {
    let command: String
    let pid: Int32
}

private struct RuntimePreparationStatus {
    let installSpecMatches: Bool
    let packageImportReady: Bool
    let runtimeReady: Bool
    let versionMatches: Bool

    var requiresRepair: Bool {
        !(versionMatches && installSpecMatches && runtimeReady && packageImportReady)
    }
}

@MainActor
final class AppViewModel: ObservableObject {
    @Published var banner: ActionBanner?
    @Published private(set) var busyLabel: String?
    @Published private(set) var dashboardReady = false
    @Published private(set) var logTail = "No daemon logs yet."
    @Published private(set) var modeLabel = "Preparing runtime"
    @Published private(set) var statusDetail = "Starting Tollama..."
    @Published private(set) var statusTitle = "Launching Tollama"
    @Published private(set) var webReloadID = UUID()

    let dashboardURL = AppConfig.dashboardURL
    let httpClient = TollamaHTTPClient(baseURL: AppConfig.baseURL)

    private var daemonProcess: Process?
    private var didBootstrap = false
    private var logFileHandle: FileHandle?
    private var logLoopTask: Task<Void, Never>?
    private var shutdownRequested = false

    var usesExternalDaemon: Bool {
        dashboardReady && daemonProcess == nil
    }

    var maintenanceActionsDisabled: Bool {
        busyLabel != nil || usesExternalDaemon
    }

    func bootstrap() async {
        guard !didBootstrap else {
            return
        }
        didBootstrap = true
        do {
            try await launchOrAttach()
        } catch {
            banner = ActionBanner(title: "Launch failed", detail: error.localizedDescription)
            busyLabel = nil
        }
    }

    func tryDemoForecast() async {
        await runAction(named: "Running demo forecast") {
            try await self.ensureDaemonReady()
            _ = try await self.pullModel(named: AppConfig.demoModel)

            let payload: [String: Any] = [
                "model": AppConfig.demoModel,
                "horizon": 3,
                "series": [[
                    "id": "demo",
                    "freq": "D",
                    "timestamps": [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                        "2026-01-05",
                    ],
                    "target": [10, 11, 12, 13, 14],
                ]],
            ]

            let data = try await self.postJSON(path: "/v1/forecast", payload: payload)
            let summary = self.demoSummary(from: data)
            self.banner = ActionBanner(
                title: "Demo forecast complete",
                detail: summary
            )
        }
    }

    func installStarterModel() async {
        await runAction(named: "Installing starter model") {
            try await self.ensureDaemonReady()
            let data = try await self.pullModel(named: AppConfig.starterModel)
            let message = self.successSummary(from: data, fallback: "Installed \(AppConfig.starterModel).")
            self.banner = ActionBanner(title: "Starter model installed", detail: message)
        }
    }

    func openLogs() {
        try? FileManager.default.createDirectoryIfNeeded(at: AppConfig.logsRoot)
        NSWorkspace.shared.activateFileViewerSelecting([AppConfig.daemonLogURL])
    }

    func repairRuntime() async {
        guard !usesExternalDaemon else {
            banner = ActionBanner(
                title: "Repair unavailable",
                detail: "Tollama is attached to an existing daemon on 127.0.0.1:11435. Stop that daemon first, then relaunch the app to repair the bundled runtime."
            )
            return
        }

        await runAction(named: "Repairing bundled runtime") {
            self.stopManagedDaemon()
            try FileManager.default.removeItemIfExists(at: AppConfig.runtimeRoot)
            try await self.launchOrAttach(forceRuntimeRepair: true)
            self.banner = ActionBanner(
                title: "Runtime repaired",
                detail: "The bundled Python runtime and Tollama core were reinstalled."
            )
        }
    }

    func resetLocalData() async {
        guard !usesExternalDaemon else {
            banner = ActionBanner(
                title: "Reset unavailable",
                detail: "Tollama is attached to an existing daemon on 127.0.0.1:11435. Stop that daemon first, then relaunch the app to reset app-local data."
            )
            return
        }

        await runAction(named: "Resetting local data") {
            self.stopManagedDaemon()
            try FileManager.default.removeItemIfExists(at: AppConfig.runtimeRoot)
            try FileManager.default.removeItemIfExists(at: AppConfig.stateRoot)
            try FileManager.default.removeItemIfExists(at: AppConfig.logsRoot)
            try await self.launchOrAttach(forceRuntimeRepair: true)
            self.banner = ActionBanner(
                title: "Local data reset",
                detail: "App-local state, runtime, and logs were cleared and rebuilt."
            )
        }
    }

    func handleAppTermination() {
        shutdownRequested = true
        logLoopTask?.cancel()
        stopManagedDaemon()
    }

    private func launchOrAttach(forceRuntimeRepair: Bool = false) async throws {
        busyLabel = "Starting Tollama"
        logTail = LogTailReader.read(url: AppConfig.daemonLogURL)

        let runtimeStatus = currentRuntimePreparationStatus()
        let daemonHealthyAtLaunch = await healthCheck()

        do {
            if
                let listener = try listeningProcess(),
                isManagedAppDaemon(listener),
                (forceRuntimeRepair || runtimeStatus.requiresRepair || !daemonHealthyAtLaunch)
            {
                statusTitle = "Refreshing runtime"
                statusDetail = daemonHealthyAtLaunch
                    ? "Stopping an older bundled daemon before refreshing Tollama."
                    : "Stopping an unresponsive bundled daemon before restarting Tollama."
                try await terminateManagedDaemon(listener)
            }
        } catch {
            statusDetail = "Unable to inspect port ownership: \(error.localizedDescription)"
        }

        if await healthCheck() {
            dashboardReady = true
            modeLabel = "Connected to existing daemon"
            statusTitle = "Tollama is ready"
            statusDetail = "Using the daemon already listening on 127.0.0.1:11435."
            busyLabel = nil
            webReloadID = UUID()
            return
        }

        do {
            if let conflict = try conflictingProcessDescription() {
                dashboardReady = false
                modeLabel = "Port conflict"
                statusTitle = "Port 11435 is already in use"
                statusDetail = conflict
                busyLabel = nil
                return
            }
        } catch {
            statusDetail = "Unable to inspect port ownership: \(error.localizedDescription)"
        }

        try await prepareRuntimeIfNeeded(forceRepair: forceRuntimeRepair || runtimeStatus.requiresRepair)
        try startManagedDaemon()
        try await waitForHealthyDaemon(timeoutSeconds: 30)

        dashboardReady = true
        modeLabel = "Managed daemon"
        statusTitle = "Tollama is ready"
        statusDetail = "The bundled dashboard is running inside the app."
        busyLabel = nil
        webReloadID = UUID()
        startLogLoop()
    }

    private func ensureDaemonReady() async throws {
        if await healthCheck() {
            return
        }
        try await launchOrAttach()
    }

    private func runAction(named label: String, action: @escaping @MainActor () async throws -> Void) async {
        let previousBusy = busyLabel
        busyLabel = label
        do {
            try await action()
        } catch {
            banner = ActionBanner(
                title: label,
                detail: error.localizedDescription
            )
        }
        logTail = LogTailReader.read(url: AppConfig.daemonLogURL)
        busyLabel = previousBusy
    }

    private func prepareRuntimeIfNeeded(forceRepair: Bool) async throws {
        statusTitle = "Preparing runtime"
        statusDetail = "Bootstrapping a private Python environment in Application Support."

        let fileManager = FileManager.default
        let runtimeStatus = currentRuntimePreparationStatus()

        if !forceRepair && !runtimeStatus.requiresRepair {
            return
        }

        guard
            let pythonArchiveURL = AppConfig.pythonArchiveURL,
            let wheelhouseURL = AppConfig.wheelhouseURL
        else {
            throw AppRuntimeError.message("Bundled runtime assets are missing from Tollama.app.")
        }

        try fileManager.createDirectoryIfNeeded(at: AppConfig.appSupportRoot)
        try fileManager.createDirectoryIfNeeded(at: AppConfig.logsRoot)
        try fileManager.removeItemIfExists(at: AppConfig.runtimeRoot)
        try fileManager.createDirectoryIfNeeded(at: AppConfig.runtimeRoot)
        try fileManager.createDirectoryIfNeeded(at: AppConfig.runtimePythonRoot)

        try await runDetached {
            try CommandRunner.runRequired(
                executable: "/usr/bin/tar",
                arguments: [
                    "-xzf",
                    pythonArchiveURL.path,
                    "-C",
                    AppConfig.runtimePythonRoot.path,
                ]
            )

            try CommandRunner.runRequired(
                executable: AppConfig.runtimePythonExecutable.path,
                arguments: [
                    "-m",
                    "venv",
                    AppConfig.runtimeVenvRoot.path,
                ]
            )

            try CommandRunner.runRequired(
                executable: AppConfig.venvPythonExecutable.path,
                arguments: [
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    wheelhouseURL.path,
                    AppConfig.bundledInstallSpec,
                ]
            )
        }

        try writeRuntimeMarker(RuntimeMarker(
            installSpec: AppConfig.bundledInstallSpec,
            preparedAt: ISO8601DateFormatter().string(from: Date()),
            tollamaVersion: AppConfig.bundleVersion
        ))
    }

    private func startManagedDaemon() throws {
        try FileManager.default.createDirectoryIfNeeded(at: AppConfig.stateRoot)
        try FileManager.default.createDirectoryIfNeeded(at: AppConfig.logsRoot)

        let logURL = AppConfig.daemonLogURL
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }

        let fileHandle = try FileHandle(forWritingTo: logURL)
        try fileHandle.truncate(atOffset: 0)
        if let header = "Starting Tollama daemon at \(ISO8601DateFormatter().string(from: Date()))\n"
            .data(using: .utf8)
        {
            fileHandle.write(header)
        }
        try fileHandle.seekToEnd()

        let process = Process()
        process.executableURL = AppConfig.venvPythonExecutable
        process.arguments = ["-m", "tollama.daemon.main"]

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment["TOLLAMA_HOME"] = AppConfig.stateRoot.path
        environment["TOLLAMA_HOST"] = "\(AppConfig.daemonHost):\(AppConfig.daemonPort)"
        environment["TOLLAMA_LOG_LEVEL"] = "info"
        if let wheelhouseURL = AppConfig.wheelhouseURL {
            environment["TOLLAMA_RUNTIME_WHEELHOUSE"] = wheelhouseURL.path
        }
        process.environment = environment
        process.standardOutput = fileHandle
        process.standardError = fileHandle
        process.currentDirectoryURL = AppConfig.appSupportRoot
        process.terminationHandler = { [weak self] terminatedProcess in
            Task { @MainActor in
                self?.handleDaemonTermination(process: terminatedProcess)
            }
        }

        shutdownRequested = false
        try process.run()
        daemonProcess = process
        logFileHandle = fileHandle
    }

    private func stopManagedDaemon() {
        logLoopTask?.cancel()
        logLoopTask = nil

        if let process = daemonProcess {
            process.terminate()
            process.waitUntilExit()
        }

        daemonProcess = nil
        try? logFileHandle?.close()
        logFileHandle = nil
    }

    private func handleDaemonTermination(process: Process) {
        daemonProcess = nil
        try? logFileHandle?.close()
        logFileHandle = nil
        logTail = LogTailReader.read(url: AppConfig.daemonLogURL)
        if shutdownRequested {
            return
        }
        dashboardReady = false
        modeLabel = "Daemon stopped"
        statusTitle = "Tollama stopped unexpectedly"
        statusDetail = "Check the log tail, then use Repair Runtime or relaunch the app."
    }

    private func waitForHealthyDaemon(timeoutSeconds: Int) async throws {
        for _ in 0..<(timeoutSeconds * 2) {
            if await healthCheck() {
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw AppRuntimeError.message("Timed out while waiting for the local Tollama daemon to become ready.")
    }

    private func healthCheck() async -> Bool {
        var request = URLRequest(url: AppConfig.healthURL)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            return httpResponse.statusCode == 200
        } catch {
            return false
        }
    }

    private func postJSON(path: String, payload: [String: Any]) async throws -> Data {
        let url = AppConfig.baseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AppRuntimeError.message("Received a non-HTTP response from Tollama.")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw AppRuntimeError.message(serverErrorMessage(from: data))
        }
        return data
    }

    private func pullModel(named model: String) async throws -> Data {
        try await postJSON(path: "/api/pull", payload: [
            "model": model,
            "stream": false,
            "accept_license": false,
        ])
    }

    private func serverErrorMessage(from data: Data) -> String {
        guard
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return String(decoding: data, as: UTF8.self)
        }

        if let detail = payload["detail"] as? String, !detail.isEmpty {
            return detail
        }
        if let detail = payload["detail"] {
            return String(describing: detail)
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func demoSummary(from data: Data) -> String {
        guard
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let series = payload["series"] as? [[String: Any]],
            let firstSeries = series.first,
            let mean = firstSeries["mean"] as? [Double]
        else {
            return "The mock forecast completed successfully."
        }

        let joined = mean.map { String(format: "%.2f", $0) }.joined(separator: ", ")
        return "Mock forecast values: \(joined)"
    }

    private func successSummary(from data: Data, fallback: String) -> String {
        guard
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = payload["model"] as? String
        else {
            return fallback
        }
        return "Installed \(model). You can now run it from the embedded dashboard."
    }

    private func currentRuntimePreparationStatus() -> RuntimePreparationStatus {
        let currentMarker = try? readRuntimeMarker()
        let runtimeReady = FileManager.default.isExecutableFile(atPath: AppConfig.venvPythonExecutable.path)
        return RuntimePreparationStatus(
            installSpecMatches: currentMarker?.installSpec == AppConfig.bundledInstallSpec,
            packageImportReady: runtimeReady && tollamaPackageImportReady(),
            runtimeReady: runtimeReady,
            versionMatches: currentMarker?.tollamaVersion == AppConfig.bundleVersion
        )
    }

    private func tollamaPackageImportReady() -> Bool {
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")

        do {
            let result = try CommandRunner.run(
                executable: AppConfig.venvPythonExecutable.path,
                arguments: [
                    "-c",
                    "import tollama.daemon.main",
                ],
                environment: environment
            )
            return result.status == 0
        } catch {
            return false
        }
    }

    private func listeningProcess() throws -> ListeningProcess? {
        let pidResult = try CommandRunner.run(
            executable: "/usr/sbin/lsof",
            arguments: [
                "-nP",
                "-t",
                "-iTCP:\(AppConfig.daemonPort)",
                "-sTCP:LISTEN",
            ]
        )

        guard pidResult.status == 0 else {
            return nil
        }

        guard
            let pidLine = pidResult.stdout
                .split(separator: "\n")
                .map(String.init)
                .first(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
            let pid = Int32(pidLine.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            return nil
        }

        let commandResult = try CommandRunner.run(
            executable: "/bin/ps",
            arguments: ["-p", String(pid), "-o", "command="]
        )

        let command = commandResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return ListeningProcess(command: command, pid: pid)
    }

    private func isManagedAppDaemon(_ process: ListeningProcess) -> Bool {
        process.command.contains(AppConfig.runtimeRoot.path) || process.command.contains(AppConfig.appSupportRoot.path)
    }

    private func terminateManagedDaemon(_ process: ListeningProcess) async throws {
        _ = try CommandRunner.run(
            executable: "/bin/kill",
            arguments: ["-TERM", String(process.pid)]
        )

        for _ in 0..<20 {
            try await Task.sleep(nanoseconds: 250_000_000)
            if try listeningProcess()?.pid != process.pid {
                return
            }
        }

        _ = try CommandRunner.run(
            executable: "/bin/kill",
            arguments: ["-KILL", String(process.pid)]
        )

        for _ in 0..<20 {
            try await Task.sleep(nanoseconds: 250_000_000)
            if try listeningProcess()?.pid != process.pid {
                return
            }
        }

        throw AppRuntimeError.message("Timed out while stopping the older bundled daemon.")
    }

    private func conflictingProcessDescription() throws -> String? {
        let result = try CommandRunner.run(
            executable: "/usr/sbin/lsof",
            arguments: [
                "-nP",
                "-iTCP:\(AppConfig.daemonPort)",
                "-sTCP:LISTEN",
            ]
        )

        guard result.status == 0 else {
            return nil
        }

        let lines = result.stdout.split(separator: "\n").map(String.init)
        guard lines.count >= 2 else {
            return nil
        }

        let processLine = lines[1]
        return "Another process is already listening on 127.0.0.1:\(AppConfig.daemonPort):\n\(processLine)\n\nStop that process and relaunch Tollama."
    }

    private func startLogLoop() {
        logLoopTask?.cancel()
        logLoopTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                self.logTail = LogTailReader.read(url: AppConfig.daemonLogURL)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func readRuntimeMarker() throws -> RuntimeMarker {
        let data = try Data(contentsOf: AppConfig.runtimeMarkerURL)
        return try JSONDecoder().decode(RuntimeMarker.self, from: data)
    }

    private func writeRuntimeMarker(_ marker: RuntimeMarker) throws {
        let data = try JSONEncoder().encode(marker)
        try data.write(to: AppConfig.runtimeMarkerURL)
    }

    private func runDetached<T>(_ operation: @escaping () throws -> T) async throws -> T {
        try await Task.detached(priority: .userInitiated) {
            try operation()
        }.value
    }
}
