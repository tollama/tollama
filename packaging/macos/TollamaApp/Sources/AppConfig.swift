import Foundation

struct BundledRuntimeManifest: Decodable {
    let pythonArchive: String
    let starterModel: String
    let tollamaVersion: String
    let wheelhouseDir: String
}

enum AppConfig {
    static let daemonHost = "127.0.0.1"
    static let daemonPort = 11435
    static let demoModel = "mock"
    static let defaultStarterModel = "sundial-base-128m"

    private static let fileManager = FileManager.default

    static var bundleVersion: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.0.0"
    }

    static var baseURL: URL {
        URL(string: "http://\(daemonHost):\(daemonPort)")!
    }

    static var dashboardURL: URL {
        baseURL.appendingPathComponent("dashboard")
    }

    static var healthURL: URL {
        baseURL.appendingPathComponent("v1/health")
    }

    static var runtimeManifest: BundledRuntimeManifest? {
        guard
            let manifestURL = Bundle.main.resourceURL?
                .appendingPathComponent("RuntimeAssets", isDirectory: true)
                .appendingPathComponent("runtime-manifest.json")
        else {
            return nil
        }
        guard let data = try? Data(contentsOf: manifestURL) else {
            return nil
        }
        return try? JSONDecoder().decode(BundledRuntimeManifest.self, from: data)
    }

    static var starterModel: String {
        runtimeManifest?.starterModel ?? defaultStarterModel
    }

    static var runtimeAssetsRoot: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("RuntimeAssets", isDirectory: true)
    }

    static var pythonArchiveURL: URL? {
        guard let runtimeAssetsRoot else {
            return nil
        }
        let archiveName = runtimeManifest?.pythonArchive ?? "python-runtime.tar.gz"
        return runtimeAssetsRoot.appendingPathComponent(archiveName)
    }

    static var wheelhouseURL: URL? {
        guard let runtimeAssetsRoot else {
            return nil
        }
        let wheelhouseDir = runtimeManifest?.wheelhouseDir ?? "wheelhouse"
        return runtimeAssetsRoot.appendingPathComponent(wheelhouseDir, isDirectory: true)
    }

    static var appSupportRoot: URL {
        fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Tollama", isDirectory: true)
    }

    static var runtimeRoot: URL {
        appSupportRoot.appendingPathComponent("runtime", isDirectory: true)
    }

    static var stateRoot: URL {
        appSupportRoot.appendingPathComponent("state", isDirectory: true)
    }

    static var logsRoot: URL {
        fileManager.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Logs", isDirectory: true)
            .appendingPathComponent("Tollama", isDirectory: true)
    }

    static var daemonLogURL: URL {
        logsRoot.appendingPathComponent("daemon.log")
    }

    static var runtimePythonRoot: URL {
        runtimeRoot.appendingPathComponent("python", isDirectory: true)
    }

    static var runtimeVenvRoot: URL {
        runtimeRoot.appendingPathComponent("venv", isDirectory: true)
    }

    static var runtimeMarkerURL: URL {
        runtimeRoot.appendingPathComponent("installed.json")
    }

    static var runtimePythonExecutable: URL {
        pythonExecutable(in: runtimePythonRoot)
    }

    static var venvPythonExecutable: URL {
        pythonExecutable(in: runtimeVenvRoot)
    }

    static func pythonExecutable(in root: URL) -> URL {
        let binDir = root.appendingPathComponent("bin", isDirectory: true)
        let candidates = [
            binDir.appendingPathComponent("python3"),
            binDir.appendingPathComponent("python3.11"),
            binDir.appendingPathComponent("python"),
        ]
        if let existing = candidates.first(where: { fileManager.isExecutableFile(atPath: $0.path) }) {
            return existing
        }
        return candidates[0]
    }
}
