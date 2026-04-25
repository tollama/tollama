import AppKit
import Foundation

struct ShellResult {
    let status: Int32
    let stdout: String
    let stderr: String
}

struct RuntimeMarker: Codable {
    let installSpec: String?
    let preparedAt: String
    let tollamaVersion: String
}

struct ActionBanner: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
}

enum AppRuntimeError: LocalizedError {
    case message(String)
    case commandFailure(String)

    var errorDescription: String? {
        switch self {
        case .message(let message):
            return message
        case .commandFailure(let message):
            return message
        }
    }
}

enum CommandRunner {
    static func run(
        executable: String,
        arguments: [String],
        environment: [String: String]? = nil,
        currentDirectory: URL? = nil
    ) throws -> ShellResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        if let environment {
            process.environment = environment
        }

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw AppRuntimeError.commandFailure("Failed to launch \(executable): \(error.localizedDescription)")
        }

        process.waitUntilExit()

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        return ShellResult(
            status: process.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self)
        )
    }

    @discardableResult
    static func runRequired(
        executable: String,
        arguments: [String],
        environment: [String: String]? = nil,
        currentDirectory: URL? = nil
    ) throws -> ShellResult {
        let result = try run(
            executable: executable,
            arguments: arguments,
            environment: environment,
            currentDirectory: currentDirectory
        )
        guard result.status == 0 else {
            let output = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            let fallback = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            let detail = output.isEmpty ? fallback : output
            let suffix = detail.isEmpty ? "" : ": \(detail)"
            throw AppRuntimeError.commandFailure(
                "\(executable) failed with status \(result.status)\(suffix)"
            )
        }
        return result
    }
}

extension FileManager {
    func createDirectoryIfNeeded(at url: URL) throws {
        try createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
    }

    func removeItemIfExists(at url: URL) throws {
        if fileExists(atPath: url.path) {
            try removeItem(at: url)
        }
    }
}

enum LogTailReader {
    static func read(url: URL, maxBytes: Int = 4096) -> String {
        guard let data = try? Data(contentsOf: url) else {
            return "No daemon logs yet."
        }
        let tail = data.suffix(maxBytes)
        let text = trimToCurrentDaemonRun(
            String(decoding: tail, as: UTF8.self)
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            return "No daemon logs yet."
        }
        return text
    }

    private static func trimToCurrentDaemonRun(_ text: String) -> String {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let currentRunIndex = lines.lastIndex { line in
            line.hasPrefix("Starting Tollama daemon at ")
                || line.hasPrefix("INFO:     Started server process ")
        }
        guard let currentRunIndex else {
            return text
        }
        return lines[currentRunIndex...].joined(separator: "\n")
    }
}
