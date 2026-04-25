import Foundation

struct TollamaHTTPClientError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

actor TollamaHTTPClient {
    private let baseURL: URL
    private let jsonDecoder = JSONDecoder()

    init(baseURL: URL) {
        self.baseURL = baseURL
    }

    func health() async throws {
        _ = try await requestData(path: "/v1/health")
    }

    func listModels() async throws -> V1ModelsResponse {
        let data = try await requestData(path: "/v1/models")
        return try jsonDecoder.decode(V1ModelsResponse.self, from: data)
    }

    func showModel(name: String) async throws -> [String: JSONValue] {
        let body = try JSONSerialization.data(withJSONObject: ["model": name])
        let data = try await requestData(
            path: "/api/show",
            method: "POST",
            body: body,
            contentType: "application/json"
        )
        return try jsonDecoder.decode([String: JSONValue].self, from: data)
    }

    func deleteModel(name: String) async throws {
        let root = baseURL.appendingPathComponent("v1").appendingPathComponent("models")
        let url = root.appendingPathComponent(name)
        _ = try await requestData(url: url, method: "DELETE")
    }

    func pullModel(
        name: String,
        acceptLicense: Bool,
        onProgress: @escaping @Sendable (String) async -> Void
    ) async throws {
        let payload: [String: Any] = [
            "model": name,
            "stream": true,
            "accept_license": acceptLicense,
        ]
        let body = try JSONSerialization.data(withJSONObject: payload)
        var request = URLRequest(url: url(path: "/api/pull"))
        request.httpMethod = "POST"
        request.timeoutInterval = 60 * 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/x-ndjson", forHTTPHeaderField: "Accept")
        request.httpBody = body

        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw TollamaHTTPClientError(message: "Received a non-HTTP response from Tollama.")
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                let data = try await collect(bytes: bytes)
                throw httpError(from: data, statusCode: httpResponse.statusCode)
            }

            var didEmit = false
            for try await line in bytes.lines {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else {
                    continue
                }
                if let streamError = pullStreamErrorMessage(from: trimmed) {
                    throw TollamaHTTPClientError(message: streamError)
                }
                didEmit = true
                await onProgress(trimmed)
            }
            if !didEmit {
                await onProgress("Pull complete.")
            }
        } catch let error as TollamaHTTPClientError {
            throw error
        } catch let error as URLError {
            throw TollamaHTTPClientError(message: daemonReachabilityMessage(error))
        } catch {
            throw TollamaHTTPClientError(message: error.localizedDescription)
        }
    }

    private func pullStreamErrorMessage(from line: String) -> String? {
        guard let data = line.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = payload["error"] else {
            return nil
        }

        if let message = error as? String, !message.isEmpty {
            return message
        }
        if let errorObject = error as? [String: Any],
           let message = errorObject["message"] as? String,
           !message.isEmpty {
            return message
        }
        return "Model pull failed."
    }

    func uploadAndForecast(
        fileURL: URL,
        model: String,
        horizon: Int,
        quantiles: [Double],
        timestampColumn: String?,
        seriesIDColumn: String?,
        targetColumn: String?,
        freqColumn: String?
    ) async throws -> ForecastResponseDTO {
        let requestPayload: [String: Any] = [
            "model": model,
            "horizon": horizon,
            "quantiles": quantiles,
            "options": [:],
        ]
        let requestJSON = try JSONSerialization.data(withJSONObject: requestPayload)
        guard let requestJSONString = String(data: requestJSON, encoding: .utf8) else {
            throw TollamaHTTPClientError(message: "Unable to encode forecast request.")
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()
        appendFormField(name: "payload", value: requestJSONString, boundary: boundary, to: &body)
        appendFormField(name: "format_hint", value: "csv", boundary: boundary, to: &body)
        appendOptionalFormField(name: "timestamp_column", value: timestampColumn, boundary: boundary, to: &body)
        appendOptionalFormField(name: "series_id_column", value: seriesIDColumn, boundary: boundary, to: &body)
        appendOptionalFormField(name: "target_column", value: targetColumn, boundary: boundary, to: &body)
        appendOptionalFormField(name: "freq_column", value: freqColumn, boundary: boundary, to: &body)
        try appendFileField(name: "file", fileURL: fileURL, boundary: boundary, to: &body)
        body.appendUTF8("--\(boundary)--\r\n")

        let data = try await requestData(
            path: "/api/forecast/upload",
            method: "POST",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)",
            timeout: 300
        )
        return try jsonDecoder.decode(ForecastResponseDTO.self, from: data)
    }

    private func requestData(
        path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 120
    ) async throws -> Data {
        try await requestData(
            url: url(path: path),
            method: method,
            body: body,
            contentType: contentType,
            timeout: timeout
        )
    }

    private func requestData(
        url: URL,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = nil,
        timeout: TimeInterval = 120
    ) async throws -> Data {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let contentType {
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        request.httpBody = body

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw TollamaHTTPClientError(message: "Received a non-HTTP response from Tollama.")
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                throw httpError(from: data, statusCode: httpResponse.statusCode)
            }
            return data
        } catch let error as TollamaHTTPClientError {
            throw error
        } catch let error as URLError {
            throw TollamaHTTPClientError(message: daemonReachabilityMessage(error))
        } catch {
            throw TollamaHTTPClientError(message: error.localizedDescription)
        }
    }

    private func url(path: String) -> URL {
        var result = baseURL
        for component in path.split(separator: "/") {
            result.appendPathComponent(String(component))
        }
        return result
    }
}

private struct HTTPErrorEnvelope: Decodable {
    let detail: JSONValue?
    let hint: String?
}

private func collect(bytes: URLSession.AsyncBytes) async throws -> Data {
    var data = Data()
    for try await byte in bytes {
        data.append(byte)
    }
    return data
}

private func httpError(from data: Data, statusCode: Int) -> TollamaHTTPClientError {
    let decoder = JSONDecoder()
    if
        let envelope = try? decoder.decode(HTTPErrorEnvelope.self, from: data),
        envelope.detail != nil || envelope.hint != nil
    {
        let detail = envelope.detail.map(detailText) ?? "HTTP \(statusCode)"
        let hint = envelope.hint.map { " \($0)" } ?? ""
        return TollamaHTTPClientError(message: "\(friendlyPrefix(statusCode: statusCode, detail: envelope.detail)): \(detail)\(hint)")
    }

    let raw = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    return TollamaHTTPClientError(message: raw.isEmpty ? "HTTP \(statusCode)" : raw)
}

private func detailText(_ value: JSONValue) -> String {
    switch value {
    case .string(let text):
        return text
    case .object(let object):
        if let message = object["message"]?.stringValue {
            return message
        }
        if let code = object["code"]?.stringValue {
            return code
        }
        return object.map { "\($0.key): \($0.value)" }.sorted().joined(separator: ", ")
    default:
        return value.description
    }
}

private func friendlyPrefix(statusCode: Int, detail: JSONValue?) -> String {
    let text = detail.map(detailText)?.lowercased() ?? ""
    if statusCode == 404 && text.contains("model") {
        return "Model missing"
    }
    if statusCode == 409 && text.contains("license") {
        return "License required"
    }
    if statusCode == 400 {
        return "Invalid request"
    }
    if statusCode == 503 {
        return "Runtime unavailable"
    }
    if statusCode == 502 {
        return "Runner failed"
    }
    return "Tollama error"
}

private func daemonReachabilityMessage(_ error: URLError) -> String {
    switch error.code {
    case .cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet, .timedOut:
        return "Daemon unreachable. Check that Tollama is still running."
    default:
        return error.localizedDescription
    }
}

private func appendFormField(name: String, value: String, boundary: String, to data: inout Data) {
    data.appendUTF8("--\(boundary)\r\n")
    data.appendUTF8("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
    data.appendUTF8("\(value)\r\n")
}

private func appendOptionalFormField(
    name: String,
    value: String?,
    boundary: String,
    to data: inout Data
) {
    let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !trimmed.isEmpty else {
        return
    }
    appendFormField(name: name, value: trimmed, boundary: boundary, to: &data)
}

private func appendFileField(name: String, fileURL: URL, boundary: String, to data: inout Data) throws {
    data.appendUTF8("--\(boundary)\r\n")
    data.appendUTF8(
        "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(fileURL.lastPathComponent)\"\r\n"
    )
    data.appendUTF8("Content-Type: text/csv\r\n\r\n")
    data.append(try Data(contentsOf: fileURL))
    data.appendUTF8("\r\n")
}

private extension Data {
    mutating func appendUTF8(_ string: String) {
        append(Data(string.utf8))
    }
}
