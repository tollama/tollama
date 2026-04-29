import Foundation

struct ForecastResponseDTO: Decodable, Identifiable, Sendable {
    let id = UUID()
    let model: String
    let forecasts: [SeriesForecastDTO]
    let timing: [String: JSONValue]?
    let preprocessing: ForecastPreprocessingDTO?
    let warnings: [String]?

    enum CodingKeys: String, CodingKey {
        case model
        case forecasts
        case timing
        case preprocessing
        case warnings
    }
}

struct SeriesForecastDTO: Decodable, Identifiable, Sendable {
    let id: String
    let freq: String
    let startTimestamp: String
    let mean: [Double]
    let quantiles: [String: [Double]]?

    enum CodingKeys: String, CodingKey {
        case id
        case freq
        case startTimestamp = "start_timestamp"
        case mean
        case quantiles
    }
}

struct ForecastHistoryPoint: Identifiable, Equatable, Sendable {
    let id = UUID()
    let step: Int
    let timestamp: String?
    let value: Double
}

struct ForecastPreprocessingDTO: Decodable, Equatable, Sendable {
    let series: [SeriesPreprocessingDiagnosticsDTO]
}

struct SeriesPreprocessingDiagnosticsDTO: Decodable, Identifiable, Equatable, Sendable {
    let id: String
    let originalRowCount: Int
    let regularizedRowCount: Int
    let rawNullTargetCount: Int
    let missingTimestampCount: Int
    let imputedPointCount: Int
    let maxGap: Int
    let missingRatio: Double
    let requestedMethod: String
    let usedMethod: String
    let warnings: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case originalRowCount = "original_row_count"
        case regularizedRowCount = "regularized_row_count"
        case rawNullTargetCount = "raw_null_target_count"
        case missingTimestampCount = "missing_timestamp_count"
        case imputedPointCount = "imputed_point_count"
        case maxGap = "max_gap"
        case missingRatio = "missing_ratio"
        case requestedMethod = "requested_method"
        case usedMethod = "used_method"
        case warnings
    }
}

struct ForecastRow: Identifiable, Equatable, Sendable {
    let id = UUID()
    let step: Int
    let timestamp: String?
    let mean: Double
    let q10: Double?
    let q50: Double?
    let q90: Double?
}

extension SeriesForecastDTO {
    func rows() -> [ForecastRow] {
        mean.enumerated().map { index, value in
            ForecastRow(
                step: index + 1,
                timestamp: synthesizedTimestamp(step: index + 1),
                mean: value,
                q10: quantiles?["0.1"]?[safe: index],
                q50: quantiles?["0.5"]?[safe: index],
                q90: quantiles?["0.9"]?[safe: index]
            )
        }
    }

    private func synthesizedTimestamp(step: Int) -> String? {
        guard step > 0 else {
            return nil
        }

        let normalizedFreq = freq.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let calendar = Calendar(identifier: .gregorian)
        let dateOnlyFormatter = DateFormatter()
        dateOnlyFormatter.calendar = calendar
        dateOnlyFormatter.locale = Locale(identifier: "en_US_POSIX")
        dateOnlyFormatter.timeZone = TimeZone(secondsFromGMT: 0)
        dateOnlyFormatter.dateFormat = "yyyy-MM-dd"

        let isoFormatter = ISO8601DateFormatter()
        let startDate = isoFormatter.date(from: startTimestamp)
            ?? dateOnlyFormatter.date(from: startTimestamp)
        guard let startDate else {
            return step == 1 ? startTimestamp : nil
        }

        let component: Calendar.Component
        switch normalizedFreq {
        case "D":
            component = .day
        case "H":
            component = .hour
        case "W":
            component = .weekOfYear
        default:
            return step == 1 ? startTimestamp : nil
        }

        guard let nextDate = calendar.date(byAdding: component, value: step - 1, to: startDate) else {
            return step == 1 ? startTimestamp : nil
        }
        return dateOnlyFormatter.string(from: nextDate)
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
