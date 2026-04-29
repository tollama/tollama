import Foundation

struct CSVFileItem: Identifiable, Equatable, Sendable {
    let url: URL
    let name: String
    let sizeBytes: Int64
    let rowCount: Int?

    var id: String { url.path }

    var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

struct CSVPreview: Equatable, Sendable {
    let url: URL
    let columns: [String]
    let sampleRows: [[String]]
    let rowCount: Int?
    let timestampColumn: String?
    let targetColumn: String?
    let seriesIDColumn: String?
    let freqColumn: String?
    let inferredFrequency: String?
    let sampledNullTargetCount: Int
    let sampledCadenceGapCount: Int
    let history: [ForecastHistoryPoint]

    var hasMissingValueSignals: Bool {
        sampledNullTargetCount > 0 || sampledCadenceGapCount > 0
    }
}

enum CSVSniffer {
    static let rowCountByteLimit = 16 * 1024 * 1024
    static let timestampCandidates = [
        "timestamp",
        "timestamps",
        "ds",
        "time",
        "date",
        "datetime",
        "date_time",
        "observation_date",
        "utc_timestamp",
        "year",
        "fecha",
    ]
    static let seriesIDCandidates = ["id", "series", "series_id", "unique_id", "entity", "country"]
    static let targetCandidates = [
        "target",
        "value",
        "y",
        "ot",
        "demand",
        "users",
        "number of flights",
        "total electricity",
        "gdp",
        "close",
        "actual",
        "pm2.5",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
    ]
    static let targetSuffixCandidates = ["_load_actual_entsoe_transparency"]
    static let freqCandidates = ["freq", "frequency"]

    static func scanCSVFiles(in folder: URL, limit: Int = 500) throws -> [CSVFileItem] {
        let fileManager = FileManager.default
        let keys: [URLResourceKey] = [.isRegularFileKey, .isDirectoryKey, .isHiddenKey, .fileSizeKey]
        guard let enumerator = fileManager.enumerator(
            at: folder,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files: [CSVFileItem] = []
        for case let fileURL as URL in enumerator {
            if files.count >= limit {
                break
            }
            let values = try fileURL.resourceValues(forKeys: Set(keys))
            if values.isHidden == true {
                if values.isDirectory == true {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values.isRegularFile == true else {
                continue
            }
            guard fileURL.pathExtension.lowercased() == "csv" else {
                continue
            }
            files.append(
                CSVFileItem(
                    url: fileURL,
                    name: fileURL.lastPathComponent,
                    sizeBytes: Int64(values.fileSize ?? 0),
                    rowCount: nil
                )
            )
        }
        return files.sorted { left, right in
            left.name.localizedCaseInsensitiveCompare(right.name) == .orderedAscending
        }
    }

    static func preview(url: URL, maxRows: Int = 200) throws -> CSVPreview {
        let data = try readPrefix(url: url, maxBytes: 512 * 1024)
        let text = decodeCSVText(data)
        let parsedRows = parse(text, maxRows: maxRows + 1)
        let columns = parsedRows.first?.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) } ?? []
        let dataRows = Array(parsedRows.dropFirst())
        let timestampColumn = firstExistingColumn(columns: columns, candidates: timestampCandidates)
        let seriesIDColumn = firstExistingColumn(columns: columns, candidates: seriesIDCandidates)
        let freqColumn = firstExistingColumn(columns: columns, candidates: freqCandidates)
        let targetColumn = firstExistingColumn(columns: columns, candidates: targetCandidates)
            ?? firstColumnWithSuffix(columns: columns, suffixes: targetSuffixCandidates)
            ?? firstNumericColumn(columns: columns, rows: dataRows, excluding: [
                timestampColumn,
                seriesIDColumn,
                freqColumn,
            ])
        let inferredFrequency = inferFrequency(
            columns: columns,
            rows: dataRows,
            timestampColumn: timestampColumn,
            targetColumn: targetColumn
        )
        let sampledNullTargetCount = countNullTargets(
            columns: columns,
            rows: dataRows,
            targetColumn: targetColumn
        )
        let sampledCadenceGapCount = countCadenceGaps(
            columns: columns,
            rows: dataRows,
            timestampColumn: timestampColumn
        )

        let history = buildHistory(
            columns: columns,
            rows: dataRows,
            timestampColumn: timestampColumn,
            targetColumn: targetColumn
        )

        return CSVPreview(
            url: url,
            columns: columns,
            sampleRows: Array(dataRows.prefix(10)),
            rowCount: try? countDataRowsIfSmall(in: url),
            timestampColumn: timestampColumn,
            targetColumn: targetColumn,
            seriesIDColumn: seriesIDColumn,
            freqColumn: freqColumn,
            inferredFrequency: inferredFrequency,
            sampledNullTargetCount: sampledNullTargetCount,
            sampledCadenceGapCount: sampledCadenceGapCount,
            history: history
        )
    }

    static func parse(_ text: String, maxRows: Int) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        var index = text.startIndex

        func finishField() {
            row.append(field)
            field = ""
        }

        func finishRow() {
            finishField()
            rows.append(row)
            row = []
        }

        while index < text.endIndex && rows.count < maxRows {
            let character = text[index]
            if inQuotes {
                if character == "\"" {
                    let nextIndex = text.index(after: index)
                    if nextIndex < text.endIndex && text[nextIndex] == "\"" {
                        field.append("\"")
                        index = nextIndex
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(character)
                }
            } else {
                switch character {
                case "\"":
                    inQuotes = true
                case ",":
                    finishField()
                case "\n":
                    finishRow()
                case "\r":
                    finishRow()
                    let nextIndex = text.index(after: index)
                    if nextIndex < text.endIndex && text[nextIndex] == "\n" {
                        index = nextIndex
                    }
                default:
                    field.append(character)
                }
            }
            index = text.index(after: index)
        }

        if rows.count < maxRows && (!field.isEmpty || !row.isEmpty) {
            finishRow()
        }
        return rows.filter { !$0.allSatisfy { $0.isEmpty } }
    }

    private static func readPrefix(url: URL, maxBytes: Int) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        return handle.readData(ofLength: maxBytes)
    }

    private static func decodeCSVText(_ data: Data) -> String {
        if let text = String(data: data, encoding: .utf8) {
            return text
        }
        if let text = String(data: data, encoding: .windowsCP1252) {
            return text
        }
        if let text = String(data: data, encoding: .isoLatin1) {
            return text
        }
        return String(decoding: data, as: UTF8.self)
    }

    private static func countDataRowsIfSmall(in url: URL) throws -> Int? {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        guard (values.fileSize ?? 0) <= rowCountByteLimit else {
            return nil
        }
        return try countDataRows(in: url)
    }

    private static func countDataRows(in url: URL) throws -> Int {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }

        var newlineCount = 0
        while true {
            let data = handle.readData(ofLength: 64 * 1024)
            if data.isEmpty {
                break
            }
            newlineCount += data.reduce(0) { count, byte in
                count + (byte == 10 ? 1 : 0)
            }
        }
        return max(0, newlineCount - 1)
    }

    private static func firstExistingColumn(columns: [String], candidates: [String]) -> String? {
        var lookup: [String: String] = [:]
        for column in columns where lookup[normalizedColumnName(column)] == nil {
            lookup[normalizedColumnName(column)] = column
        }
        for candidate in candidates {
            if let match = lookup[normalizedColumnName(candidate)] {
                return match
            }
        }
        return nil
    }

    private static func normalizedColumnName(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\u{feff}"))
            .lowercased()
    }

    private static func firstColumnWithSuffix(columns: [String], suffixes: [String]) -> String? {
        for column in columns {
            let normalized = normalizedColumnName(column)
            if suffixes.contains(where: { normalized.hasSuffix(normalizedColumnName($0)) }) {
                return column
            }
        }
        return nil
    }

    private static func firstNumericColumn(
        columns: [String],
        rows: [[String]],
        excluding excluded: [String?]
    ) -> String? {
        let excludedNames = Set(excluded.compactMap { $0?.lowercased() })
        for (index, column) in columns.enumerated() where !excludedNames.contains(column.lowercased()) {
            let sample = rows.prefix(25).compactMap { $0[safe: index] }
            guard !sample.isEmpty else {
                continue
            }
            if sample.contains(where: { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) != nil }) {
                return column
            }
        }
        return nil
    }

    private static func buildHistory(
        columns: [String],
        rows: [[String]],
        timestampColumn: String?,
        targetColumn: String?
    ) -> [ForecastHistoryPoint] {
        guard
            let targetColumn,
            let targetIndex = columns.firstIndex(of: targetColumn)
        else {
            return []
        }
        let timestampIndex = timestampColumn.flatMap { columns.firstIndex(of: $0) }
        return rows.enumerated().compactMap { index, row in
            guard
                let rawValue = row[safe: targetIndex]?.trimmingCharacters(in: .whitespacesAndNewlines),
                let value = Double(rawValue)
            else {
                return nil
            }
            return ForecastHistoryPoint(
                step: index + 1,
                timestamp: timestampIndex.flatMap { row[safe: $0] },
                value: value
            )
        }
    }

    private static func inferFrequency(
        columns: [String],
        rows: [[String]],
        timestampColumn: String?,
        targetColumn: String?
    ) -> String? {
        guard
            let timestampColumn,
            let targetColumn,
            let timestampIndex = columns.firstIndex(of: timestampColumn),
            let targetIndex = columns.firstIndex(of: targetColumn)
        else {
            return nil
        }

        let dates = rows.compactMap { row -> Date? in
            guard
                let rawTarget = row[safe: targetIndex]?.trimmingCharacters(in: .whitespacesAndNewlines),
                Double(rawTarget) != nil,
                let rawTimestamp = row[safe: timestampIndex]
            else {
                return nil
            }
            return parseTimestamp(rawTimestamp)
        }.sorted()
        guard dates.count >= 3 else {
            return nil
        }

        let deltas = zip(dates.dropLast(), dates.dropFirst())
            .map { previous, current in current.timeIntervalSince(previous) }
            .filter { $0 > 0 }
        guard !deltas.isEmpty else {
            return nil
        }

        var counts: [Int: Int] = [:]
        for delta in deltas {
            let seconds = Int(delta.rounded())
            guard abs(delta - Double(seconds)) <= 0.000001 else {
                continue
            }
            counts[seconds, default: 0] += 1
        }
        guard
            let dominant = counts.max(by: { left, right in left.value < right.value }),
            Double(dominant.value) / Double(deltas.count) >= 0.8
        else {
            return nil
        }
        return frequencyAlias(seconds: dominant.key)
    }

    private static func countNullTargets(
        columns: [String],
        rows: [[String]],
        targetColumn: String?
    ) -> Int {
        guard
            let targetColumn,
            let targetIndex = columns.firstIndex(of: targetColumn)
        else {
            return 0
        }

        return rows.reduce(0) { count, row in
            let rawValue = row[safe: targetIndex]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return count + (isNullToken(rawValue) ? 1 : 0)
        }
    }

    private static func countCadenceGaps(
        columns: [String],
        rows: [[String]],
        timestampColumn: String?
    ) -> Int {
        guard
            let timestampColumn,
            let timestampIndex = columns.firstIndex(of: timestampColumn)
        else {
            return 0
        }

        let dates = rows.compactMap { row -> Date? in
            guard let rawTimestamp = row[safe: timestampIndex] else {
                return nil
            }
            return parseTimestamp(rawTimestamp)
        }.sorted()
        guard dates.count >= 3 else {
            return 0
        }

        let deltas = zip(dates.dropLast(), dates.dropFirst())
            .map { previous, current in current.timeIntervalSince(previous) }
            .filter { $0 > 0 }
        guard
            let dominant = dominantIntervalSeconds(deltas),
            dominant > 0
        else {
            return 0
        }
        return deltas.filter { $0 > Double(dominant) * 1.5 }.count
    }

    private static func isNullToken(_ value: String) -> Bool {
        if value.isEmpty {
            return true
        }
        switch value.lowercased() {
        case "null", "none", "nan", "na", "n/a":
            return true
        default:
            return false
        }
    }

    private static func dominantIntervalSeconds(_ deltas: [TimeInterval]) -> Int? {
        var counts: [Int: Int] = [:]
        for delta in deltas {
            let seconds = Int(delta.rounded())
            guard abs(delta - Double(seconds)) <= 0.000001 else {
                continue
            }
            counts[seconds, default: 0] += 1
        }
        guard
            let dominant = counts.max(by: { left, right in left.value < right.value }),
            Double(dominant.value) / Double(deltas.count) >= 0.8
        else {
            return nil
        }
        return dominant.key
    }

    private static func parseTimestamp(_ rawValue: String) -> Date? {
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            return nil
        }

        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = isoFormatter.date(from: value) {
            return date
        }
        isoFormatter.formatOptions = [.withInternetDateTime]
        if let date = isoFormatter.date(from: value) {
            return date
        }

        for format in [
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd HH:mm",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm",
            "yyyy-MM-dd",
        ] {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = format
            if let date = formatter.date(from: value) {
                return date
            }
        }
        return nil
    }

    private static func frequencyAlias(seconds: Int) -> String? {
        guard seconds > 0 else {
            return nil
        }
        for (alias, unitSeconds) in [
            ("D", 86_400),
            ("h", 3_600),
            ("min", 60),
            ("s", 1),
        ] where seconds % unitSeconds == 0 {
            let multiple = seconds / unitSeconds
            return multiple == 1 ? alias : "\(multiple)\(alias)"
        }
        return nil
    }
}
