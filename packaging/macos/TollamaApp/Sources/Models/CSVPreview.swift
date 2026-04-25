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
    let history: [ForecastHistoryPoint]
}

enum CSVSniffer {
    static let timestampCandidates = ["timestamp", "timestamps", "ds", "time"]
    static let seriesIDCandidates = ["id", "series_id", "unique_id"]
    static let targetCandidates = ["target", "value", "y"]
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
                    rowCount: try? countDataRows(in: fileURL)
                )
            )
        }
        return files.sorted { left, right in
            left.name.localizedCaseInsensitiveCompare(right.name) == .orderedAscending
        }
    }

    static func preview(url: URL, maxRows: Int = 200) throws -> CSVPreview {
        let data = try readPrefix(url: url, maxBytes: 512 * 1024)
        let text = String(decoding: data, as: UTF8.self)
        let parsedRows = parse(text, maxRows: maxRows + 1)
        let columns = parsedRows.first?.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) } ?? []
        let dataRows = Array(parsedRows.dropFirst())
        let timestampColumn = firstExistingColumn(columns: columns, candidates: timestampCandidates)
        let seriesIDColumn = firstExistingColumn(columns: columns, candidates: seriesIDCandidates)
        let freqColumn = firstExistingColumn(columns: columns, candidates: freqCandidates)
        let targetColumn = firstExistingColumn(columns: columns, candidates: targetCandidates)
            ?? firstNumericColumn(columns: columns, rows: dataRows, excluding: [
                timestampColumn,
                seriesIDColumn,
                freqColumn,
            ])

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
            rowCount: try? countDataRows(in: url),
            timestampColumn: timestampColumn,
            targetColumn: targetColumn,
            seriesIDColumn: seriesIDColumn,
            freqColumn: freqColumn,
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
        for column in columns where lookup[column.lowercased()] == nil {
            lookup[column.lowercased()] = column
        }
        for candidate in candidates {
            if let match = lookup[candidate] {
                return match
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
}
