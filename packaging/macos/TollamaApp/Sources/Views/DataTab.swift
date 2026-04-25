import AppKit
import SwiftUI

struct DataTab: View {
    @EnvironmentObject private var workspace: ForecastWorkspace

    var body: some View {
        GeometryReader { proxy in
            HStack(spacing: 0) {
                fileBrowserPane
                    .frame(width: sidebarWidth(for: proxy.size.width))
                    .frame(maxHeight: .infinity)
                    .background(Color(nsColor: .windowBackgroundColor))
                    .clipped()

                Divider()

                ScrollView {
                    previewPane
                        .padding(24)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.45))
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var fileBrowserPane: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Text("Data")
                    .font(.largeTitle.bold())
                    .lineLimit(1)
                Spacer(minLength: 12)
                Button("Choose Folder") {
                    chooseFolder()
                }
            }

            if let selectedFolder = workspace.selectedFolder {
                Text(selectedFolder.path)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }

            if workspace.isScanningFolder {
                ProgressView("Scanning CSV files")
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(workspace.csvFiles) { file in
                        fileRow(file)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(24)
    }

    private func sidebarWidth(for totalWidth: CGFloat) -> CGFloat {
        min(420, max(320, totalWidth * 0.34))
    }

    @ViewBuilder
    private var previewPane: some View {
        if let preview = workspace.csvPreview {
            VStack(alignment: .leading, spacing: 18) {
                Text(preview.url.lastPathComponent)
                    .font(.title2.bold())
                Text("\(preview.rowCount.map(String.init) ?? "Unknown") rows · \(preview.columns.count) columns")
                    .foregroundStyle(.secondary)

                roleBadges(preview)
                overrideControls(preview)
                sampleRows(preview)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                Text("No CSV selected")
                    .font(.title2.bold())
                Text("Choose a folder, then select one CSV file to preview.")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
    }

    private func fileRow(_ file: CSVFileItem) -> some View {
        Button {
            Task { await workspace.selectCSV(file) }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(file.name)
                        .font(.headline)
                    Text("\(file.sizeLabel) · \(file.rowCount.map(String.init) ?? "Unknown") rows")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(10)
            .background(
                workspace.selectedCSV?.id == file.id
                    ? Color.accentColor.opacity(0.18)
                    : Color.secondary.opacity(0.08)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }

    private func roleBadges(_ preview: CSVPreview) -> some View {
        FlowLayout(alignment: .leading, spacing: 8) {
            roleBadge("timestamp", preview.timestampColumn)
            roleBadge("target", preview.targetColumn)
            roleBadge("series_id", preview.seriesIDColumn)
            roleBadge("freq column", preview.freqColumn)
        }
    }

    private func roleBadge(_ role: String, _ column: String?) -> some View {
        Text("\(role): \(column ?? "not detected")")
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background((column == nil ? Color.orange : Color.green).opacity(0.16))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func overrideControls(_ preview: CSVPreview) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Column Overrides")
                .font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    Text("Timestamp")
                    TextField("auto", text: $workspace.timestampColumnOverride)
                }
                GridRow {
                    Text("Target")
                    TextField("auto", text: $workspace.targetColumnOverride)
                }
                GridRow {
                    Text("Series ID")
                    TextField("auto", text: $workspace.seriesIDColumnOverride)
                }
                GridRow {
                    Text("Frequency")
                    TextField("auto", text: $workspace.frequencyOverride)
                }
                GridRow {
                    Text("Freq Column")
                    TextField("auto", text: $workspace.freqColumnOverride)
                }
            }
            .textFieldStyle(.roundedBorder)
        }
    }

    private func sampleRows(_ preview: CSVPreview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preview")
                .font(.headline)
            ScrollView(.horizontal) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(preview.columns.joined(separator: "  |  "))
                        .font(.system(.caption, design: .monospaced).bold())
                    ForEach(Array(preview.sampleRows.enumerated()), id: \.offset) { _, row in
                        Text(row.joined(separator: "  |  "))
                            .font(.system(.caption, design: .monospaced))
                    }
                }
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            Task { await workspace.scanFolder(url) }
        }
    }
}

struct FlowLayout<Content: View>: View {
    let alignment: HorizontalAlignment
    let spacing: CGFloat
    @ViewBuilder let content: Content

    init(alignment: HorizontalAlignment, spacing: CGFloat, @ViewBuilder content: () -> Content) {
        self.alignment = alignment
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        VStack(alignment: alignment, spacing: spacing) {
            content
        }
    }
}
