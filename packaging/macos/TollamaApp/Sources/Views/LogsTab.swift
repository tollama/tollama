import SwiftUI

struct LogsTab: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        LogsTailView(logTail: model.logTail, minHeight: 520)
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct LogsTailView: View {
    let logTail: String
    var minHeight: CGFloat = 220

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Daemon Log Tail")
                .font(.headline)
            ScrollView {
                Text(logTail)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(minHeight: minHeight)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}
