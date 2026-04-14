import AppKit
import SwiftUI

@main
struct TollamaDesktopApp: App {
    @StateObject private var model = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 1280, minHeight: 800)
                .task {
                    await model.bootstrap()
                }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    model.handleAppTermination()
                }
        }
        .windowStyle(.hiddenTitleBar)
    }
}
