import SwiftUI
import WebKit

struct DashboardWebView: NSViewRepresentable {
    let url: URL
    let reloadID: UUID

    final class Coordinator {
        var lastReloadID: UUID?
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.setValue(false, forKey: "drawsBackground")
        webView.allowsBackForwardNavigationGestures = true
        webView.load(URLRequest(url: url))
        context.coordinator.lastReloadID = reloadID
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        if context.coordinator.lastReloadID != reloadID || webView.url != url {
            webView.load(URLRequest(url: url))
            context.coordinator.lastReloadID = reloadID
        }
    }
}
