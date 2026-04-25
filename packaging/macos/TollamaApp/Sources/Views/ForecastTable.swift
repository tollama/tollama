import SwiftUI

struct ForecastTable: View {
    let forecast: SeriesForecastDTO

    private var rows: [ForecastRow] {
        forecast.rows()
    }

    var body: some View {
        Table(rows) {
            TableColumn("Step") { row in
                Text(String(row.step))
            }
            TableColumn("Timestamp") { row in
                Text(row.timestamp ?? "—")
            }
            TableColumn("Mean") { row in
                Text(format(row.mean))
            }
            TableColumn("q10") { row in
                Text(row.q10.map(format) ?? "—")
            }
            TableColumn("q50") { row in
                Text(row.q50.map(format) ?? "—")
            }
            TableColumn("q90") { row in
                Text(row.q90.map(format) ?? "—")
            }
        }
        .frame(height: 240)
    }

    private func format(_ value: Double) -> String {
        String(format: "%.4g", value)
    }
}
