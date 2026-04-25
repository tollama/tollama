import Charts
import SwiftUI

struct ForecastChart: View {
    let history: [ForecastHistoryPoint]
    let forecast: SeriesForecastDTO

    private var rows: [ForecastRow] {
        forecast.rows()
    }

    var body: some View {
        Chart {
            ForEach(history) { point in
                LineMark(
                    x: .value("Step", point.step),
                    y: .value("Value", point.value)
                )
                .foregroundStyle(.secondary)
            }

            ForEach(rows) { row in
                LineMark(
                    x: .value("Step", forecastX(row.step)),
                    y: .value("Mean", row.mean)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 2, dash: [6, 4]))
            }

            ForEach(rows) { row in
                if let q10 = row.q10, let q90 = row.q90 {
                    AreaMark(
                        x: .value("Step", forecastX(row.step)),
                        yStart: .value("q10", q10),
                        yEnd: .value("q90", q90)
                    )
                    .foregroundStyle(.blue.opacity(0.18))
                }
            }
        }
        .chartXAxisLabel("Step")
        .chartYAxisLabel("Value")
        .frame(minHeight: 320)
    }

    private func forecastX(_ step: Int) -> Int {
        history.count + step
    }
}
