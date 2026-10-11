import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var streamer = PoseStreamer()
    @AppStorage("host") private var host = "192.168.1.100"
    @AppStorage("port") private var port = 5005

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                TextField("Mac IP", text: $host)
                    .keyboardType(.numbersAndPunctuation)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .textFieldStyle(.roundedBorder)
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .keyboardType(.numberPad)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 72)
                Button("Connect") {
                    streamer.connect(host: host, port: UInt16(clamping: port))
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                    to: nil, from: nil, for: nil)
                }
                .buttonStyle(.borderedProminent)
            }

            Group {
                Text("Link: \(streamer.connected ? "ready" : "not connected")")
                Text("Tracking: \(streamer.trackingText)")
                Text(String(format: "Rate: %.0f Hz", streamer.sendRate))
                Text(streamer.rttMs.isNaN ? "RTT: — (receiver not echoing)"
                                          : String(format: "RTT: %.1f ms  (one-way ≈ %.1f ms)",
                                                   streamer.rttMs, streamer.rttMs / 2))
                Text(String(format: "Pos [m]: x %+.3f  y %+.3f  z %+.3f",
                            streamer.position.x, streamer.position.y, streamer.position.z))
            }
            .font(.system(.body, design: .monospaced))

            Button("Reset world origin") { streamer.startAR() }

            Spacer()

            HStack(spacing: 16) {
                HoldButton(title: "GRIP", color: .orange) { streamer.setButtons(grip: $0) }
                HoldButton(title: "ENGAGE", color: .green) { streamer.setButtons(engage: $0) }
            }
            .frame(height: 240)
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            streamer.startAR()
            streamer.connect(host: host, port: UInt16(clamping: port))
        }
    }
}

/// Press-and-hold pad: reports true while any finger is on it, false on release.
/// Backed by a UIView so both pads can be held simultaneously.
struct HoldButton: View {
    let title: String
    let color: Color
    let onChange: (Bool) -> Void
    @State private var pressed = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24).fill(pressed ? color : color.opacity(0.2))
            Text(title).font(.title.bold()).foregroundStyle(pressed ? .white : color)
            TouchPad { p in
                pressed = p
                onChange(p)
            }
        }
    }
}

struct TouchPad: UIViewRepresentable {
    let onChange: (Bool) -> Void

    func makeUIView(context: Context) -> PadView {
        let v = PadView()
        v.onChange = onChange
        return v
    }

    func updateUIView(_ v: PadView, context: Context) { v.onChange = onChange }

    final class PadView: UIView {
        var onChange: ((Bool) -> Void)?
        private var touchesDown = Set<UITouch>()

        override init(frame: CGRect) {
            super.init(frame: frame)
            backgroundColor = .clear
            isMultipleTouchEnabled = true
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            let wasIdle = touchesDown.isEmpty
            touchesDown.formUnion(touches)
            if wasIdle { onChange?(true) }
        }
        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) { release(touches) }
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { release(touches) }

        private func release(_ touches: Set<UITouch>) {
            touchesDown.subtract(touches)
            if touchesDown.isEmpty { onChange?(false) }
        }
    }
}
