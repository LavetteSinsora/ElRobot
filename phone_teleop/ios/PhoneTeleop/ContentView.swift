import SwiftUI
import UIKit

/// The grip (see ../README.md): screen facing LEFT, back camera RIGHT, the top (Dynamic
/// Island) pointing FORWARD, held in the right hand with the thumb on the screen. So the
/// screen's top is "forward" and its bottom is nearest the hand.
///
/// One thumb drives everything, on one pad: touch anywhere on it to ENGAGE (the arm follows
/// the phone while the thumb is down), and slide the thumb into the forward zone to also GRIP
/// (the jaw closes while it is there, and opens when it leaves). Lifting the thumb releases
/// the arm, which holds where it is; the jaw holds too.
struct ContentView: View {
    @StateObject private var streamer = PoseStreamer()
    @AppStorage("host") private var host = "192.168.1.100"
    @AppStorage("port") private var port = 5005

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
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

            StatusBanner(state: streamer.macState, why: streamer.macWhy)

            Group {
                Text("Link: \(streamer.connected ? "ready" : "not connected")   "
                     + "World epoch: \(streamer.epoch)")
                Text("Tracking: \(streamer.trackingText)")
                Text(String(format: "Rate: %.0f Hz", streamer.sendRate))
                Text(streamer.rttMs.isNaN ? "RTT: — (Mac not echoing)"
                                          : String(format: "RTT: %.1f ms  (one-way ≈ %.1f ms)",
                                                   streamer.rttMs, streamer.rttMs / 2))
                Text(String(format: "Pos [m]: x %+.3f  y %+.3f  z %+.3f",
                            streamer.position.x, streamer.position.y, streamer.position.z))
            }
            .font(.system(.footnote, design: .monospaced))

            Button("Reset world origin") { streamer.startAR() }

            ClutchPad { engage, grip in streamer.setButtons(engage: engage, grip: grip) }
                .frame(maxHeight: .infinity)
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            streamer.startAR()
            streamer.connect(host: host, port: UInt16(clamping: port))
        }
        .onChange(of: streamer.macState) { state in
            Haptics.on(state)
        }
    }
}

/// A tap in the hand when the arm starts following, and a warning when a guard lets it go.
enum Haptics {
    static func on(_ state: String) {
        switch state {
        case "engaged":
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        case "released":
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
        default:
            break
        }
    }
}

/// What the Mac says the arm is doing (its echo's `state` and `why`).
struct StatusBanner: View {
    let state: String
    let why: String

    var body: some View {
        let look: (String, Color) = {
            switch state {
            case "engaged": return ("ENGAGED: the arm follows the phone", .green)
            case "idle": return ("READY: put your thumb on the pad to drive", .blue)
            case "released": return ("RELEASED: lift your thumb, then press again", .red)
            case "inactive": return ("WAITING: the arm is not ready yet", .gray)
            default: return ("NO REPLY from the Mac", .gray)
            }
        }()
        VStack(alignment: .leading, spacing: 2) {
            Text(look.0).font(.headline).foregroundStyle(.white)
            if !why.isEmpty && state == "released" {
                Text(why).font(.caption).foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12).fill(look.1))
    }
}

/// The one-thumb pad: any touch engages; a touch in the forward (top) zone also grips.
struct ClutchPad: View {
    let onChange: (Bool, Bool) -> Void
    @State private var engaged = false
    @State private var gripping = false

    /// The forward fraction of the pad that also grips.
    static let gripFraction: CGFloat = 0.4

    var body: some View {
        GeometryReader { geo in
            ZStack {
                VStack(spacing: 0) {
                    ZStack {
                        Rectangle().fill(gripping ? Color.orange : Color.orange.opacity(0.2))
                        Text("GRIP\nslide the thumb here")
                            .multilineTextAlignment(.center)
                            .font(.title2.bold())
                            .foregroundStyle(gripping ? Color.white : Color.orange)
                    }
                    .frame(height: geo.size.height * Self.gripFraction)
                    ZStack {
                        Rectangle().fill(engaged ? Color.green : Color.green.opacity(0.2))
                        Text("ENGAGE\nhold to drive")
                            .multilineTextAlignment(.center)
                            .font(.title.bold())
                            .foregroundStyle(engaged ? Color.white : Color.green)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 24))
                TouchZones(gripFraction: Self.gripFraction) { e, g in
                    engaged = e
                    gripping = g
                    onChange(e, g)
                }
            }
        }
    }
}

/// Reports (engage, grip) whenever either changes: engage while any finger is down, grip while
/// any finger is in the top `gripFraction` of the view. Backed by a UIView for multi-touch and
/// for touches that slide between the zones.
struct TouchZones: UIViewRepresentable {
    let gripFraction: CGFloat
    let onChange: (Bool, Bool) -> Void

    func makeUIView(context: Context) -> PadView {
        let v = PadView()
        v.gripFraction = gripFraction
        v.onChange = onChange
        return v
    }

    func updateUIView(_ v: PadView, context: Context) {
        v.gripFraction = gripFraction
        v.onChange = onChange
    }

    final class PadView: UIView {
        var gripFraction: CGFloat = 0.4
        var onChange: ((Bool, Bool) -> Void)?
        private var touchesDown = Set<UITouch>()
        private var lastEngage = false
        private var lastGrip = false

        override init(frame: CGRect) {
            super.init(frame: frame)
            backgroundColor = .clear
            isMultipleTouchEnabled = true
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            touchesDown.formUnion(touches)
            report()
        }
        override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) { report() }
        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
            touchesDown.subtract(touches)
            report()
        }
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
            touchesDown.subtract(touches)
            report()
        }

        private func report() {
            let engage = !touchesDown.isEmpty
            let limit = bounds.height * gripFraction
            let grip = touchesDown.contains { $0.location(in: self).y < limit }
            if engage != lastEngage || grip != lastGrip {
                lastEngage = engage
                lastGrip = grip
                onChange?(engage, grip)
            }
        }
    }
}
