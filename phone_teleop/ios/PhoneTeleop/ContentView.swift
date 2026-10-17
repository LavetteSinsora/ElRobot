import SwiftUI
import UIKit

/// The grip (see ../README.md): screen facing LEFT, back camera RIGHT, the top (Dynamic
/// Island) pointing FORWARD, held in the right hand with the thumb on the screen.
///
/// ENGAGE is a toggle: tap once and the arm follows the phone, tap again and it holds where it
/// is. The app turns it off by itself whenever the Mac releases the arm (a guard fired, the
/// rollout ended), so it never shows "on" while the arm is not following.
/// GRIP is held: the jaw closes while your finger is on it and opens again when you let go.
struct ContentView: View {
    @StateObject private var streamer = PoseStreamer()
    @State private var engaged = false
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

            EngageToggle(on: $engaged)
                .frame(height: 150)
            GripPad { streamer.setButtons(grip: $0) }
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
            // The Mac let go (a guard, the end of the rollout) or is not driving yet: the
            // toggle follows, so a new tap is a deliberate new engagement.
            if engaged && (state == "released" || state == "inactive") {
                engaged = false
            }
        }
        .onChange(of: engaged) { on in
            streamer.setButtons(engage: on)
            UIImpactFeedbackGenerator(style: on ? .heavy : .light).impactOccurred()
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
            case "idle": return ("READY: tap ENGAGE to drive", .blue)
            case "released": return ("RELEASED by the Mac: tap ENGAGE to drive again", .red)
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

/// ENGAGE: tap to start following the phone, tap again to stop (the arm then holds).
struct EngageToggle: View {
    @Binding var on: Bool

    var body: some View {
        Button { on.toggle() } label: {
            ZStack {
                RoundedRectangle(cornerRadius: 24).fill(on ? Color.green : Color.green.opacity(0.2))
                VStack(spacing: 4) {
                    Text(on ? "ENGAGED" : "ENGAGE").font(.largeTitle.bold())
                    Text(on ? "tap to stop (the arm holds)" : "tap: the arm follows the phone")
                        .font(.subheadline)
                }
                .foregroundStyle(on ? Color.white : Color.green)
            }
        }
        .buttonStyle(.plain)
    }
}

/// GRIP: hold to close the jaw; let go and it opens again.
struct GripPad: View {
    let onChange: (Bool) -> Void
    @State private var pressed = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24).fill(pressed ? Color.orange : Color.orange.opacity(0.2))
            VStack(spacing: 4) {
                Text("GRIP").font(.largeTitle.bold())
                Text("hold to close, let go to open").font(.subheadline)
            }
            .foregroundStyle(pressed ? Color.white : Color.orange)
            HoldArea { p in
                pressed = p
                onChange(p)
            }
        }
    }
}

/// Reports true while any finger is on it, false when the last one lifts (a UIView, so it also
/// works while another finger is elsewhere on the screen).
struct HoldArea: UIViewRepresentable {
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
            let wasDown = !touchesDown.isEmpty
            touchesDown.subtract(touches)
            if wasDown && touchesDown.isEmpty { onChange?(false) }
        }
    }
}
