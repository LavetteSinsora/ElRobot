import SwiftUI
import UIKit

/// The Controller role. The grip (see ../README.md): screen facing LEFT, back camera RIGHT, the
/// top (Dynamic Island) pointing FORWARD, held in the right hand with the thumb on the screen.
///
/// Built not to be mispressed while driving:
///   - ENGAGE starts only on a deliberate HOLD (`EngagePad.holdS`, a bar fills, a tap in the
///     hand); stopping is one TAP anywhere on it. Hard to start by accident, easy to stop.
///   - ENGAGE is greyed out unless the Mac says it is ready and tracking is normal.
///   - Nothing else on the screen changes the session: the Mac, the world-origin reset and the
///     role are in Settings, behind a HELD gear that is locked while engaged.
///   - Edge swipes (home, Control Centre) need a second swipe: a stray thumb does not leave the
///     app.
///   - The app turns ENGAGE off by itself whenever the Mac releases the arm, stops answering, or
///     says another phone drives, so it never shows "on" while the arm is not following.
/// GRIP is held: the jaw closes while your finger is on it and opens again when you let go.
struct ControllerView: View {
    @StateObject private var streamer = PoseStreamer()
    @StateObject private var browser = MacBrowser()
    @State private var engaged = false
    @State private var settings = false
    @State private var connectedTo = ""             // the Mac (or address) the link points at
    @AppStorage("phoneName") private var phoneName = "hand"
    @AppStorage("macName") private var macName = ""
    @AppStorage("manualHost") private var manualHost = ""
    @AppStorage("port") private var port = 5005

    private var ready: Bool {
        streamer.linkAlive && streamer.trackingText == "normal"
            && (streamer.macState == "idle" || streamer.macState == "released")
    }

    private var notReadyWhy: String {
        if connectedTo.isEmpty { return "choose your Mac above" }
        if !streamer.linkAlive { return "no reply from \(connectedTo)" }
        if streamer.macState == "busy" { return "another phone is driving" }
        if streamer.macState == "inactive" { return "press Start on the Mac first" }
        if streamer.trackingText != "normal" { return "tracking: \(streamer.trackingText)" }
        return ""
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Circle().fill(streamer.linkAlive ? Color.green : Color.red).frame(width: 10)
                Text(connectedTo.isEmpty ? "No Mac yet" : connectedTo).font(.headline)
                    .lineLimit(1)
                Spacer()
                Text(phoneName).font(.subheadline).foregroundStyle(.secondary)
                HoldGear(disabled: engaged) { settings = true }
            }

            if connectedTo.isEmpty {
                MacList(browser: browser) { mac in
                    macName = mac.name
                    connect(force: true)
                }
            }

            StatusBanner(state: streamer.macState, why: streamer.macWhy,
                         alive: streamer.linkAlive)

            Group {
                Text("Tracking: \(streamer.trackingText)")
                Text(streamer.rttMs.isNaN ? String(format: "Rate: %.0f Hz", streamer.sendRate)
                     : String(format: "Rate: %.0f Hz   RTT: %.1f ms", streamer.sendRate,
                              streamer.rttMs))
            }
            .font(.system(.footnote, design: .monospaced))

            EngagePad(engaged: engaged, enabled: ready, hint: notReadyWhy,
                      onEngage: { engaged = true }, onRelease: { engaged = false })
                .frame(height: 170)
            GripPad(active: engaged) { streamer.setButtons(grip: $0) }
                .frame(maxHeight: .infinity)
        }
        .padding()
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .defersSystemGestures(on: .all)
        .sheet(isPresented: $settings) {
            SettingsSheet(resetWorld: { streamer.startAR() },
                          reconnect: { connect(force: true) })
        }
        .onAppear {
            streamer.setName(phoneName)
            streamer.startAR()
            browser.start()
            connect(force: true)
        }
        .onDisappear {
            streamer.stop()
            browser.stop()
        }
        .onChange(of: browser.macs) { _ in
            // A Mac appeared (a session started) or went: reconnect by name, never mid-drive.
            if !engaged && (!streamer.linkAlive || connectedTo.isEmpty) { connect(force: true) }
        }
        .onChange(of: phoneName) { streamer.setName($0) }
        .onChange(of: streamer.macState) { state in
            Haptics.on(state)
            if engaged && ["released", "inactive", "busy", "—"].contains(state) {
                engaged = false
            }
        }
        .onChange(of: engaged) { on in
            streamer.setButtons(engage: on)
            if on { settings = false }
        }
    }

    /// The Mac to talk to: an address typed in Settings, else the remembered Mac by name, else
    /// the only Mac on the WiFi (then remembered). Several and none remembered: the list.
    private func connect(force: Bool) {
        streamer.setName(phoneName)
        let host = manualHost.trimmingCharacters(in: .whitespaces)
        if !host.isEmpty {
            if force || connectedTo != host {
                streamer.connect(host: host, port: UInt16(clamping: port))
                connectedTo = host
            }
            return
        }
        let mac = browser.macs.first(where: { $0.name == macName })
            ?? (macName.isEmpty && browser.macs.count == 1 ? browser.macs.first : nil)
        guard let mac else {
            if macName.isEmpty || !browser.macs.isEmpty { connectedTo = "" }
            return
        }
        if force || connectedTo != mac.name {
            streamer.connect(to: mac.endpoint)
            connectedTo = mac.name
        }
        macName = mac.name
    }
}

/// The Macs on this WiFi: tap one to drive it (remembered for next time).
struct MacList: View {
    @ObservedObject var browser: MacBrowser
    let pick: (MacBrowser.Mac) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if browser.macs.isEmpty {
                Text("Looking for Macs… Start `elrobot_deploy run` (or `phone-bench`) on the Mac, "
                     + "on this WiFi.")
                    .font(.subheadline).foregroundStyle(.secondary)
                if !browser.error.isEmpty {
                    Text(browser.error).font(.caption).foregroundStyle(.red)
                }
            } else {
                Text("Choose your Mac").font(.subheadline.bold())
                ForEach(browser.macs) { mac in
                    Button { pick(mac) } label: {
                        HStack {
                            Image(systemName: "laptopcomputer")
                            Text(mac.name)
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding(12)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Color.blue.opacity(0.15)))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

/// A tap in the hand when the arm starts following, and a warning when a guard lets it go.
enum Haptics {
    static func on(_ state: String) {
        switch state {
        case "engaged":
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        case "released", "busy":
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
    let alive: Bool

    var body: some View {
        let look: (String, Color) = {
            if !alive { return ("NO REPLY from the Mac", .gray) }
            switch state {
            case "engaged": return ("ENGAGED: the arm follows the phone", .green)
            case "idle": return ("READY: hold ENGAGE to drive", .blue)
            case "released": return ("RELEASED by the Mac: hold ENGAGE to drive again", .red)
            case "inactive": return ("WAITING: the arm is not ready yet", .gray)
            case "busy": return ("ANOTHER PHONE is driving", .orange)
            default: return ("NO REPLY from the Mac", .gray)
            }
        }()
        VStack(alignment: .leading, spacing: 2) {
            Text(look.0).font(.headline).foregroundStyle(.white)
            if !why.isEmpty && (state == "released" || state == "busy") {
                Text(why).font(.caption).foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12).fill(look.1))
    }
}

/// ENGAGE: HOLD to start following the phone, TAP to stop (the arm then holds).
struct EngagePad: View {
    let engaged: Bool
    let enabled: Bool
    let hint: String
    let onEngage: () -> Void
    let onRelease: () -> Void
    static let holdS = 0.4
    @State private var holding = false

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 24)
                .fill(engaged ? Color.green : (enabled ? Color.green.opacity(0.2)
                                                       : Color.gray.opacity(0.15)))
            // The hold's progress: the bar fills over holdS; let go early and nothing happens.
            GeometryReader { g in
                RoundedRectangle(cornerRadius: 24)
                    .fill(Color.green.opacity(0.6))
                    .frame(width: holding && !engaged ? g.size.width : 0)
                    .animation(holding ? .linear(duration: Self.holdS) : .none, value: holding)
            }
            VStack(spacing: 4) {
                Text(engaged ? "ENGAGED" : "HOLD TO ENGAGE").font(.largeTitle.bold())
                Text(engaged ? "tap to stop (the arm holds)"
                     : (enabled ? "hold: the arm follows the phone" : hint))
                    .font(.subheadline)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .foregroundStyle(engaged ? Color.white : (enabled ? Color.green : Color.gray))
            PressArea(holdS: Self.holdS, engaged: engaged, enabled: enabled,
                      began: { holding = true }, cancelled: { holding = false },
                      engage: {
                          holding = false
                          UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
                          onEngage()
                      },
                      release: {
                          UIImpactFeedbackGenerator(style: .light).impactOccurred()
                          onRelease()
                      })
        }
        .clipShape(RoundedRectangle(cornerRadius: 24))
    }
}

/// The touch logic of ENGAGE, in UIKit so it sees each finger: one touch at a time; a touch
/// that lands while engaged releases at once (and is spent: it cannot re-engage); a touch that
/// lands while disengaged engages only if it stays down, inside, for `holdS`.
struct PressArea: UIViewRepresentable {
    let holdS: Double
    let engaged: Bool
    let enabled: Bool
    let began: () -> Void
    let cancelled: () -> Void
    let engage: () -> Void
    let release: () -> Void

    func makeUIView(context: Context) -> PressView { update(PressView()) }
    func updateUIView(_ v: PressView, context: Context) { _ = update(v) }

    private func update(_ v: PressView) -> PressView {
        v.holdS = holdS
        v.engaged = engaged
        v.enabled = enabled
        v.began = began
        v.cancelled = cancelled
        v.engage = engage
        v.release = release
        return v
    }

    final class PressView: UIView {
        var holdS = 0.4
        var engaged = false
        var enabled = false
        var began: () -> Void = {}
        var cancelled: () -> Void = {}
        var engage: () -> Void = {}
        var release: () -> Void = {}
        private var touch: UITouch?
        private var spent = false
        private var timer: DispatchWorkItem?

        override init(frame: CGRect) {
            super.init(frame: frame)
            backgroundColor = .clear
            isMultipleTouchEnabled = true
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard touch == nil, let t = touches.first else { return }
            touch = t
            spent = false
            if engaged {
                spent = true
                release()
                return
            }
            guard enabled else { spent = true; return }
            began()
            let work = DispatchWorkItem { [weak self] in
                guard let self, self.touch === t, !self.spent, self.enabled else { return }
                self.spent = true
                self.engage()
            }
            timer = work
            DispatchQueue.main.asyncAfter(deadline: .now() + holdS, execute: work)
        }

        override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard let t = touch, touches.contains(t), !spent else { return }
            if !bounds.insetBy(dx: -20, dy: -20).contains(t.location(in: self)) {
                abort()                                   // slid off: not a deliberate hold
            }
        }

        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) { end(touches) }
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { end(touches) }

        private func end(_ touches: Set<UITouch>) {
            guard let t = touch, touches.contains(t) else { return }
            if !spent { abort() }
            touch = nil
        }

        private func abort() {
            timer?.cancel()
            timer = nil
            spent = true
            cancelled()
        }
    }
}

/// GRIP: hold to close the jaw; let go and it opens again. Dimmed while not engaged (the jaw
/// then holds whatever is pressed).
struct GripPad: View {
    let active: Bool
    let onChange: (Bool) -> Void
    @State private var pressed = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24)
                .fill(pressed ? Color.orange : Color.orange.opacity(active ? 0.2 : 0.08))
            VStack(spacing: 4) {
                Text("GRIP").font(.largeTitle.bold())
                Text("hold to close, let go to open").font(.subheadline)
            }
            .foregroundStyle(pressed ? Color.white : Color.orange.opacity(active ? 1 : 0.5))
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
