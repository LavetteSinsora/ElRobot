import Foundation
import ARKit
import Network
import QuartzCore

/// Runs ARKit world tracking and sends one UDP datagram per ARFrame (~60 Hz).
///
/// Packet v2 (UTF-8 JSON, one datagram):
/// {
///   "v": 2,                      // schema version
///   "seq": 1234,                 // monotonically increasing counter
///   "t": 5123.456,               // ARFrame capture time, seconds (iOS uptime clock)
///   "t_send": 5123.470,          // send time, same clock (echoed back for RTT)
///   "pose": [16 floats],         // 4x4 T_world_phone, ROW-major, metres
///   "track": "normal",           // ARKit tracking state
///   "world_epoch": 0,            // bumped whenever the world frame may have changed
///   "buttons": {"engage": false, "grip": false},
///   "name": "hand"               // this phone's name (the Mac's logs, its busy echo)
/// }
/// The Mac echoes {"seq", "t_send", "t_mac", "state", "why", "intervene"} back.
/// New fields can be added freely; receivers ignore unknown keys.
final class PoseStreamer: NSObject, ObservableObject, ARSessionDelegate {
    // UI-facing stats (main thread, throttled to ~5 Hz)
    @Published var trackingText = "—"
    @Published var sendRate: Double = 0
    @Published var rttMs: Double = .nan
    @Published var position = SIMD3<Float>(repeating: 0)
    @Published var connected = false
    @Published var epoch = 0
    // What the Mac says the arm is doing (from its echo; updated when it changes).
    @Published var macState = "—"      // inactive | idle | engaged | released | busy
    @Published var macWhy = ""
    /// An echo arrived within the last second: the Mac is there and answering.
    @Published var linkAlive = false

    private let session = ARSession()
    private let queue = DispatchQueue(label: "pose.stream", qos: .userInteractive)
    private var connection: NWConnection?          // touched only on `queue`

    // Shared between the main thread (buttons, reset) and `queue` (frames).
    private let lock = NSLock()
    private var engage = false
    private var grip = false
    private var worldEpoch = 0
    private var name = ""

    // Touched only on `queue`
    private var seq: UInt64 = 0
    private var rateCount = 0
    private var rateWindowStart = CACurrentMediaTime()
    private var rttEMA: Double = .nan
    private var lastTrack = ""
    private var lastMacState = ""
    private var lastMacWhy = ""
    private var lastEcho = -Double.infinity

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue   // keep ARFrame handling off the main thread
    }

    // MARK: Buttons

    func setName(_ n: String) {
        lock.lock()
        name = n
        lock.unlock()
    }

    func setButtons(engage e: Bool? = nil, grip g: Bool? = nil) {
        lock.lock()
        if let e { engage = e }
        if let g { grip = g }
        lock.unlock()
    }

    private func readState() -> (Bool, Bool, Int, String) {
        lock.lock(); defer { lock.unlock() }
        return (engage, grip, worldEpoch, name)
    }

    /// Poses before and after this are in different frames: the Mac releases the clutch.
    private func bumpEpoch() {
        lock.lock()
        worldEpoch += 1
        let e = worldEpoch
        lock.unlock()
        DispatchQueue.main.async { self.epoch = e }
    }

    // MARK: ARKit

    /// (Re)starts tracking. The world origin is the phone pose at this moment,
    /// with +Y opposite gravity.
    func startAR() {
        guard ARWorldTrackingConfiguration.isSupported else {
            trackingText = "ARKit world tracking unsupported"
            return
        }
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity
        bumpEpoch()
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let m = frame.camera.transform               // column-major simd_float4x4
        var pose = [Float]()
        pose.reserveCapacity(16)
        for r in 0..<4 { for c in 0..<4 { pose.append(m[c][r]) } }   // emit row-major

        let track = Self.describe(frame.camera.trackingState)
        if track == "limited:relocalizing" && lastTrack != track {
            bumpEpoch()                              // the map may be re-anchored after this
        }
        lastTrack = track
        let (e, g, epochNow, phoneName) = readState()
        seq &+= 1

        let packet: [String: Any] = [
            "v": 2,
            "seq": seq,
            "t": frame.timestamp,
            "t_send": CACurrentMediaTime(),
            "pose": pose,
            "track": track,
            "world_epoch": epochNow,
            "buttons": ["engage": e, "grip": g],
            "name": phoneName,
        ]
        if let conn = connection,
           let data = try? JSONSerialization.data(withJSONObject: packet) {
            conn.send(content: data, completion: .idempotent)   // fire-and-forget
        }

        // Throttled UI update
        rateCount += 1
        let now = CACurrentMediaTime()
        if now - rateWindowStart >= 0.25 {
            let rate = Double(rateCount) / (now - rateWindowStart)
            rateCount = 0
            rateWindowStart = now
            let pos = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            let rtt = rttEMA
            let alive = now - lastEcho < 1.0
            DispatchQueue.main.async {
                self.sendRate = rate
                self.trackingText = track
                self.position = pos
                self.rttMs = rtt
                self.linkAlive = alive
                if !alive && self.macState != "—" {
                    self.macState = "—"                  // no reply: nothing is known
                    self.macWhy = ""
                }
            }
            if !alive { lastMacState = "" }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        bumpEpoch()
        DispatchQueue.main.async { self.trackingText = "AR error: \(error.localizedDescription)" }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        bumpEpoch()
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        bumpEpoch()
    }

    static func describe(_ s: ARCamera.TrackingState) -> String {
        switch s {
        case .normal: return "normal"
        case .notAvailable: return "notAvailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited:initializing"
            case .excessiveMotion: return "limited:excessiveMotion"
            case .insufficientFeatures: return "limited:insufficientFeatures"
            case .relocalizing: return "limited:relocalizing"
            @unknown default: return "limited"
            }
        }
    }

    /// Leaving the Controller role: tracking and the link off.
    func stop() {
        session.pause()
        queue.async {
            self.connection?.cancel()
            self.connection = nil
            DispatchQueue.main.async {
                self.connected = false
                self.linkAlive = false
                self.macState = "—"
            }
        }
    }

    // MARK: Network

    /// A Mac by its address (typed in the settings).
    func connect(host: String, port: UInt16) {
        guard let p = NWEndpoint.Port(rawValue: port), !host.isEmpty else { return }
        connect(to: .hostPort(host: NWEndpoint.Host(host), port: p))
    }

    /// A Mac by its Bonjour service (`MacBrowser`): found again by name if its IP changes.
    func connect(to endpoint: NWEndpoint) {
        queue.async {
            self.connection?.cancel()
            self.rttEMA = .nan
            self.lastEcho = -Double.infinity
            let params = NWParameters.udp
            params.serviceClass = .interactiveVoice   // ask Wi-Fi for low-latency (voice) queueing
            let c = NWConnection(to: endpoint, using: params)
            c.stateUpdateHandler = { state in
                var ready = false
                if case .ready = state { ready = true }
                DispatchQueue.main.async { self.connected = ready }
            }
            c.start(queue: self.queue)
            self.connection = c
            self.receiveEcho(on: c)
        }
    }

    /// The Mac echoes {"seq", "t_send", "state", "why", ...} back. The round-trip time is
    /// measured on the phone's own clock, so no clock sync is needed; `state` is what the
    /// arm is doing for the operator.
    private func receiveEcho(on c: NWConnection) {
        c.receiveMessage { data, _, _, error in
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                self.lastEcho = CACurrentMediaTime()
                if let ts = obj["t_send"] as? Double {
                    let rtt = (CACurrentMediaTime() - ts) * 1000
                    self.rttEMA = self.rttEMA.isNaN ? rtt : 0.9 * self.rttEMA + 0.1 * rtt
                }
                let state = obj["state"] as? String ?? ""
                let why = obj["why"] as? String ?? ""
                if state != self.lastMacState || why != self.lastMacWhy {
                    self.lastMacState = state
                    self.lastMacWhy = why
                    DispatchQueue.main.async {
                        self.macState = state
                        self.macWhy = why
                    }
                }
            }
            if error == nil, self.connection === c { self.receiveEcho(on: c) }
        }
    }
}
