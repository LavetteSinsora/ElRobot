import Foundation
import ARKit
import Network
import QuartzCore

/// Runs ARKit world tracking and sends one UDP datagram per ARFrame (~60 Hz).
///
/// Packet (UTF-8 JSON, one datagram):
/// {
///   "v": 1,                      // schema version
///   "seq": 1234,                 // monotonically increasing counter
///   "t": 5123.456,               // ARFrame capture time, seconds (iOS uptime clock)
///   "t_send": 5123.470,          // send time, same clock (echoed back for RTT)
///   "pose": [16 floats],         // 4x4 T_world_phone, ROW-major, metres
///   "track": "normal",           // ARKit tracking state
///   "buttons": {"engage": false, "grip": false}
/// }
/// New fields can be added freely; receivers should ignore unknown keys.
final class PoseStreamer: NSObject, ObservableObject, ARSessionDelegate {
    // UI-facing stats (main thread, throttled to ~5 Hz)
    @Published var trackingText = "—"
    @Published var sendRate: Double = 0
    @Published var rttMs: Double = .nan
    @Published var position = SIMD3<Float>(repeating: 0)
    @Published var connected = false

    private let session = ARSession()
    private let queue = DispatchQueue(label: "pose.stream", qos: .userInteractive)
    private var connection: NWConnection?          // touched only on `queue`

    private let lock = NSLock()
    private var engage = false
    private var grip = false

    // Stats, touched only on `queue`
    private var seq: UInt64 = 0
    private var rateCount = 0
    private var rateWindowStart = CACurrentMediaTime()
    private var rttEMA: Double = .nan

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue   // keep ARFrame handling off the main thread
    }

    // MARK: Buttons

    func setButtons(engage e: Bool? = nil, grip g: Bool? = nil) {
        lock.lock()
        if let e { engage = e }
        if let g { grip = g }
        lock.unlock()
    }

    private func readButtons() -> (Bool, Bool) {
        lock.lock(); defer { lock.unlock() }
        return (engage, grip)
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
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let m = frame.camera.transform               // column-major simd_float4x4
        var pose = [Float]()
        pose.reserveCapacity(16)
        for r in 0..<4 { for c in 0..<4 { pose.append(m[c][r]) } }   // emit row-major

        let (e, g) = readButtons()
        let track = Self.describe(frame.camera.trackingState)
        seq &+= 1

        let packet: [String: Any] = [
            "v": 1,
            "seq": seq,
            "t": frame.timestamp,
            "t_send": CACurrentMediaTime(),
            "pose": pose,
            "track": track,
            "buttons": ["engage": e, "grip": g],
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
            DispatchQueue.main.async {
                self.sendRate = rate
                self.trackingText = track
                self.position = pos
                self.rttMs = rtt
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.trackingText = "AR error: \(error.localizedDescription)" }
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

    // MARK: Network

    func connect(host: String, port: UInt16) {
        queue.async {
            self.connection?.cancel()
            self.rttEMA = .nan
            guard let p = NWEndpoint.Port(rawValue: port), !host.isEmpty else { return }
            let params = NWParameters.udp
            params.serviceClass = .interactiveVoice   // ask Wi-Fi for low-latency (voice) queueing
            let c = NWConnection(host: NWEndpoint.Host(host), port: p, using: params)
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

    /// The receiver echoes {"seq", "t_send"} back; we measure round-trip time on the
    /// phone's own clock, so no clock sync between phone and Mac is needed.
    private func receiveEcho(on c: NWConnection) {
        c.receiveMessage { data, _, _, error in
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let ts = obj["t_send"] as? Double {
                let rtt = (CACurrentMediaTime() - ts) * 1000
                self.rttEMA = self.rttEMA.isNaN ? rtt : 0.9 * self.rttEMA + 0.1 * rtt
            }
            if error == nil, self.connection === c { self.receiveEcho(on: c) }
        }
    }
}
