import Foundation
import Network
import QuartzCore

/// The Mac's side of this is elrobot_deploy's `runtime/cameras/phone_camera.py`, whose
/// docstring is the protocol's reference. In short (TCP, big-endian, every message
/// `u32 length, u8 type, payload`):
///
///   Mac -> phone   h  hello  JSON {"v", "width", "height", "fps", "quality", "lens",
///                                  "exposure", "rotate"}
///                  p  ping   f64 t_mac
///   phone -> Mac   i  info   JSON {"v", "ok", "error", "model", "name", "width", "height",
///                                  "fps", "lens", "locked"}
///                  q  pong   f64 t_mac (echoed), f64 t_phone (when the ping arrived)
///                  f  frame  u32 seq, u32 dropped, f64 t_capture, f32 exposure s, f32 ISO,
///                            then the JPEG
///
/// A TCP listener advertised over Bonjour as `_elrobotcam._tcp` under `name`. One Mac at a
/// time: a new connection replaces the old one. At most `maxInFlight` frames wait on the link;
/// a frame that would make it more is dropped (and counted), so a slow WiFi costs frames, not
/// latency.
final class CameraServer: ObservableObject {
    static let serviceType = "_elrobotcam._tcp"
    static let protocolVersion = 1
    static let maxInFlight = 2

    @Published var listening = ""        // "" = not yet / failed
    @Published var client = ""           // the Mac's address, "" = none
    @Published var streaming = false
    @Published var locked = false
    @Published var fps: Double = 0
    @Published var mbps: Double = 0
    @Published var dropped = 0
    @Published var mode = ""
    @Published var error = ""

    let capture = Capture()
    private let queue = DispatchQueue(label: "camera.net", qos: .userInteractive)
    private var listener: NWListener?
    private var name = "exo"

    // On `queue`.
    private var conn: NWConnection?
    private var config = StreamConfig()
    private var seq: UInt32 = 0
    private var droppedCount: UInt32 = 0
    private var statFrames = 0
    private var statBytes = 0
    private var statStart = CACurrentMediaTime()

    // Read from the video queue before every encode.
    private let gate = NSLock()
    private var open = false
    private var inFlight = 0

    init() {
        capture.wantsFrame = { [weak self] in
            guard let self else { return false }
            self.gate.lock(); defer { self.gate.unlock() }
            if !self.open { return false }
            if self.inFlight >= Self.maxInFlight {
                self.queue.async { self.droppedCount &+= 1 }
                return false
            }
            self.inFlight += 1
            return true
        }
        capture.onFrame = { [weak self] jpeg, t, exposure, iso in
            self?.queue.async { self?.sendFrame(jpeg, t, exposure, iso) }
        }
        capture.onEncodeFailed = { [weak self] in self?.release() }
        capture.onLocked = { [weak self] locked in
            self?.queue.async {
                guard let self else { return }
                DispatchQueue.main.async { self.locked = locked }
                if self.conn != nil { self.sendInfo(ok: true, extra: ["locked": locked]) }
            }
        }
    }

    // MARK: listening

    func start(name: String, port: UInt16) {
        queue.async {
            self.name = name.isEmpty ? "exo" : name
            self.listener?.cancel()
            self.drop()
            let params = NWParameters.tcp
            if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options {
                tcp.noDelay = true
            }
            params.serviceClass = .interactiveVideo
            params.allowLocalEndpointReuse = true
            guard let p = NWEndpoint.Port(rawValue: port),
                  let l = try? NWListener(using: params, on: p) else {
                self.publish { $0.listening = ""; $0.error = "cannot listen on port \(port)" }
                return
            }
            l.service = NWListener.Service(name: self.name, type: Self.serviceType)
            l.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    self.publish { $0.listening = "'\(self.name)' on port \(port)"; $0.error = "" }
                case .failed(let e):
                    self.publish { $0.listening = ""; $0.error = "listener: \(e)" }
                default:
                    break
                }
            }
            l.newConnectionHandler = { [weak self] c in self?.accept(c) }
            l.start(queue: self.queue)
            self.listener = l
        }
    }

    /// Leaving the Camera role: not listening, no Mac, the camera off.
    func stop() {
        queue.async {
            self.listener?.cancel()
            self.listener = nil
            self.drop()
            self.publish { $0.listening = "" }
        }
        capture.stop()
    }

    private func accept(_ c: NWConnection) {
        conn?.cancel()
        setOpen(false)
        conn = c
        seq = 0
        droppedCount = 0
        c.stateUpdateHandler = { [weak self, weak c] state in
            guard let self, let c, self.conn === c else { return }
            switch state {
            case .failed, .cancelled:
                self.drop()
            default:
                break
            }
        }
        c.start(queue: queue)
        let who: String
        if case .hostPort(let host, _) = c.endpoint { who = "\(host)" } else { who = "\(c.endpoint)" }
        publish { $0.client = who; $0.error = "" }
        receive(on: c)
    }

    /// The Mac went away: back to waiting for one. (queue)
    private func drop() {
        conn?.cancel()
        conn = nil
        setOpen(false)
        publish { $0.client = ""; $0.streaming = false; $0.fps = 0; $0.mbps = 0 }
    }

    private func setOpen(_ on: Bool) {
        gate.lock()
        open = on
        inFlight = 0
        gate.unlock()
    }

    // MARK: receiving

    private func receive(on c: NWConnection) {
        c.receive(minimumIncompleteLength: 5, maximumLength: 5) { [weak self] head, _, _, err in
            guard let self, self.conn === c else { return }
            guard err == nil, let head, head.count == 5 else { self.drop(); return }
            let b = [UInt8](head)
            let n = Int(b[0]) << 24 | Int(b[1]) << 16 | Int(b[2]) << 8 | Int(b[3])
            let kind = b[4]
            let tArrived = CACurrentMediaTime()
            guard n <= 1 << 20 else { self.drop(); return }
            if n == 0 {
                self.handle(kind, Data(), tArrived, on: c)
                self.receive(on: c)
                return
            }
            c.receive(minimumIncompleteLength: n, maximumLength: n) { body, _, _, err in
                guard self.conn === c else { return }
                guard err == nil, let body, body.count == n else { self.drop(); return }
                self.handle(kind, body, tArrived, on: c)
                self.receive(on: c)
            }
        }
    }

    private func handle(_ kind: UInt8, _ body: Data, _ tArrived: Double, on c: NWConnection) {
        switch kind {
        case UInt8(ascii: "p"):
            // The pong: the ping's own bytes back, and when it arrived on our clock.
            var out = Data(body.prefix(8))
            out.append(Self.be(tArrived))
            send("q", out, on: c)
        case UInt8(ascii: "h"):
            let json = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]
            let cfg = StreamConfig(json: json)
            setOpen(false)
            capture.configure(cfg) { [weak self] err in
                self?.queue.async {
                    guard let self, self.conn === c else { return }
                    self.config = cfg
                    if let err {
                        self.sendInfo(ok: false, extra: ["error": err])
                        self.publish { $0.error = err; $0.mode = "" }
                        return
                    }
                    self.sendInfo(ok: true, extra: ["locked": false])
                    self.statStart = CACurrentMediaTime()
                    self.statFrames = 0
                    self.statBytes = 0
                    self.setOpen(true)
                    let mode = "\(cfg.width)x\(cfg.height) @ \(Int(cfg.fps)) fps, \(cfg.lens), "
                        + "q\(cfg.quality), exposure \(cfg.exposure)"
                    self.publish { $0.mode = mode; $0.streaming = true; $0.error = "" }
                }
            }
        default:
            break       // unknown: newer Mac; ignore
        }
    }

    // MARK: sending

    private func sendInfo(ok: Bool, extra: [String: Any]) {
        guard let c = conn else { return }
        var info: [String: Any] = [
            "v": Self.protocolVersion, "ok": ok, "model": Self.model(), "name": name,
            "width": config.width, "height": config.height, "fps": config.fps,
            "lens": config.lens,
        ]
        info.merge(extra) { _, new in new }
        if let data = try? JSONSerialization.data(withJSONObject: info) {
            send("i", data, on: c)
        }
    }

    private func sendFrame(_ jpeg: Data, _ t: Double, _ exposure: Float, _ iso: Float) {
        guard let c = conn else { release(); return }
        var head = Data()
        head.append(Self.be(seq))
        head.append(Self.be(droppedCount))
        head.append(Self.be(t))
        head.append(Self.be(exposure))
        head.append(Self.be(iso))
        seq &+= 1
        let payload = head + jpeg
        send("f", payload, on: c) { [weak self] in self?.release() }
        statFrames += 1
        statBytes += payload.count + 5
        let now = CACurrentMediaTime()
        if now - statStart >= 1.0 {
            let fps = Double(statFrames) / (now - statStart)
            let mbps = Double(statBytes) * 8 / 1e6 / (now - statStart)
            let dropped = Int(droppedCount)
            statStart = now
            statFrames = 0
            statBytes = 0
            publish { $0.fps = fps; $0.mbps = mbps; $0.dropped = dropped }
        }
    }

    private func release() {
        gate.lock()
        inFlight = max(0, inFlight - 1)
        gate.unlock()
    }

    private func send(_ kind: Character, _ payload: Data, on c: NWConnection,
                      sent: (() -> Void)? = nil) {
        var msg = Self.be(UInt32(payload.count))
        msg.append(kind.asciiValue!)
        msg.append(payload)
        c.send(content: msg, completion: .contentProcessed { _ in sent?() })
    }

    // MARK: helpers

    private func publish(_ change: @escaping (CameraServer) -> Void) {
        DispatchQueue.main.async { change(self) }
    }

    static func be(_ v: UInt32) -> Data { withUnsafeBytes(of: v.bigEndian) { Data($0) } }
    static func be(_ v: Double) -> Data { be64(v.bitPattern) }
    static func be(_ v: Float) -> Data { be(v.bitPattern) }
    static func be64(_ v: UInt64) -> Data { withUnsafeBytes(of: v.bigEndian) { Data($0) } }

    /// "iPhone15,3", not the marketing name.
    static func model() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
