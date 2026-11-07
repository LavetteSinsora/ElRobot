import AVFoundation
import CoreImage
import ImageIO

/// What the Mac asks for in its hello (elrobot_deploy's profile, `[cameras.<name>]`).
struct StreamConfig: Equatable {
    var width = 1280
    var height = 720
    var fps = 30.0
    var quality = 85            // JPEG, 1-100
    var lens = "wide"           // wide | ultrawide
    var exposure = "lock"       // lock (settle, then lock exposure / WB / focus) | auto
    var rotate = 0              // 0 | 180

    init() {}

    init(json: [String: Any]) {
        width = json["width"] as? Int ?? width
        height = json["height"] as? Int ?? height
        fps = (json["fps"] as? NSNumber)?.doubleValue ?? fps
        quality = json["quality"] as? Int ?? quality
        lens = json["lens"] as? String ?? lens
        exposure = json["exposure"] as? String ?? exposure
        rotate = json["rotate"] as? Int ?? rotate
    }
}

/// The back camera at the Mac's mode, one JPEG per frame with its capture time.
///
/// The session runs from launch (the preview, for aiming the stand) at the last mode asked for;
/// frames are encoded only while `wantsFrame()` says a Mac is connected and the link has room.
/// Frames come out in the sensor's own landscape orientation (rotated 180 on request): what the
/// preview shows is exactly what the Mac gets.
final class Capture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    /// Before the first encode: is anyone listening, is there room on the link? (video queue)
    var wantsFrame: () -> Bool = { false }
    /// (jpeg, capture time on CACurrentMediaTime's clock, exposure duration s, ISO) (video queue)
    var onFrame: ((Data, Double, Float, Float) -> Void)?
    /// Exposure, white balance and focus were locked (true) or released (false). (session queue)
    var onLocked: ((Bool) -> Void)?
    /// A frame `wantsFrame` took could not be encoded: give its place back. (video queue)
    var onEncodeFailed: (() -> Void)?
    /// The preview's connection, to show the same orientation as the stream.
    var onRotation: ((Int) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoQueue = DispatchQueue(label: "camera.video", qos: .userInteractive)
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?           // set on the session queue
    private var config: StreamConfig?
    private var lockWork: DispatchWorkItem?
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var quality: CGFloat = 0.85            // read on the video queue; a word, set rarely

    /// Seconds of continuous auto-exposure before `lock` locks it.
    static let settleS = 2.0

    /// Applies `cfg` (nothing if it is already running) and answers nil, or why not.
    func configure(_ cfg: StreamConfig, done: @escaping (String?) -> Void) {
        sessionQueue.async {
            if cfg == self.config, self.session.isRunning {
                // The same mode again (a new Mac connection): settle and lock afresh.
                self.unlockAndSchedule(cfg)
                done(nil)
                return
            }
            let err = self.apply(cfg)
            if err == nil {
                self.config = cfg
            }
            done(err)
        }
    }

    private func apply(_ cfg: StreamConfig) -> String? {
        let type: AVCaptureDevice.DeviceType =
            cfg.lens == "ultrawide" ? .builtInUltraWideCamera : .builtInWideAngleCamera
        guard let dev = AVCaptureDevice.default(type, for: .video, position: .back) else {
            return "this iPhone has no \(cfg.lens) back camera"
        }
        let fits = dev.formats.filter { f in
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return Int(d.width) == cfg.width && Int(d.height) == cfg.height
                && f.videoSupportedFrameRateRanges.contains {
                    $0.minFrameRate <= cfg.fps + 0.01 && cfg.fps - 0.01 <= $0.maxFrameRate
                }
        }
        let full = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        guard let format = fits.first(where: {
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == full
        }) ?? fits.first else {
            let have = Set(dev.formats.map { f -> String in
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let top = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                return "\(d.width)x\(d.height)@\(Int(top))"
            }).sorted().joined(separator: ", ")
            return "the \(cfg.lens) camera has no \(cfg.width)x\(cfg.height) mode at "
                + "\(cfg.fps) fps (it has \(have))"
        }

        session.beginConfiguration()
        session.sessionPreset = .inputPriority      // the device's own format, not a preset
        for input in session.inputs { session.removeInput(input) }
        guard let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            session.commitConfiguration()
            return "cannot open the \(cfg.lens) camera (in use by another app?)"
        }
        session.addInput(input)
        if session.outputs.isEmpty {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: full]
            output.alwaysDiscardsLateVideoFrames = true    // the newest frame wins
            output.setSampleBufferDelegate(self, queue: videoQueue)
            guard session.canAddOutput(output) else {
                session.commitConfiguration()
                return "cannot add the video output"
            }
            session.addOutput(output)
        }
        do {
            try dev.lockForConfiguration()
            dev.activeFormat = format
            let period = CMTime(value: 1000, timescale: CMTimeScale((cfg.fps * 1000).rounded()))
            dev.activeVideoMinFrameDuration = period
            dev.activeVideoMaxFrameDuration = period
            Self.setAuto(dev)
            dev.unlockForConfiguration()
        } catch {
            session.commitConfiguration()
            return "cannot configure the camera: \(error.localizedDescription)"
        }
        if let c = output.connection(with: .video) {
            if c.isVideoStabilizationSupported {
                c.preferredVideoStabilizationMode = .off    // no crop, no added latency
            }
            Self.rotate(c, by: cfg.rotate)
        }
        session.commitConfiguration()
        device = dev
        quality = CGFloat(max(1, min(100, cfg.quality))) / 100
        if !session.isRunning { session.startRunning() }
        onRotation?(cfg.rotate)
        unlockAndSchedule(cfg)
        return nil
    }

    func stop() {
        sessionQueue.async {
            self.lockWork?.cancel()
            if self.session.isRunning { self.session.stopRunning() }
            self.config = nil
        }
    }

    /// The preview at launch, before any Mac asks: the default mode, auto exposure.
    func startPreview() {
        configure(StreamConfig()) { _ in }
    }

    static func rotate(_ c: AVCaptureConnection, by degrees: Int) {
        if #available(iOS 17.0, *) {
            let angle = CGFloat(degrees == 180 ? 180 : 0)
            if c.isVideoRotationAngleSupported(angle) { c.videoRotationAngle = angle }
        } else if c.isVideoOrientationSupported {
            c.videoOrientation = degrees == 180 ? .landscapeLeft : .landscapeRight
        }
    }

    // MARK: exposure

    private static func setAuto(_ dev: AVCaptureDevice) {
        if dev.isExposureModeSupported(.continuousAutoExposure) {
            dev.exposureMode = .continuousAutoExposure
        }
        if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            dev.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        if dev.isFocusModeSupported(.continuousAutoFocus) {
            dev.focusMode = .continuousAutoFocus
        }
    }

    /// Back to auto; with `lock`, locked again after `settleS`. (session queue)
    private func unlockAndSchedule(_ cfg: StreamConfig) {
        lockWork?.cancel()
        if let dev = device, (try? dev.lockForConfiguration()) != nil {
            Self.setAuto(dev)
            dev.unlockForConfiguration()
        }
        onLocked?(false)
        guard cfg.exposure == "lock" else { return }
        let work = DispatchWorkItem { [weak self] in self?.lock() }
        lockWork = work
        sessionQueue.asyncAfter(deadline: .now() + Self.settleS, execute: work)
    }

    private func lock() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }
        if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
        if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
        dev.unlockForConfiguration()
        onLocked?(true)
    }

    // MARK: frames

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard wantsFrame(), let pixels = CMSampleBufferGetImageBuffer(sb) else { return }
        // The capture time on the host clock (the one CACurrentMediaTime reads, and the pong's).
        var pts = CMSampleBufferGetPresentationTimeStamp(sb)
        if let clock = session.synchronizationClock {
            pts = CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock())
        }
        let exposure = Float(device?.exposureDuration.seconds ?? .nan)
        let iso = device?.iso ?? .nan
        let options = [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption:
                       quality]
        guard let jpeg = context.jpegRepresentation(of: CIImage(cvPixelBuffer: pixels),
                                                    colorSpace: colorSpace,
                                                    options: options) else {
            onEncodeFailed?()
            return
        }
        onFrame?(jpeg, pts.seconds, exposure, iso)
    }
}
