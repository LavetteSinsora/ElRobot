import AVFoundation
import SwiftUI
import UIKit

/// The Camera role: the phone on its stand, in landscape, seeing the arm. The preview is
/// exactly what the Mac gets (same orientation, same field of view). The Mac finds the phone by
/// its name (Settings; Bonjour), so nothing is typed on the Mac. Keep the app in front: iOS
/// stops the camera of an app in the background.
struct CameraView: View {
    @StateObject private var server = CameraServer()
    @AppStorage("phoneName") private var name = "exo"
    @AppStorage("cameraPort") private var port = 5006
    @State private var rotation = 0
    @State private var settings = false

    var body: some View {
        HStack(spacing: 12) {
            Preview(session: server.capture.session, rotation: rotation)
                .aspectRatio(16.0 / 9.0, contentMode: .fit)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Camera '\(name)'").font(.headline)
                    Spacer()
                    HoldGear { settings = true }
                }
                CameraBanner(server: server)
                Group {
                    Text(server.listening.isEmpty ? "Not listening"
                                                  : "Listening as \(server.listening)")
                    if !server.mode.isEmpty { Text(server.mode) }
                    if server.streaming {
                        Text(String(format: "%.0f fps, %.1f Mbit/s, dropped %d", server.fps,
                                    server.mbps, server.dropped))
                        Text(server.locked ? "Exposure, white balance, focus: locked"
                                           : "Exposure: auto")
                    }
                    if !server.error.isEmpty {
                        Text(server.error).foregroundStyle(.red)
                    }
                }
                .font(.system(.footnote, design: .monospaced))
                Spacer()
                Text("On the Mac: [cameras.<view>] device = \"\(name)\". Keep this app in "
                     + "front, the phone plugged in.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .frame(width: 300)
        }
        .padding()
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .sheet(isPresented: $settings, onDismiss: restart) { SettingsSheet() }
        .onAppear {
            server.capture.onRotation = { r in DispatchQueue.main.async { rotation = r } }
            server.capture.startPreview()
            restart()
        }
        .onDisappear { server.stop() }
    }

    /// (Re)advertise under the current name and port (after Settings).
    private func restart() {
        server.start(name: name, port: UInt16(clamping: port))
    }
}

struct CameraBanner: View {
    @ObservedObject var server: CameraServer

    var body: some View {
        let look: (String, Color) = {
            if server.streaming { return ("STREAMING to \(server.client)", .green) }
            if !server.client.isEmpty { return ("CONNECTED: \(server.client)", .blue) }
            if !server.listening.isEmpty { return ("WAITING for the Mac", .gray) }
            return ("NOT LISTENING", .red)
        }()
        Text(look.0)
            .font(.headline).foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 12).fill(look.1))
    }
}

/// The capture session's preview, rotated like the streamed frames (sensor landscape, or
/// turned 180).
struct Preview: UIViewRepresentable {
    let session: AVCaptureSession
    let rotation: Int

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.layer.videoGravity = .resizeAspect
        v.layer.session = session
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        v.rotation = rotation
        v.apply()
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        override var layer: AVCaptureVideoPreviewLayer { super.layer as! AVCaptureVideoPreviewLayer }
        var rotation = 0

        /// The stream's own rotation, whichever way the interface is turned: the screen shows
        /// the Mac's frame as the Mac gets it. Upside down here = upside down there
        /// (`phone_rotate = 180` in the profile, or turn the phone over).
        func apply() {
            guard let c = layer.connection else { return }
            Capture.rotate(c, by: rotation)
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            apply()
        }
    }
}
