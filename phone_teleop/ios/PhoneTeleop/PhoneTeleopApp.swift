import SwiftUI
import UIKit

/// One app, two roles, chosen once per phone and remembered:
///
///   Controller   in your hand: ARKit pose to the Mac, ENGAGE and GRIP (`ControllerView`)
///   Camera       on a stand: the back camera to the Mac (`CameraView`)
///
/// Each phone has a name (Settings): the Mac's logs say which phone drives, and a profile's
/// `[cameras.<name>] device` picks a camera phone by it. Any number of phones can run the app;
/// only one Controller drives at a time (the Mac tells the others "busy").
@main
struct PhoneTeleopApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate

    var body: some Scene {
        WindowGroup { RootView() }
    }
}

enum Role: String {
    case controller, camera

    var orientations: UIInterfaceOrientationMask {
        self == .controller ? .portrait : .landscape
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    /// What the current role allows; the scene is asked to rotate when it changes.
    static var orientations: UIInterfaceOrientationMask = .portrait

    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        Self.orientations
    }

    static func lock(_ mask: UIInterfaceOrientationMask) {
        orientations = mask
        for case let scene as UIWindowScene in UIApplication.shared.connectedScenes {
            scene.requestGeometryUpdate(.iOS(interfaceOrientations: mask)) { _ in }
            scene.windows.first?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        }
    }
}

struct RootView: View {
    @AppStorage("role") private var roleRaw = ""
    @AppStorage("phoneName") private var phoneName = ""

    var body: some View {
        Group {
            switch Role(rawValue: roleRaw) {
            case .controller: ControllerView()
            case .camera: CameraView()
            case nil: RolePicker { role, name in
                phoneName = name
                roleRaw = role.rawValue
            }
            }
        }
        .onAppear { apply() }
        .onChange(of: roleRaw) { _ in apply() }
    }

    private func apply() {
        AppDelegate.lock(Role(rawValue: roleRaw)?.orientations ?? .portrait)
        UIApplication.shared.isIdleTimerDisabled = Role(rawValue: roleRaw) != nil
    }
}

/// First launch: what is this phone for? (Settings can change it later.)
struct RolePicker: View {
    let choose: (Role, String) -> Void
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("What is this phone for?").font(.largeTitle.bold())
            Text("You can change this later in Settings.").foregroundStyle(.secondary)
            TextField("This phone's name (e.g. hand, exo, side)", text: $name)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
            RoleCard(title: "Controller", icon: "hand.raised.fill", color: .green,
                     text: "In your hand. Drives the arm: hold ENGAGE, move the phone, hold GRIP.") {
                choose(.controller, clean(name, or: "hand"))
            }
            RoleCard(title: "Camera", icon: "video.fill", color: .blue,
                     text: "On a stand, watching the arm. Streams its back camera to the Mac.") {
                choose(.camera, clean(name, or: "exo"))
            }
            Spacer()
        }
        .padding()
    }

    private func clean(_ s: String, or fallback: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? fallback : t
    }
}

struct RoleCard: View {
    let title: String
    let icon: String
    let color: Color
    let text: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon).font(.system(size: 36)).frame(width: 52)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.title2.bold())
                    Text(text).font(.subheadline).multilineTextAlignment(.leading)
                }
                Spacer()
            }
            .foregroundStyle(.white)
            .padding(18)
            .background(RoundedRectangle(cornerRadius: 18).fill(color))
        }
        .buttonStyle(.plain)
    }
}

/// The gear: opens Settings only on a deliberate hold (`holdS`), never on a brush of the thumb.
/// Disabled outright while the arm follows the phone.
struct HoldGear: View {
    var disabled = false
    let open: () -> Void
    static let holdS = 0.6

    var body: some View {
        Image(systemName: "gearshape.fill")
            .font(.title2)
            .foregroundStyle(disabled ? Color.secondary.opacity(0.3) : Color.secondary)
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
            .onLongPressGesture(minimumDuration: Self.holdS) {
                if !disabled {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    open()
                }
            }
            .accessibilityLabel("Settings (hold)")
    }
}

/// Everything that is not driving: the phone's name and role, the Mac, the world origin.
/// Opened by holding the gear; the Controller keeps it closed while engaged.
struct SettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("role") private var roleRaw = ""
    @AppStorage("phoneName") private var phoneName = ""
    @AppStorage("macName") private var macName = ""
    @AppStorage("manualHost") private var manualHost = ""
    @AppStorage("port") private var port = 5005
    @AppStorage("cameraPort") private var cameraPort = 5006
    var resetWorld: (() -> Void)?
    var reconnect: (() -> Void)?
    @State private var confirmRole = false

    var body: some View {
        NavigationStack {
            Form {
                Section("This phone") {
                    TextField("Name", text: $phoneName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    LabeledContent("Role", value: roleRaw.capitalized)
                    Button("Change role…") { confirmRole = true }
                }
                if roleRaw == Role.controller.rawValue {
                    Section {
                        LabeledContent("Mac", value: macName.isEmpty ? "the one found" : macName)
                        Button("Forget this Mac (choose again)") {
                            macName = ""
                            reconnect?()
                            dismiss()
                        }
                        TextField("Or its IP address (leave empty: by name)", text: $manualHost)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Stepper("Port \(port)", value: $port, in: 1024...65535)
                    } header: { Text("Mac") } footer: {
                        Text("Macs running a session are found by name on this WiFi. An IP is only "
                             + "needed if the Mac is not listed.")
                    }
                    Section {
                        Button("Reset world origin") {
                            resetWorld?()
                            dismiss()
                        }
                    } footer: {
                        Text("Restarts ARKit tracking. The Mac releases the arm if it was following.")
                    }
                } else {
                    Section("Camera") {
                        Stepper("Port \(cameraPort)", value: $cameraPort, in: 1024...65535)
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        reconnect?()
                        dismiss()
                    }
                }
            }
            .confirmationDialog("Change this phone's role?", isPresented: $confirmRole,
                                titleVisibility: .visible) {
                Button("Controller (in your hand)") { switchRole(.controller) }
                Button("Camera (on a stand)") { switchRole(.camera) }
                Button("Cancel", role: .cancel) {}
            }
        }
    }

    private func switchRole(_ role: Role) {
        dismiss()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { roleRaw = role.rawValue }
    }
}
