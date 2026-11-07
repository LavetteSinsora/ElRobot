import Foundation
import Network

/// Every Mac running an elrobot_deploy session (or `phone-bench`) on this WiFi: P11 advertises
/// its teleop port over Bonjour as `_elrobot._udp` under the Mac's computer name
/// (`runtime/teleop/discovery.py`). The Controller connects by that name, so no IP is typed
/// and a Mac whose address changed is found again.
final class MacBrowser: ObservableObject {
    static let serviceType = "_elrobot._udp"

    struct Mac: Identifiable, Hashable {
        let name: String
        let endpoint: NWEndpoint
        var id: String { name }
    }

    @Published var macs: [Mac] = []
    @Published var error = ""

    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let b = NWBrowser(for: .bonjour(type: Self.serviceType, domain: nil), using: .udp)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let list = results.compactMap { r -> Mac? in
                if case .service(let name, _, _, _) = r.endpoint {
                    return Mac(name: name, endpoint: r.endpoint)
                }
                return nil
            }
            self?.macs = list.sorted { $0.name < $1.name }
        }
        b.stateUpdateHandler = { [weak self] state in
            if case .failed(let e) = state {
                self?.error = "cannot look for Macs: \(e)"
            } else if case .waiting(let e) = state {
                // Typically Local Network access denied: Settings > Privacy > Local Network.
                self?.error = "waiting: \(e)"
            } else {
                self?.error = ""
            }
        }
        b.start(queue: .main)
        browser = b
    }

    func stop() {
        browser?.cancel()
        browser = nil
        macs = []
    }
}
