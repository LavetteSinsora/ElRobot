import AVFoundation
import AppKit
let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1]); let out = args[2]; let n = Int(args[3])!
let asset = AVURLAsset(url: url)
let dur = CMTimeGetSeconds(asset.duration)
print("duration", dur)
let gen = AVAssetImageGenerator(asset: asset)
gen.appliesPreferredTrackTransform = true
gen.maximumSize = CGSize(width: 1000, height: 1000)
gen.requestedTimeToleranceBefore = .zero; gen.requestedTimeToleranceAfter = .zero
for i in 0..<n {
  let t = CMTime(seconds: dur * Double(i) / Double(n - 1) * 0.98, preferredTimescale: 600)
  if let cg = try? gen.copyCGImage(at: t, actualTime: nil) {
    let rep = NSBitmapImageRep(cgImage: cg)
    let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8])!
    try! data.write(to: URL(fileURLWithPath: "\(out)/f\(String(format: "%02d", i)).jpg"))
  }
}
