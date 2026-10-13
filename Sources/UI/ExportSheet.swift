import SwiftUI
import AVFoundation
import VideoToolbox

enum ExportCodec: String, CaseIterable, Identifiable {
    case proRes422HQ = "ProRes 422 HQ"
    case hevc10 = "HEVC 10-bit"
    case h264 = "H.264"
    var id: String { rawValue }

    var avCodec: AVVideoCodecType {
        switch self {
        case .proRes422HQ: return .proRes422HQ
        case .hevc10: return .hevc
        case .h264: return .h264
        }
    }
}

struct ExportSheet: View {
    @EnvironmentObject var project: ProjectModel
    @Environment(\.dismiss) private var dismiss
    @State private var codec: ExportCodec = .hevc10
    @State private var progress: Double = 0
    @State private var isExporting = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Picker("Codec", selection: $codec) {
                    ForEach(ExportCodec.allCases) { Text($0.rawValue).tag($0) }
                }
                if isExporting { ProgressView(value: progress) }
                if let error { Text(error).foregroundStyle(.red) }
                Button("Export") { Task { await run() } }.disabled(isExporting)
            }
            .navigationTitle("Export")
            .toolbar { Button("Close") { dismiss() } }
        }
        .presentationDetents([.medium])
    }

    private func run() async {
        isExporting = true; error = nil
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("BP-export").appendingPathExtension(codec == .proRes422HQ ? "mov" : "mp4")
        do {
            let engine = ExportEngine(width: project.composition.width, height: project.composition.height,
                                      fps: project.composition.frameRate, duration: project.composition.duration,
                                      codec: codec)
            try await engine.export(to: url) { p in Task { @MainActor in progress = p } }
        } catch { self.error = error.localizedDescription }
        isExporting = false
    }
}

// MARK: - AVAssetWriter engine (hardware-accelerated via VideoToolbox)

final class ExportEngine {
    let width: Int, height: Int, fps: Double, duration: Double, codec: ExportCodec

    init(width: Int, height: Int, fps: Double, duration: Double, codec: ExportCodec) {
        self.width = width; self.height = height; self.fps = fps; self.duration = duration; self.codec = codec
    }

    /// Supply rendered frames; replace `renderFrame` with the Metal compositor readback.
    var renderFrame: (_ frameIndex: Int, _ into: CVPixelBuffer) -> Void = { _, _ in }

    func export(to url: URL, progress: @escaping (Double) -> Void) async throws {
        try? FileManager.default.removeItem(at: url)
        let fileType: AVFileType = codec == .proRes422HQ ? .mov : .mp4
        let writer = try AVAssetWriter(outputURL: url, fileType: fileType)

        var settings: [String: Any] = [
            AVVideoCodecKey: codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        if codec == .hevc10 {
            settings[AVVideoCompressionPropertiesKey] = [
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                AVVideoAverageBitRateKey: 40_000_000
            ]
        } else if codec == .h264 {
            settings[AVVideoCompressionPropertiesKey] = [AVVideoAverageBitRateKey: 20_000_000]
        }

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false

        let pixelFormat: OSType = codec == .hevc10
            ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            : (codec == .proRes422HQ ? kCVPixelFormatType_422YpCbCr10 : kCVPixelFormatType_32BGRA)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ])
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? NSError(domain: "BP", code: 1) }
        writer.startSession(atSourceTime: .zero)

        let total = Int(duration * fps)
        let timescale = Int32(fps * 100)
        let frameDuration = Int64(100)

        for i in 0..<total {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            guard let pool = adaptor.pixelBufferPool else { throw NSError(domain: "BP", code: 2) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
            guard let buffer = pb else { throw NSError(domain: "BP", code: 3) }
            renderFrame(i, buffer)
            adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(i) * frameDuration, timescale: timescale))
            progress(Double(i + 1) / Double(total))
        }
        input.markAsFinished()
        await writer.finishWriting()
        if let e = writer.error { throw e }
    }
}
