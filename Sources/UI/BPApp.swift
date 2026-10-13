import SwiftUI
import MetalKit

// MARK: - App entry

@main
struct BPApp: App {
    @StateObject private var project = ProjectModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(project)
                .preferredColorScheme(.dark)
        }
    }
}

// MARK: - Project model

final class ProjectModel: ObservableObject {
    struct Composition {
        var name = "Comp 1"
        var width = 3840
        var height = 2160
        var frameRate = 30.0
        var duration = 10.0
    }

    struct Layer: Identifiable {
        let id = UUID()
        var name: String
        var inPoint: Double
        var outPoint: Double
        var is3D = false
    }

    @Published var composition = Composition()
    @Published var layers: [Layer] = [
        .init(name: "Footage", inPoint: 0, outPoint: 6),
        .init(name: "Title", inPoint: 1, outPoint: 5, is3D: true),
        .init(name: "Glow FX", inPoint: 0, outPoint: 10)
    ]
    @Published var playhead: Double = 0
    @Published var isPlaying = false
    @Published var showExport = false
    @Published var showNewProject = false
}

// MARK: - Root view

struct RootView: View {
    @EnvironmentObject var project: ProjectModel
    @State private var timelineExpanded = true
    @GestureState private var dragOffset: CGFloat = 0

    private let spring = Animation.interpolatingSpring(mass: 1, stiffness: 180, damping: 18)

    var body: some View {
        GeometryReader { geo in
            let drawerHeight = timelineExpanded ? geo.size.height * 0.45 : 64
            VStack(spacing: 0) {
                TopBar()
                ViewportCanvas()
                    .frame(maxHeight: .infinity)
                TimelineDrawer(expanded: $timelineExpanded)
                    .frame(height: max(64, drawerHeight - dragOffset))
                    .gesture(
                        DragGesture()
                            .updating($dragOffset) { value, state, _ in state = value.translation.height * -1 }
                            .onEnded { value in
                                withAnimation(spring) {
                                    if value.translation.height > 60 { timelineExpanded = false }
                                    if value.translation.height < -60 { timelineExpanded = true }
                                }
                            }
                    )
                    .animation(spring, value: timelineExpanded)
            }
            .background(Color.black.ignoresSafeArea())
        }
        .sheet(isPresented: $project.showExport) { ExportSheet() }
        .sheet(isPresented: $project.showNewProject) { NewProjectSheet() }
    }
}

struct TopBar: View {
    @EnvironmentObject var project: ProjectModel

    var body: some View {
        HStack {
            Button { project.showNewProject = true } label: { Image(systemName: "plus.square") }
            Spacer()
            Text(project.composition.name).font(.subheadline.weight(.semibold))
            Spacer()
            Button { project.showExport = true } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Metal viewport (triple-buffered)

struct ViewportCanvas: UIViewRepresentable {
    func makeCoordinator() -> Renderer { Renderer() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.framebufferOnly = false
        view.delegate = context.coordinator
        context.coordinator.configure(view: view)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

final class Renderer: NSObject, MTKViewDelegate {
    private var queue: MTLCommandQueue?
    private let inflight = DispatchSemaphore(value: 3)   // triple buffering

    func configure(view: MTKView) {
        queue = view.device?.makeCommandQueue()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let queue, let drawable = view.currentDrawable,
              let pass = view.currentRenderPassDescriptor,
              let cb = queue.makeCommandBuffer() else { return }
        inflight.wait()
        cb.addCompletedHandler { [inflight] _ in inflight.signal() }
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.06, alpha: 1)
        cb.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

// MARK: - Timeline drawer

struct TimelineDrawer: View {
    @EnvironmentObject var project: ProjectModel
    @Binding var expanded: Bool
    private let pxPerSecond: CGFloat = 80

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(.secondary).frame(width: 40, height: 4).padding(.vertical, 6)
            HStack {
                Button { project.isPlaying.toggle() } label: {
                    Image(systemName: project.isPlaying ? "pause.fill" : "play.fill")
                }
                Text(String(format: "%.2fs", project.playhead)).monospacedDigit()
                Spacer()
            }
            .padding(.horizontal, 16)

            if expanded {
                ScrollView([.horizontal, .vertical]) {
                    ZStack(alignment: .topLeading) {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(project.layers) { layer in
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(layer.is3D ? Color.purple.opacity(0.7) : Color.blue.opacity(0.7))
                                    .overlay(Text(layer.name).font(.caption).padding(.leading, 6), alignment: .leading)
                                    .frame(width: CGFloat(layer.outPoint - layer.inPoint) * pxPerSecond, height: 32)
                                    .offset(x: CGFloat(layer.inPoint) * pxPerSecond)
                            }
                        }
                        .padding(.top, 8)
                        Rectangle().fill(.red).frame(width: 2)
                            .offset(x: CGFloat(project.playhead) * pxPerSecond)
                            .gesture(DragGesture().onChanged { v in
                                project.playhead = max(0, min(project.composition.duration, Double(v.location.x / pxPerSecond)))
                            })
                    }
                    .frame(width: CGFloat(project.composition.duration) * pxPerSecond + 40, alignment: .topLeading)
                }
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
            Spacer(minLength: 0)
        }
        .background(.ultraThinMaterial)
    }
}

// MARK: - New project sheet

struct NewProjectSheet: View {
    @EnvironmentObject var project: ProjectModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                TextField("Name", text: $project.composition.name)
                Picker("Preset", selection: $project.composition.width) {
                    Text("1080p").tag(1920)
                    Text("4K UHD").tag(3840)
                }
                Picker("Frame rate", selection: $project.composition.frameRate) {
                    ForEach([24.0, 30.0, 60.0], id: \.self) { Text("\(Int($0)) fps").tag($0) }
                }
            }
            .navigationTitle("New Composition")
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}
