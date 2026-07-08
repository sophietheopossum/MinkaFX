//! MinkaFX — the Guido-style overlay half of the Minka hybrid shell.
//!
//! Per-frame-hot, geometrically simple, inputless surfaces rendered with
//! wgpu on wlr-layer-shell overlay layers, one per output. Consumes ShojiWM
//! broadcasts through MinkaIPC (thread + calloop channel, so the render
//! loop never blocks on the socket — rule R1).
//!
//! Currently owned: the snap-zone preview (`snap.preview` broadcast:
//! `{ monitor, rect: {x,y,w,h} | null, kind: "floating"|"tiling" }`).
//! The compositor transmits state, not frames (rule R3): the target rect
//! arrives here and springs + fades run client-side at frame-callback pace.
//!
//! Surfaces take no input (empty input region) and never grab the keyboard,
//! so a crash or hang here can never wedge the session.

use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use calloop::channel::Event as ChannelEvent;
use calloop::EventLoop;
use calloop_wayland_source::WaylandSource;
use minka_ipc::{IpcClient, IpcEvent};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState, Region};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
    LayerSurfaceConfigure,
};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::{
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, registry_handlers,
};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{wl_output, wl_surface};
use wayland_client::{Connection, Proxy, QueueHandle};

// Eternal Darkness tokens (Theme.qml is the source of truth).
// #e0263c
const BORDER_RGB: [f32; 3] = [0.878, 0.149, 0.235];
// #8f1e2d
const FILL_RGB: [f32; 3] = [0.561, 0.118, 0.176];
const FILL_ALPHA_FLOATING: f32 = 0.16;
const FILL_ALPHA_TILING: f32 = 0.26;
const BORDER_ALPHA: f32 = 0.9;
const BORDER_WIDTH: f32 = 2.0;
const CORNER_RADIUS: f32 = 10.0;

// Slightly underdamped rect spring (a whisper of overshoot, zephyr-style);
// critically damped fade so alpha never overshoots past 1.
const RECT_STIFFNESS: f32 = 700.0;
// 2*sqrt(700) ~ 52.9 => damping ratio ~0.85
const RECT_DAMPING: f32 = 45.0;
const FADE_STIFFNESS: f32 = 900.0;
// critical
const FADE_DAMPING: f32 = 60.0;

fn main() {
    env_logging();
    let conn = Connection::connect_to_env()
        .expect(
            "wayland connection"
        );
    let (
        globals,
        event_queue,
    ) = registry_queue_init::<App>(
        &conn,
    ).expect(
        "registry init",
    );
    let qh: QueueHandle<App> = event_queue
        .handle();

    let compositor =
        CompositorState::bind(
            &globals,
            &qh,
        ).expect(
            "wl_compositor not available",
        );
    let layer_shell = LayerShell::bind(
        &globals,
        &qh,
    ).expect(
        "layer shell not available",
    );

    let (
        ipc,
        ipc_events,
    ) = IpcClient::spawn()
        .expect(
            "ipc thread",
        );

    let mut event_loop: EventLoop<App> = EventLoop::try_new()
        .expect(
            "event loop",
        );
    WaylandSource::new(
        conn
            .clone(),
        event_queue,
    ).insert(
        event_loop
            .handle(),
    ).expect(
        "wayland source",
    );
    event_loop
        .handle()
        .insert_source(
            ipc_events,
            |event,
             _,
             app| {
                if let ChannelEvent::Msg(msg) = event { app.handle_ipc(msg); }
            }
        ).expect(
        "ipc source",
    );

    let mut app = App {
        conn,
        qh,
        registry_state: RegistryState::new(
            &globals
        ),
        output_state: OutputState::new(
            &globals,
            &event_loop
                .handle()
                .clone()
                .into(),
        ),
        compositor,
        layer_shell,
        instance: wgpu::Instance::new(
            &wgpu::InstanceDescriptor::default()
        ),
        gpu: None,
        overlays: Vec::new(),
        _ipc: ipc,
    };

    // registry_queue_init already delivered globals; OutputState needs the
    // real queue handle, not a loop handle — fixed below (see App::new note).
    eprintln!("[MinkaFX] up; waiting for outputs and snap.preview broadcasts");

    loop {
        event_loop
            .dispatch(
                None,
                &mut app,
            )
            .expect(
                "event loop dispatch",
            );
    }
}

fn env_logging() {
    // Timestamps make /tmp/minkafx.log correlatable with minkashell.log.
    eprintln!(
        "[minka-fx] start {:?}",
        std::time::SystemTime::now(),
    );
}

// ---------------------------------------------------------------------------

struct SharedGpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct OverlayGpu {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

#[derive(
    Clone,
    Copy,
    Debug
)]
struct Spring {
    value: f32,
    velocity: f32,
}

impl Spring {
    fn new(
        value: f32
    ) -> Self {

        Spring {
            value,
            velocity: 0.0,
        }
    }

    fn step(
        &mut self,
        target: f32,
        dt: f32,
        stiffness: f32,
        damping: f32,
    ) {
        let accel = stiffness * (target - self.value) - damping * self.velocity;
        self.velocity += accel * dt;
        self.value += self.velocity * dt;
    }

    fn snap(
        &mut self,
        value: f32,
    ) {
        self.value = value;
        self.velocity = 0.0;
    }

    fn settled(
        &self,
        target: f32,
        epsilon: f32,
    ) -> bool {
        (self.value - target).abs() < epsilon && self.velocity.abs() < epsilon * 10.0
    }
}

struct Overlay {
    // Declared before `layer` so the wgpu surface drops before the
    // wl_surface it borrows.
    gpu: Option<OverlayGpu>,
    output: wl_output::WlOutput,
    connector: Option<String>,
    layer: LayerSurface,
    _input_region: Region,
    width: u32,
    height: u32,
    scale: i32,
    configured: bool,

    // Animation state, logical coordinates.
    target: Option<[f32; 4]>,
    tiling: bool,
    x: Spring,
    y: Spring,
    w: Spring,
    h: Spring,
    alpha: Spring,
    animating: bool,
    last_frame: Option<Instant>,
}

struct App {
    conn: Connection,
    qh: QueueHandle<App>,
    registry_state: RegistryState,
    output_state: OutputState,
    compositor: CompositorState,
    layer_shell: LayerShell,
    instance: wgpu::Instance,
    gpu: Option<SharedGpu>,
    overlays: Vec<Overlay>,
    _ipc: Arc<IpcClient>,
}

impl App {
    fn add_output(
        &mut self,
        output: wl_output::WlOutput
    ) {
        let connector = self
            .output_state
            .info(
                &output
            )
            .and_then(|info| info.name);
        let surface = self.compositor
            .create_surface(
                &self.qh,
            );
        let layer = self.layer_shell
            .create_layer_surface(
                &self.qh,
                surface,
                Layer::Overlay,
                Some("minka-fx"),
                Some(&output),
            );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        // span the full output, over exclusive zones
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(
            KeyboardInteractivity::None
        );
        layer.set_size(
            0,
            0,
        );

        // Inputless by construction: pointer and touch pass straight through.
        let region = Region::new(&self.compositor).expect("wl_region");
        layer
            .wl_surface()
            .set_input_region(Some(region.wl_region()));
        layer.commit();

        eprintln!(
            "[minka-fx] overlay created for output {:?}",
            connector.as_deref().unwrap_or("<unnamed>")
        );
        self.overlays.push(Overlay {
            gpu: None,
            output,
            connector,
            layer,
            _input_region: region,
            width: 0,
            height: 0,
            scale: 1,
            configured: false,
            target: None,
            tiling: false,
            x: Spring::new(0.0),
            y: Spring::new(0.0),
            w: Spring::new(0.0),
            h: Spring::new(0.0),
            alpha: Spring::new(0.0),
            animating: false,
            last_frame: None,
        });
    }

    fn handle_ipc(
        &mut self,
        event: IpcEvent,
    ) {
        match event {
            IpcEvent::Connected => eprintln!(
                "[minka-fx] ipc connected",
            ),
            IpcEvent::Disconnected => eprintln!(
                "[minka-fx] ipc lost, retrying",
            ),
            IpcEvent::Broadcast { event, payload } if event == "snap.preview" => {
                let Some(monitor) = payload.get("monitor").and_then(|v| v.as_str()) else {
                    return;
                };
                let monitor = monitor.to_string();
                let rect = payload.get("rect").and_then(|r| {
                    Some([
                        r.get("x")?.as_f64()? as f32,
                        r.get("y")?.as_f64()? as f32,
                        r.get("w")?.as_f64()? as f32,
                        r.get("h")?.as_f64()? as f32,
                    ])
                });
                let tiling = payload.get("kind").and_then(|v| v.as_str()) == Some(
                    "tiling",
                );
                self
                    .apply_snap_preview(
                        &monitor,
                        rect,
                        tiling,
                    );
            }
            IpcEvent::Broadcast { .. } | IpcEvent::Response { .. } => {}
        }
    }

    fn apply_snap_preview(
        &mut self,
        monitor: &str,
        rect: Option<[f32; 4]>,
        tiling: bool,
    ) {
        let qh = self.qh
            .clone();
        let Some(index) = self
            .overlays
            .iter()
            .position(|o| o.connector.as_deref() == Some(monitor))
        else {
            return;
        };

        {
            let overlay = &mut self.overlays[index];
            overlay.tiling = tiling;
            if let Some(rect) = rect {
                // Appearing from nothing: materialize at the target and only
                // fade, instead of flying in from a stale rect.
                if overlay.alpha.value < 0.02 && overlay.target.is_none() {
                    overlay.x.snap(rect[0]);
                    overlay.y.snap(rect[1]);
                    overlay.w.snap(rect[2]);
                    overlay.h.snap(rect[3]);
                }
                overlay.target = Some(rect);
            } else {
                overlay.target = None; // fade out in place
            }
        }

        // Kick the frame loop if it is idle.
        if !self.overlays[index].animating && self.overlays[index].configured {
            self
                .render_overlay(
                    index,
                    &qh,
                );
        }
    }

    fn ensure_shared_gpu(
        &mut self,
        surface: &wgpu::Surface<'static>,
    ) -> &SharedGpu {
        if self.gpu.is_none() {
            let adapter = pollster::block_on(self.instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: Some(surface),
                    force_fallback_adapter: false,
                },
            ))
            .expect("no wgpu adapter");
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("minka-fx"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    ..Default::default()
                },
            ))
            .expect("wgpu device");
            eprintln!("[minka-fx] gpu: {}", adapter.get_info().name);
            self.gpu = Some(
                SharedGpu {
                    adapter,
                    device,
                    queue,
                }
            );
        }
        self.gpu
            .as_ref()
            .unwrap()
    }

    fn init_overlay_gpu(&mut self, index: usize) {
        let raw_display_handle = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
            NonNull::new(self.conn.backend().display_ptr() as *mut _).unwrap(),
        ));
        let wl_surface = self.overlays[index].layer
            .wl_surface()
            .clone();
        let raw_window_handle = RawWindowHandle::Wayland(WaylandWindowHandle::new(
            NonNull::new(wl_surface.id().as_ptr() as *mut _).unwrap(),
        ));
        let surface = unsafe {
            self.instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle,
                    raw_window_handle,
                })
        }
        .expect(
            "wgpu surface",
        );

        let shared = self
            .ensure_shared_gpu(
                &surface,
            );
        let caps = surface.get_capabilities(
            &shared.adapter,
        );
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            .unwrap_or(caps.formats[0]);
        let alpha_mode = if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            caps.alpha_modes[0]
        };

        let overlay = &self.overlays[index];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: overlay.width * overlay.scale as u32,
            height: overlay.height * overlay.scale as u32,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&shared.device, &config);

        let device = &shared.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(
                "snap-preview",
            ),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(
                    "shader.wgsl",
                ).into(),
            ),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                "snap-preview-uniforms",
            ),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(
                "snap-preview",
            ),
            layout: Some(
                &pipeline_layout,
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(
                    "vs_main",
                ),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(
                    "fs_main",
                ),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Single draw over a transparent clear; premultiplied
                    // output is written as-is.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.overlays[index].gpu = Some(OverlayGpu {
            surface,
            config,
            pipeline,
            bind_group,
            uniforms,
        });
    }

    fn render_overlay(&mut self, index: usize, qh: &QueueHandle<App>) {
        let Some(shared) = self.gpu.as_ref() else { return };
        let overlay = &mut self.overlays[index];
        let Some(gpu) = overlay.gpu.as_ref() else { return };

        let now = Instant::now();
        let dt = overlay
            .last_frame
            .map(|t| (now - t).as_secs_f32().clamp(0.0001, 0.05))
            .unwrap_or(1.0 / 60.0);
        overlay.last_frame = Some(now);

        // Step springs toward the target (or fade out toward alpha 0).
        let (alpha_target, rect_target) = match overlay.target {
            Some(rect) => (1.0, rect),
            None => (0.0, [overlay.x.value, overlay.y.value, overlay.w.value, overlay.h.value]),
        };
        overlay.x.step(rect_target[0], dt, RECT_STIFFNESS, RECT_DAMPING);
        overlay.y.step(rect_target[1], dt, RECT_STIFFNESS, RECT_DAMPING);
        overlay.w.step(rect_target[2], dt, RECT_STIFFNESS, RECT_DAMPING);
        overlay.h.step(rect_target[3], dt, RECT_STIFFNESS, RECT_DAMPING);
        overlay.alpha.step(alpha_target, dt, FADE_STIFFNESS, FADE_DAMPING);
        overlay.alpha.value = overlay.alpha.value
            .clamp(
                0.0,
                1.0,
            );

        let settled = overlay.alpha.settled(alpha_target, 0.004)
            && overlay.x.settled(rect_target[0], 0.3)
            && overlay.y.settled(rect_target[1], 0.3)
            && overlay.w.settled(rect_target[2], 0.3)
            && overlay.h.settled(rect_target[3], 0.3);
        let idle = settled && overlay.target.is_none();
        if idle {
            overlay.alpha.snap(0.0);
        }

        let scale = overlay.scale as f32;
        let fill_alpha = if overlay.tiling {
            FILL_ALPHA_TILING
        } else {
            FILL_ALPHA_FLOATING
        };
        let data: [f32; 16] = [
            overlay.x.value * scale,
            overlay.y.value * scale,
            overlay.w.value * scale,
            overlay.h.value * scale,
            BORDER_RGB[0],
            BORDER_RGB[1],
            BORDER_RGB[2],
            BORDER_ALPHA,
            FILL_RGB[0],
            FILL_RGB[1],
            FILL_RGB[2],
            fill_alpha,
            CORNER_RADIUS * scale,
            overlay.alpha.value,
            BORDER_WIDTH * scale,
            0.0,
        ];
        let mut bytes = [0u8; 64];
        for (i, v) in data.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
        }
        shared.queue.write_buffer(&gpu.uniforms, 0, &bytes);

        let frame = match gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(err) => {
                eprintln!("[minka-fx] get_current_texture: {err:?}");
                gpu.surface.configure(&shared.device, &gpu.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if overlay.alpha.value > 0.003 {
                pass.set_pipeline(
                    &gpu.pipeline
                );
                pass
                    .set_bind_group(
                        0,
                        &gpu.bind_group,
                        &[],
                    );
                pass
                    .draw(
                        0..3,
                        0..1,
                    );
            }
        }

        // Keep animating? Request the next frame callback BEFORE the commit
        // that wgpu's present() performs, so it rides the same commit.
        overlay.animating = !idle;
        if overlay.animating {
            overlay
                .layer
                .wl_surface()
                .frame(
                    qh,
                    overlay.layer
                        .wl_surface()
                        .clone(),
                );
        } else {
            overlay.last_frame = None;
        }

        shared.queue.submit(
            Some(
                encoder
                    .finish()
            )
        );
        frame.present();
    }

    fn overlay_index_for_surface(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
        self.overlays
            .iter()
            .position(|o| o.layer.wl_surface() == surface)
    }
}

// ---------------------------------------------------------------------------
// smithay-client-toolkit plumbing

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        let Some(index) = self.overlay_index_for_surface(surface) else { return };
        let overlay = &mut self.overlays[index];
        if overlay.scale == new_factor {
            return;
        }
        overlay.scale = new_factor;
        overlay.layer
            .wl_surface()
            .set_buffer_scale(
                new_factor,
            );
        if let Some(gpu) = overlay.gpu.as_mut() {
            gpu.config.width = overlay.width * new_factor as u32;
            gpu.config.height = overlay.height * new_factor as u32;
            if let Some(shared) = self.gpu.as_ref() {
                gpu.surface
                    .configure(
                        &shared.device,
                        &gpu.config,
                    );
            }
        }
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        let qh = qh
            .clone();
        if let Some(index) = self.overlay_index_for_surface(surface) {
            if self.overlays[index].animating {
                self
                    .render_overlay(
                        index,
                        &qh,
                    );
            }
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.add_output(output);
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        // Connector names arrive after bind; refresh them.
        let name = self.output_state.info(&output).and_then(|info| info.name);
        if let Some(overlay) = self.overlays.iter_mut().find(|o| o.output == output) {
            if overlay.connector != name {
                eprintln!(
                    "[minka-fx] output named {:?}",
                    name
                        .as_deref()
                        .unwrap_or(
                            "<unnamed>",
                        )
                );
                overlay.connector = name;
            }
        }
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.overlays.retain(|o| o.output != output);
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, layer: &LayerSurface) {
        self.overlays.retain(|o| &o.layer != layer);
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let qh = qh
            .clone();
        let Some(index) = self
            .overlays
            .iter()
            .position(|o| &o.layer == layer)
        else {
            return;
        };
        let (width, height) = configure.new_size;
        let overlay = &mut self.overlays[index];
        let resized = overlay.width != width || overlay.height != height;
        overlay.width = width.max(1);
        overlay.height = height.max(1);
        let first = !overlay.configured;
        overlay.configured = true;

        if first {
            self.init_overlay_gpu(
                index,
            );
        } else if resized {
            let overlay = &mut self.overlays[index];
            if let Some(gpu) = overlay.gpu.as_mut() {
                gpu.config.width = overlay.width * overlay.scale as u32;
                gpu.config.height = overlay.height * overlay.scale as u32;
                if let Some(shared) = self.gpu.as_ref() {
                    gpu.surface.configure(&shared.device, &gpu.config);
                }
            }
        }
        // A configure demands a commit with content; draw the current state
        // (usually fully transparent).
        self.render_overlay(index, &qh);
    }
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_compositor!(App);
delegate_output!(App);
delegate_layer!(App);
delegate_registry!(App);
